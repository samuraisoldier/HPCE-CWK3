Now we need to fix the kernel, as currently it has dependencies on
events that no longer exist, so remove the input dependency list
and the output event. You'll now have a loop which just enqueues
kernels, without knowing when they have executed. The problem
is that there is no fixed ordering between the kernels: the run-time
could queue up 10 or 20 calls to enqueue kernel, then issue them
all in parallel. So we need one kernel to finish before the next one starts,
in order to make sure the output buffer is completely written
before the next kernel call uses it as an input buffer.

There are multiple solutions to this, but the easiest is to add a call to
`enqueueBarrierWithWaitList` just after the kernel is enqueue'd:

	queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
		
	queue.enqueueBarrierWithWaitList();	// <- new barrier here
	
	std::swap(buffState, buffBuffer);

This creates a synchronisation point within the command queue,
so anything before the barrier must complete before anything
after the barrier can start. However, it doesn't say anything
about the relationship with the C++ program. It is entirely
possible that the host program could queue up hundreds of
kernel calls and barriers before the OpenCL run-time gets round
to executing them. Indeed, it may only be the synchronised
memory copy _after_ the loop that forces the chain of
kernels and barriers to run at all.

So the inner loop should now look something like:

    kernel.setArg(3, buffState);
	kernel.setArg(4, buffBuffer);
	queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
		
	queue.enqueueBarrierWithWaitList();
	
	std::swap(buffState, buffBuffer);

The kernel code itself is identical to that for the previous
implementation, so you don't need to create a new .cl file. Sometimes
you can optimise execution purely by modifying the host code,
and other times tweaks to the kernel code are all that is needed.

A final problem is that our memory buffers were originally
declared with read-only and write-only flags, but now we
are violating that requirements. Go back to your buffer
declarations, and modify them to:

	cl::Buffer buffState(context, CL_MEM_READ_WRITE, cbBuffer);
	cl::Buffer buffBuffer(context, CL_MEM_READ_WRITE, cbBuffer);
	
There is a lot going on here, so to summarise what the program should
now look like, it should be something like:

``` text
Create buffProperties : read only
Create buffState, buffBuffer : read/write

Create kernel
Set inner, outer, and properties arguments

write properties data to buffProperties

write initial state to buffState

for t = 0 .. n-1 begin
   Set state argument to buffState
   Set output argument to buffBuffer
   
   enqueue kernel
   
   barrier
   
   swap(buffState, buffBuffer);
end

read buffState into output
```
	
You may find it confusing due to the abstraction of
the `cl::Buffer`, but it helps if you think of them
as a special kind of pointer. For example, imagine
`buffState` and `buffBuffer` have notional addresses
0x4000 and 0x8000 on the GPU. Then what happens
for n=2 is:

    buffState = (void*)0x4000;
	buffBuffer = (void*)0x8000;
	
	memcpy(originalState , buffState, ...);
	
	///////////////////////////////////////////////////

	// loop iteration t=0
	// buffState==0x4000, buffBuffer==0x8000
	
	// Reading from 0x4000, writing to 0x8000
	kernel(..., input = buffState (0x4000), output = buffBuffer (0x8000))
	
	swap(buffState, buffBuffer); // swaps the values of the pointers
	
    //////////////////////////////////////////////////

	// loop iteration t=1
	// buffState==0x8000, buffBuffer==0x4000
	
	// reading from 0x8000, writing to 0x4000
	kernel(..., input = buffState (0x8000), output = buffBuffer (0x4000))
	
	swap(buffState, buffBuffer);

	//////////////////////////////////////////////////
	
	// After loop
	// buffState==0x4000, buffBuffer==0x8000
	
	memcpy(buffState, outputState, ...);
	
I highly recommend you try this with n=1, n=2, and n=3,
as it is easy to think this is working for n=1 when it fails
for larger numbers.

Depending on your platform, you may now start to see a reasonable
speed-up over software (though maybe still not over TBB - depends
a lot on the hardware).


Optimising global to GPU memory accesses (v5)
=============================================

One of the biggest problems in GPU programming is managing the different memory
spaces. Slight differences in memory layout can cause large changes in
execution time, while moving arrays and variables from global to private
memory can have huge performance implications.
In general, the best memory accesses are those which never happen, so
it is worth-while trying to optimise them out. GPU compilers can be
more conservative then CPU compilers, so it is a good idea to help them
out.

Looking at our current kernel, we can see that there are five reads to
the `properties` array for a normal cell (non insulator). However, four
of those reads are getting very little information back, as we only depend
on a single bit of information from the four neighbours. A good approach
in such scenarios is to try to pack data into spare bit-fields, increasing
the information density and reducing the number of memory reads. In
this case, we already have one 32-bit integer describing the properties,
of which only 2 bits are currently being used. So as well as the 2 bits describing
the properties of the current cell, we could quite reasonably include four
bits describing whether the four neigbouring cells are insulators or not,
saving a memory access.

This requires two modifications: one in the host code to set up the more
complex flags, and another in the kernel code to take advantage of the
bits.

### Kernel code

Create a new kernel called `src/your_login/step_world_v5_packed_properties.cl`
based on the v4 kernel.

At the top of the code, read the properties for the current cell into
a private uint variable. This value will be read once into fast private
memory, and then from then on it can be accessed very cheaply.
For each of the branches, rewrite it to check bit-flags in the
variable. For example, if your private properties variable is called `myProps`,
and you decided bit 3 of the properties indicated whether the cell
above is an insulator, the first branch could be re-written to:

	// Cell above
	if(myProps & 0x4) {
		contrib += outer;
		acc += outer * world_state[index-w];
	}

The other neighbours will need to depend on different bits in the properties.

### Host code

Create a new implementation called `src/your_login/step_world_v5_packed_properties.cpp`
based on the v4 host code.

In the host code, you need to make sure the the correct flags have
been inserted into the properties buffer. This should only have local
effect, so we cannot modify `world.properties` directly. Instead
create a temporary array in host memory:
	
	std::vector<uint32_t> packed(w*h, 0);

and fill it with the appropriate bits. This will involve looping over all
the co-ordinates, using the following process at each (x,y) co-ordinate:

    packed(x,y) = world.properties(x,y)
    if world.properties(x,y) is normal:
        if world.properties(x,y-1) is insulator:
            packed(x,y) = packed(x,y) + 4
        if world.properties(x,y+1) is insulator:
            packed(x,y) = packed(x,y) + 8
        # Handle left and right cases

This process takes some time (though it could be parallellised), but we
don't care too much, as it only happens once for multiple time iterations.
Once the array is prepared, it can be transferred to the
GPU instead of the properties array.

This got my laptop up to about 2.5x speedup, so faster than the two
cores in my device can go. On an AWS g2 GPU instance I looked at a 5000x5000
grid, stepped over 1000 time steps, which is more at the resolution
we typically use:

	time (bin/make_world 5000 0.1 1 | bin/step_world 0.1 1000 1 > /dev/null)
	time (bin/make_world 5000 0.1 1 | bin/dt10/step_world_v3_opencl 0.1 1000 1 > /dev/null)
	time (bin/make_world 5000 0.1 1 | bin/dt10/step_world_v4_double_buffered 0.1 1000 1 > /dev/null)
	time (bin/make_world 5000 0.1 1 | bin/dt10/step_world_v5_packed_properties 0.1 1000 1 > /dev/null)
	
For my code, I found:

Method     | time (secs)   | speedup (total) | speedup (incremental)
-----------|---------------|-----------------|---------------------
software   |         164.6 |           1.0   |                  1.0
opencl     |          66.2 |           2.5   |                  2.5
doublebuff |           9.0 |          18.3   |                  7.5
packing    |           6.5 |          25.3   |                  1.4
	

I strongly encourage you to also try the software OpenCL provider.
The AWS GPU instance has both a GPU and software provider installed,
and you can use `HPCE_SELECT_PLATFORM` to choose which one you
want. The GPU instance only has 16 cores, and the code is not
optimised for CPU-based OpenCL providers, but it should still be
less than half the time of the original software.