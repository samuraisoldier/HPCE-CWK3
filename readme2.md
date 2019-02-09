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