__kernel void kernel_xy(
			float inner,	// 0
			float outer,	// 1
			__global const uint *world_properties,	// 2
			__global const float *world_state,	// 3
			__global float *buffer	// 4
		)
{
	

	
	uint x=get_global_id(0);
	uint y=get_global_id(1);
	uint w=get_global_size(0);
	
	unsigned index=y*w + x;

	
	enum cell_flags_t{
		Cell_Fixed		=0x1,
		Cell_Insulator	=0x2
	};

	uint myProps = world_properties[index];	
	
	// Cell above
	//if(myProps & 0x4) {
	//	contrib += outer;
		//acc += outer * world_state[index-w];
	//}

	
	if((myProps & Cell_Fixed) || (myProps & Cell_Insulator)){
		// Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
		buffer[index]=world_state[index];
	}else{
		float contrib=inner;
		float acc=inner*world_state[index];
		
		
		// Cell above
		if(myProps & 0x4) {
			contrib += outer;
			acc += outer * world_state[index-w];
		}
		
		
		// Cell below
		if(myProps & 0x8) {
			contrib += outer;
			acc += outer * world_state[index+w];
		}
		
		
		// Cell left
		if(myProps & 0x10)) {
			contrib += outer;
			acc += outer * world_state[index-1];
		}
		
		
		// Cell right
		if(myProps & 0x20) {
			contrib += outer;
			acc += outer * world_state[index+1];
		}
		
		// Scale the accumulate value by the number of places contributing to it
		float res=acc/contrib;
		// Then clamp to the range [0,1]
		res=min(1.0f, max(0.0f, res));
		buffer[index] = res;
		
	} // end of if(insulator){ ... } else {	
	
} //end of kernel_xy fn