#include "heat.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <string>

namespace hpce{
namespace hs2715{	

//! Reference world stepping program
/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
	\param n Number of times to step the world
	\note Overall time increment will be n*dt
*/
void StepWorldV1Lambda(world_t &world, float dt, unsigned n)
{
	unsigned w=world.w, h=world.h;
	
	float outer=world.alpha*dt;		// We spread alpha to other cells per time
	float inner=1-outer/4;				// Anything that doesn't spread stays
	
	// This is our temporary working space
	std::vector<float> buffer(w*h);
	
	auto kernel_xy = [&](unsigned x, unsigned y){
		
		unsigned index=y*w + x;
		
		if((world.properties[index] & Cell_Fixed) || (world.properties[index] & Cell_Insulator)){
			// Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
			buffer[index]=world.state[index];
		}else{
			float contrib=inner;
			float acc=inner*world.state[index];
			
			// Cell above
			if(! (world.properties[index-w] & Cell_Insulator)) {
				contrib += outer;
				acc += outer * world.state[index-w];
			}
			
			// Cell below
			if(! (world.properties[index+w] & Cell_Insulator)) {
				contrib += outer;
				acc += outer * world.state[index+w];
			}
			
			// Cell left
			if(! (world.properties[index-1] & Cell_Insulator)) {
				contrib += outer;
				acc += outer * world.state[index-1];
			}
			
			// Cell right
			if(! (world.properties[index+1] & Cell_Insulator)) {
				contrib += outer;
				acc += outer * world.state[index+1];
			}
			
			// Scale the accumulate value by the number of places contributing to it
			float res=acc/contrib;
			// Then clamp to the range [0,1]
			res=std::min(1.0f, std::max(0.0f, res));
			buffer[index] = res;
			
		} // end of if(insulator){ ... } else {
		
	}; //end of lambda kernel_xy
	
	for(unsigned t=0;t<n;t++){
		for(unsigned y=0;y<h;y++){
			for(unsigned x=0;x<w;x++){
				
				kernel_xy(x,y);
				
				

			}  // end of for(x...
		} // end of for(y...
		
		// All cells have now been calculated and placed in buffer, so we replace
		// the old state with the new state
		std::swap(world.state, buffer);
		// Swapping rather than assigning is cheaper: just a pointer swap
		// rather than a memcpy, so O(1) rather than O(w*h)
	
		world.t += dt; // We have moved the world forwards in time
		
	} // end of for(t...
}

}; // namepspace hs2715	
}; // namepspace hpce

int main(int argc, char *argv[])
{
	float dt=0.1;
	unsigned n=1;
	bool binary=false;
	
	if(argc>1){
		dt=(float)strtod(argv[1], NULL);
	}
	if(argc>2){
		n=atoi(argv[2]);
	}
	if(argc>3){
		if(atoi(argv[3]))
			binary=true;
	}
	
	try{
		hpce::world_t world=hpce::LoadWorld(std::cin);
		std::cerr<<"Loaded world with w="<<world.w<<", h="<<world.h<<std::endl;
		
		std::cerr<<"Stepping by dt="<<dt<<" for n="<<n<<std::endl;
		hpce::hs2715::StepWorldV1Lambda(world, dt, n);
		
		hpce::SaveWorld(std::cout, world, binary);
	}catch(const std::exception &e){
		std::cerr<<"Exception : "<<e.what()<<std::endl;
		return 1;
	}
		
	return 0;
}

