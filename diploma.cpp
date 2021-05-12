#include "diploma.hpp"
#include "declarations.hpp"

int main(int argc, char *argv[]) {
	if(argc < 2) {	
		std::cout << "usage: " << argv[0] << " video.mp4" << std::endl;
		return 0;
	}

//	cpu_surf(argv[1]);
//	opencl_surf(argv[1]);
//	cuda_surf(argv[1]);
//	cpu_orb(argv[1]);
//	opencl_orb(argv[1]);
//	cuda_orb(argv[1]);
//	cpu_brisk(argv[1]);
//	cpu_harris_brief(argv[1]);
//	cpu_harris_freak(argv[1]);
//	cpu_shi_tomasi_brief(argv[1]);
//	cpu_shi_tomasi_freak(argv[1]);
//	cpu_star_brief(argv[1]);
//	cpu_star_freak(argv[1]);
	cpu_fast_freak(argv[1]);

	

	return 0;
}

