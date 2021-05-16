#include "diploma.hpp"
#include "declarations.hpp"
#include <map>

int main(int argc, char *argv[]) {
	if(argc < 3) {	
		std::cout << "usage: " << argv[0] << " video.mp4 alg_type" << std::endl;
		return 0;
	}

	std::string algo(argv[2]);

	std::map<std::string, fea_det> detectors = {
		{ "cpu_surf", cpu_surf },
		{ "opencl_surf", opencl_surf },
		{ "cuda_surf", cuda_surf },
		{ "cpu_orb", cpu_orb },
		{ "opencl_orb", opencl_orb },
		{ "cuda_orb", cuda_orb },
		{ "cpu_brisk", cpu_brisk },
		{ "cpu_harris_brief", cpu_harris_brief },
		{ "cpu_harris_freak", cpu_harris_freak },
		{ "cpu_shi_tomasi_brief", cpu_shi_tomasi_brief },
		{ "cpu_shi_tomasi_freak", cpu_shi_tomasi_freak },
		{ "cpu_star_brief", cpu_star_brief },
		{ "cpu_star_freak", cpu_star_freak },
		{ "cpu_fast_freak", cpu_fast_freak }
	};

	if(detectors[algo] == nullptr) {
		std::cout << "no such algorithm" << std::endl;
		return 0;
	}

	detectors[algo](argv[1]);
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
//	cpu_fast_freak(argv[1]);

	return 0;
}

