#ifndef __DECLARATIONS_HPP
#define __DECLARATIONS_HPP

void cuda_surf(char *filename);
void cpu_surf(char *fileName);
void opencl_surf(char *filename);
void cpu_orb(char *filename);
void opencl_orb(char *filename);
void cuda_orb(char *filename);
void cpu_brisk(char *filename);
void cpu_harris_brief(char *filename);
void cpu_harris_freak(char *filename);
void cpu_shi_tomasi_brief(char *filename);
void cpu_shi_tomasi_freak(char *filename);
void cpu_star_brief(char *filename);
void cpu_star_freak(char *filename);
void cpu_fast_freak(char *filename);

#endif
