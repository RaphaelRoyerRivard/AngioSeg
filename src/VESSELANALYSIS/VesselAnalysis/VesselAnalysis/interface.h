#ifndef INTERFACE_H
#define INTERFACE_H

#ifdef __cplusplus
extern "C" {
__declspec(dllexport) int vesselanalysis_getskeleton(unsigned char *seg, int nrow, int ncol, unsigned char *skeleton);
__declspec(dllexport) int va_spur_pruning(unsigned char *img, int nrow, int ncol, int n, unsigned char *dest);
__declspec(dllexport) int vesselanalysis_getcomponents(unsigned char *skel,  int nrow, int ncol, int *nbcomponents, int *vessels_component);
__declspec(dllexport) int vesselanalysis_getstats(unsigned char *cc, unsigned char *dft, unsigned char *skel, int nrow, int ncol, int nbelem,  int nbcomponents, unsigned char *buffer);
#ifdef __cplusplus
}
#endif
#endif

#endif