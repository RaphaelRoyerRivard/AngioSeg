#ifndef GPU4_CUH
#define GPU4_CUH

#include <stdint.h>
#include "lspbmp.hpp"

void and_reduction(uint8_t* g_equ_data, int g_size, dim3 grid_dim, dim3 block_dim);
__global__ void and_reduction(uint8_t* g_data, int g_size);
__device__ uint8_t black_neighbors_around(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t block_and_reduce(uint8_t* s_data);
__device__ uint8_t border_global_mem_read(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height);
__device__ uint8_t is_outside_image(int g_row, int g_col, int g_width, int g_height);
__device__ uint8_t is_white(uint8_t* s_src, int s_src_row, int s_src_col, int s_src_width, uint8_t* s_equ, int s_equ_col);
__device__ void load_s_src(uint8_t* g_src, int g_row, int g_col, int g_width, int g_height, uint8_t* s_src, int s_row, int s_col, int s_width);
__device__ uint8_t P2_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P3_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P4_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P5_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P6_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P7_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P8_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P9_f(uint8_t* s_data, int s_row, int s_col, int s_width);

#ifdef __cplusplus
extern "C" {
#endif
__declspec(dllexport) int skeletonize(uint8_t* g_src_data0, uint8_t* g_dst_data0, int width, int height, int grid_dim_i, int block_dim_i);
__declspec(dllexport) int distancetransform(uint8_t* g_src_data0, uint8_t* g_dst_data0, uint8_t *voronoi, int width, int height);
__declspec(dllexport) int getconnectedcomponents(uint8_t* g_src_data0, uint8_t* g_dst_data0, int width, int height, int degree_of_connectivity, int *nbcomponents);
#ifdef __cplusplus
}
#endif
__global__ void skeletonize_pass(uint8_t* g_src, uint8_t* g_dst, uint8_t* g_equ, int g_src_width, int g_src_height);
__device__ uint8_t wb_transitions_around(uint8_t* s_data, int s_row, int s_col, int s_width);

#endif
