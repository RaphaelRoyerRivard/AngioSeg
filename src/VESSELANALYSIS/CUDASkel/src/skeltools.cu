#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "skeltools.cuh"
#include "gpu_only_utils.cuh"
#include "lspbmp.hpp"
#include "utils.hpp"

#define PAD_TOP (2)
#define PAD_LEFT (2)
#define PAD_BOTTOM (1)
#define PAD_RIGHT (1)


// Parameters for CUDA kernel executions; more or less optimized for a 1024x1024 image.
#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	16

#define MARKER      -32768


#define TOID(x, y, size)    (__mul24((y), (size)) + (x))


/****** Global Variables *******/
const int NB = 7;						// Nr buffers we use and store in the entire framework
short2 **pbaTextures;					// Work buffers used to compute and store resident images
										//	0: work buffer
										//	1: FT
										//	2: thresholded DT
										//	3: thresholded skeleton
										//	4: topology analysis
										//  5: work buffer for topology
										//  6: skeleton FT
										//

float*			pbaTexSiteParam;		// Stores boundary parameterization
float*			pbaTexSiteParam2;		// Stores boundary parameterization
int				pbaTexSize;				// Texture size (squared) actually used in all computations
int				floodBand = 4,			// Various FT computation parameters; defaults are good for an 1024x1024 image.	
maurerBand = 4,
colorBand = 4;

texture<short2> pbaTexColor;			// 2D textures (bound to various buffers defined above as needed)
texture<short2> pbaTexColor2;			//
texture<short2> pbaTexLinks;
texture<float>  pbaTexParam;			// 1D site parameterization texture (bound to pbaTexSiteParam)
texture<float>  pbaTexParam2;
texture<unsigned char>
pbaTexGray;				// 2D texture of unsigned char values, e.g. the binary skeleton

void and_reduction(uint8_t* g_equ_data, int g_size, dim3 grid_dim, dim3 block_dim) {
    // iterative reductions of g_equ_data
    // important to have a block size which is a power of 2, because the
    // reduction algorithm depends on this for the /2 at each iteration.
    // This will give an odd number at some iterations if the block size is
    // not a power of 2.
    do {
        int and_reduction_shared_mem_size = block_dim.x * sizeof(uint8_t);
        and_reduction<<<grid_dim, block_dim, and_reduction_shared_mem_size>>>(g_equ_data, g_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        g_size = ceil(g_size / ((double) block_dim.x));
        grid_dim.x = (g_size <= block_dim.x) ? 1 : grid_dim.x;
    } while (g_size != 1);
}

__global__ void and_reduction(uint8_t* g_data, int g_size) {
    // shared memory for tile
    extern __shared__ uint8_t s_data[];

    int blockReductionIndex = blockIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Attention : For loop needed here instead of a while loop, because at each
    // iteration there will be work for all threads. A while loop wouldn't allow
    // you to do this.
    int num_iterations_needed = ceil(g_size / ((double) (blockDim.x * gridDim.x)));
    for (int iteration = 0; iteration < num_iterations_needed; iteration++) {
        // Load equality values into shared memory tile. We use 1 as the default
        // value, as it is an AND reduction
        s_data[threadIdx.x] = (i < g_size) ? g_data[i] : 1;
        __syncthreads();

        // do reduction in shared memory
        block_and_reduce(s_data);

        // write result for this block to global memory
        if (threadIdx.x == 0) {
            g_data[blockReductionIndex] = s_data[0];
        }

        blockReductionIndex += gridDim.x;
        i += (gridDim.x * blockDim.x);
    }
}

// Computes the number of black neighbors around a pixel.
__device__ uint8_t black_neighbors_around(uint8_t* s_data, int s_row, int s_col, int s_width) {
    uint8_t count = 0;

    count += (P2_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P3_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P4_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P5_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P6_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P7_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P8_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);
    count += (P9_f(s_data, s_row, s_col, s_width) == BINARY_BLACK);

    return count;
}

__device__ uint8_t block_and_reduce(uint8_t* s_data) {
    for (int s = (blockDim.x / 2); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] &= s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    return s_data[0];
}

__device__ uint8_t border_global_mem_read(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return is_outside_image(g_row, g_col, g_width, g_height) ? BINARY_WHITE : g_data[g_row * g_width + g_col];
}

__device__ uint8_t is_outside_image(int g_row, int g_col, int g_width, int g_height) {
    return (g_row < 0) | (g_row > (g_height - 1)) | (g_col < 0) | (g_col > (g_width - 1));
}

__device__ uint8_t is_white(uint8_t* s_src, int s_src_row, int s_src_col, int s_src_width, uint8_t* s_equ, int s_equ_col) {
    s_equ[s_equ_col] = (s_src[s_src_row * s_src_width + s_src_col] == BINARY_WHITE);
    __syncthreads();

    return block_and_reduce(s_equ);
}

__device__ void load_s_src(uint8_t* g_src, int g_row, int g_col, int g_width, int g_height, uint8_t* s_src, int s_row, int s_col, int s_width) {
    if (threadIdx.x == 0) {
        // left
        s_src[(s_row - 2) * s_width + (s_col - 2)] = border_global_mem_read(g_src, g_row - 2, g_col - 2, g_width, g_height);
        s_src[(s_row - 2) * s_width + (s_col - 1)] = border_global_mem_read(g_src, g_row - 2, g_col - 1, g_width, g_height);
        s_src[(s_row - 2) * s_width + s_col] = border_global_mem_read(g_src, g_row - 2, g_col, g_width, g_height);

        s_src[(s_row - 1) * s_width + (s_col - 2)] = border_global_mem_read(g_src, g_row - 1, g_col - 2, g_width, g_height);
        s_src[(s_row - 1) * s_width + (s_col - 1)] = border_global_mem_read(g_src, g_row - 1, g_col - 1, g_width, g_height);
        s_src[(s_row - 1) * s_width + s_col] = border_global_mem_read(g_src, g_row - 1, g_col, g_width, g_height);

        s_src[s_row * s_width + (s_col - 2)] = border_global_mem_read(g_src, g_row, g_col - 2, g_width, g_height);
        s_src[s_row * s_width + (s_col - 1)] = border_global_mem_read(g_src, g_row, g_col - 1, g_width, g_height);
        s_src[s_row * s_width + s_col] = border_global_mem_read(g_src, g_row, g_col, g_width, g_height);

        s_src[(s_row + 1) * s_width + (s_col - 2)] = border_global_mem_read(g_src, g_row + 1, g_col - 2, g_width, g_height);
        s_src[(s_row + 1) * s_width + (s_col - 1)] = border_global_mem_read(g_src, g_row + 1, g_col - 1, g_width, g_height);
        s_src[(s_row + 1) * s_width + s_col] = border_global_mem_read(g_src, g_row + 1, g_col, g_width, g_height);
    } else if (threadIdx.x == (blockDim.x - 1)) {
        // right
        s_src[(s_row - 2) * s_width + s_col] = border_global_mem_read(g_src, g_row - 2, g_col, g_width, g_height);
        s_src[(s_row - 2) * s_width + (s_col + 1)] = border_global_mem_read(g_src, g_row - 2, g_col + 1, g_width, g_height);

        s_src[(s_row - 1) * s_width + s_col] = border_global_mem_read(g_src, g_row - 1, g_col, g_width, g_height);
        s_src[(s_row - 1) * s_width + (s_col + 1)] = border_global_mem_read(g_src, g_row - 1, g_col + 1, g_width, g_height);

        s_src[s_row * s_width + s_col] = border_global_mem_read(g_src, g_row, g_col, g_width, g_height);
        s_src[s_row * s_width + (s_col + 1)] = border_global_mem_read(g_src, g_row, g_col + 1, g_width, g_height);

        s_src[(s_row + 1) * s_width + s_col] = border_global_mem_read(g_src, g_row + 1, g_col, g_width, g_height);
        s_src[(s_row + 1) * s_width + (s_col + 1)] = border_global_mem_read(g_src, g_row + 1, g_col + 1, g_width, g_height);
    } else {
        // center
        s_src[(s_row - 2) * s_width + s_col] = border_global_mem_read(g_src, g_row - 2, g_col, g_width, g_height);

        s_src[(s_row - 1) * s_width + s_col] = border_global_mem_read(g_src, g_row - 1, g_col, g_width, g_height);

        s_src[s_row * s_width + s_col] = border_global_mem_read(g_src, g_row, g_col, g_width, g_height);

        s_src[(s_row + 1) * s_width + s_col] = border_global_mem_read(g_src, g_row + 1, g_col, g_width, g_height);
    }

    __syncthreads();
}

__device__ uint8_t P2_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row - 1) * s_width + s_col];
}

__device__ uint8_t P3_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row - 1) * s_width + (s_col - 1)];
}

__device__ uint8_t P4_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[s_row * s_width + (s_col - 1)];
}

__device__ uint8_t P5_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row + 1) * s_width + (s_col - 1)];
}

__device__ uint8_t P6_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row + 1) * s_width + s_col];
}

__device__ uint8_t P7_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row + 1) * s_width + (s_col + 1)];
}

__device__ uint8_t P8_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[s_row * s_width + (s_col + 1)];
}

__device__ uint8_t P9_f(uint8_t* s_data, int s_row, int s_col, int s_width) {
    return s_data[(s_row - 1) * s_width + (s_col + 1)];
}




// Transpose a square matrix
__global__ void kernelTranspose(short2 *data, int size)
{
	__shared__ short2 block1[TILE_DIM][TILE_DIM + 1];
	__shared__ short2 block2[TILE_DIM][TILE_DIM + 1];

	int blockIdx_y = blockIdx.x;
	int blockIdx_x = blockIdx.x + blockIdx.y;

	if (blockIdx_x >= gridDim.x) return;

	int blkX, blkY, x, y, id1, id2;
	short2 pixel;

	blkX = __mul24(blockIdx_x, TILE_DIM);
	blkY = __mul24(blockIdx_y, TILE_DIM);

	x = blkX + threadIdx.x;
	y = blkY + threadIdx.y;
	id1 = __mul24(y, size) + x;

	x = blkY + threadIdx.x;
	y = blkX + threadIdx.y;
	id2 = __mul24(y, size) + x;

	// read the matrix tile into shared memory
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		int idx = __mul24(i, size);
		block1[threadIdx.y + i][threadIdx.x] = tex1Dfetch(pbaTexColor, id1 + idx);
		block2[threadIdx.y + i][threadIdx.x] = tex1Dfetch(pbaTexColor, id2 + idx);
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		int idx = __mul24(i, size);
		pixel = block1[threadIdx.x][threadIdx.y + i];
		data[id2 + idx] = make_short2(pixel.y, pixel.x);
		pixel = block2[threadIdx.x][threadIdx.y + i];
		data[id1 + idx] = make_short2(pixel.y, pixel.x);
	}
}

__global__ void kernelFloodDown(short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;
	int id = TOID(tx, ty, size);

	short2 pixel1, pixel2;

	pixel1 = make_short2(MARKER, MARKER);

	for (int i = 0; i < bandSize; ++i, id += size)
	{
		pixel2 = tex1Dfetch(pbaTexColor, id);

		if (pixel2.x != MARKER)  pixel1 = pixel2;

		output[id] = pixel1;
	}
}


__global__ void kernelFloodUp(short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = (blockIdx.y + 1) * bandSize - 1;
	int id = TOID(tx, ty, size);

	short2 pixel1, pixel2;
	int dist1, dist2;

	pixel1 = make_short2(MARKER, MARKER);

	for (int i = 0; i < bandSize; i++, id -= size)
	{
		dist1 = abs(pixel1.y - ty + i);

		pixel2 = tex1Dfetch(pbaTexColor, id);
		dist2 = abs(pixel2.y - ty + i);

		if (dist2 < dist1) pixel1 = pixel2;

		output[id] = pixel1;
	}
}

__global__ void kernelPropagateInterband(short2 *output, int size, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int inc = __mul24(bandSize, size);
	int ny, nid, nDist;
	short2 pixel;

	// Top row, look backward
	int ty = __mul24(blockIdx.y, bandSize);
	int topId = TOID(tx, ty, size);
	int bottomId = TOID(tx, ty + bandSize - 1, size);

	pixel = tex1Dfetch(pbaTexColor, topId);
	int myDist = abs(pixel.y - ty);

	for (nid = bottomId - inc; nid >= 0; nid -= inc)
	{
		pixel = tex1Dfetch(pbaTexColor, nid);
		if (pixel.x != MARKER)
		{
			nDist = abs(pixel.y - ty);
			if (nDist < myDist) output[topId] = pixel;
			break;
		}
	}

	// Last row, look downward
	ty = ty + bandSize - 1;
	pixel = tex1Dfetch(pbaTexColor, bottomId);
	myDist = abs(pixel.y - ty);

	for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc)
	{
		pixel = tex1Dfetch(pbaTexColor, nid);

		if (pixel.x != MARKER)
		{
			nDist = abs(pixel.y - ty);
			if (nDist < myDist) output[bottomId] = pixel;
			break;
		}
	}
}

__global__ void kernelUpdateVertical(short2 *output, int size, int band, int bandSize)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * bandSize;

	short2 top = tex1Dfetch(pbaTexLinks, TOID(tx, ty, size));
	short2 bottom = tex1Dfetch(pbaTexLinks, TOID(tx, ty + bandSize - 1, size));
	short2 pixel;

	int dist, myDist;

	int id = TOID(tx, ty, size);

	for (int i = 0; i < bandSize; i++, id += size)
	{
		pixel = tex1Dfetch(pbaTexColor, id);
		myDist = abs(pixel.y - (ty + i));

		dist = abs(top.y - (ty + i));
		if (dist < myDist) { myDist = dist; pixel = top; }

		dist = abs(bottom.y - (ty + i));
		if (dist < myDist) pixel = bottom;

		output[id] = pixel;
	}
}

// Input: y1 < y2
__device__ float interpoint(int x1, int y1, int x2, int y2, int x0)
{
	float xM = float(x1 + x2) / 2.0f;
	float yM = float(y1 + y2) / 2.0f;
	float nx = x2 - x1;
	float ny = y2 - y1;

	return yM + nx * (xM - x0) / ny;
}

__global__ void kernelProximatePoints(short2 *stack, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y, bandSize);
	int id = TOID(tx, ty, size);
	int lasty = -1;
	short2 last1, last2, current;
	float i1, i2;

	last1.y = -1; last2.y = -1;

	for (int i = 0; i < bandSize; i++, id += size) {
		current = tex1Dfetch(pbaTexColor, id);

		if (current.x != MARKER) {
			while (last2.y >= 0) {
				i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx);
				i2 = interpoint(last2.x, lasty, current.x, current.y, tx);

				if (i1 < i2) break;

				lasty = last2.y; last2 = last1;

				if (last1.y >= 0) last1 = stack[TOID(tx, last1.y, size)];
			}

			last1 = last2; last2 = make_short2(current.x, lasty); lasty = current.y;

			stack[id] = last2;
		}
	}

	// Store the pointer to the tail at the last pixel of this band
	if (lasty != ty + bandSize - 1)
		stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, lasty);
}

__global__ void kernelCreateForwardPointers(short2 *output, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = __mul24(blockIdx.y + 1, bandSize) - 1;
	int id = TOID(tx, ty, size);
	int lasty = -1, nexty;
	short2 current;

	// Get the tail pointer
	current = tex1Dfetch(pbaTexLinks, id);

	if (current.x == MARKER)
		nexty = current.y;
	else
		nexty = ty;

	for (int i = 0; i < bandSize; i++, id -= size)
		if (ty - i == nexty) {
			current = make_short2(lasty, tex1Dfetch(pbaTexLinks, id).y);
			output[id] = current;

			lasty = nexty;
			nexty = current.y;
		}

	// Store the pointer to the head at the first pixel of this band
	if (lasty != ty - bandSize + 1)
		output[id + size] = make_short2(lasty, MARKER);
}

__global__ void kernelMergeBands(short2 *output, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int band1 = blockIdx.y * 2;
	int band2 = band1 + 1;
	int firsty, lasty;
	short2 last1, last2, current;
	// last1 and last2: x component store the x coordinate of the site, 
	// y component store the backward pointer
	// current: y component store the x coordinate of the site, 
	// x component store the forward pointer

	// Get the two last items of the first list
	lasty = __mul24(band2, bandSize) - 1;
	last2 = make_short2(tex1Dfetch(pbaTexColor, TOID(tx, lasty, size)).x,
		tex1Dfetch(pbaTexLinks, TOID(tx, lasty, size)).y);

	if (last2.x == MARKER) {
		lasty = last2.y;

		if (lasty >= 0)
			last2 = make_short2(tex1Dfetch(pbaTexColor, TOID(tx, lasty, size)).x,
				tex1Dfetch(pbaTexLinks, TOID(tx, lasty, size)).y);
		else
			last2 = make_short2(MARKER, MARKER);
	}

	if (last2.y >= 0) {
		// Second item at the top of the stack
		last1 = make_short2(tex1Dfetch(pbaTexColor, TOID(tx, last2.y, size)).x,
			tex1Dfetch(pbaTexLinks, TOID(tx, last2.y, size)).y);
	}

	// Get the first item of the second band
	firsty = __mul24(band2, bandSize);
	current = make_short2(tex1Dfetch(pbaTexLinks, TOID(tx, firsty, size)).x,
		tex1Dfetch(pbaTexColor, TOID(tx, firsty, size)).x);

	if (current.y == MARKER) {
		firsty = current.x;

		if (firsty >= 0)
			current = make_short2(tex1Dfetch(pbaTexLinks, TOID(tx, firsty, size)).x,
				tex1Dfetch(pbaTexColor, TOID(tx, firsty, size)).x);
		else
			current = make_short2(MARKER, MARKER);
	}

	float i1, i2;

	// Count the number of item in the second band that survive so far. 
	// Once it reaches 2, we can stop. 
	int top = 0;

	while (top < 2 && current.y >= 0) {
		// While there's still something on the left
		while (last2.y >= 0) {
			i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx);
			i2 = interpoint(last2.x, lasty, current.y, firsty, tx);

			if (i1 < i2)
				break;

			lasty = last2.y; last2 = last1;
			--top;

			if (last1.y >= 0)
				last1 = make_short2(tex1Dfetch(pbaTexColor, TOID(tx, last1.y, size)).x,
					output[TOID(tx, last1.y, size)].y);
		}

		// Update the current pointer 
		output[TOID(tx, firsty, size)] = make_short2(current.x, lasty);

		if (lasty >= 0)
			output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y);

		last1 = last2; last2 = make_short2(current.y, lasty); lasty = firsty;
		firsty = current.x;

		top = max(1, top + 1);

		// Advance the current pointer to the next one
		if (firsty >= 0)
			current = make_short2(tex1Dfetch(pbaTexLinks, TOID(tx, firsty, size)).x,
				tex1Dfetch(pbaTexColor, TOID(tx, firsty, size)).x);
		else
			current = make_short2(MARKER, MARKER);
	}

	// Update the head and tail pointer. 
	firsty = __mul24(band1, bandSize);
	lasty = __mul24(band2, bandSize);
	current = tex1Dfetch(pbaTexLinks, TOID(tx, firsty, size));

	if (current.y == MARKER && current.x < 0) {	// No head?
		last1 = tex1Dfetch(pbaTexLinks, TOID(tx, lasty, size));

		if (last1.y == MARKER)
			current.x = last1.x;
		else
			current.x = lasty;

		output[TOID(tx, firsty, size)] = current;
	}

	firsty = __mul24(band1, bandSize) + bandSize - 1;
	lasty = __mul24(band2, bandSize) + bandSize - 1;
	current = tex1Dfetch(pbaTexLinks, TOID(tx, lasty, size));

	if (current.x == MARKER && current.y < 0) {	// No tail?
		last1 = tex1Dfetch(pbaTexLinks, TOID(tx, firsty, size));

		if (last1.x == MARKER)
			current.y = last1.y;
		else
			current.y = firsty;

		output[TOID(tx, lasty, size)] = current;
	}
}

__global__ void kernelDoubleToSingleList(short2 *output, int size)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int ty = blockIdx.y;
	int id = TOID(tx, ty, size);

	output[id] = make_short2(tex1Dfetch(pbaTexColor, id).x, tex1Dfetch(pbaTexLinks, id).y);
}

__global__ void kernelColor(short2 *output, int size)
{
	__shared__ short2 s_last1[BLOCKSIZE], s_last2[BLOCKSIZE];
	__shared__ int s_lasty[BLOCKSIZE];

	int col = threadIdx.x;
	int tid = threadIdx.y;
	int tx = __mul24(blockIdx.x, blockDim.x) + col;
	int dx, dy, lasty;
	unsigned int best, dist;
	short2 last1, last2;

	if (tid == blockDim.y - 1)
	{
		lasty = size - 1;

		last2 = tex1Dfetch(pbaTexColor, __mul24(lasty, size) + tx);

		if (last2.x == MARKER) {
			lasty = last2.y;
			last2 = tex1Dfetch(pbaTexColor, __mul24(lasty, size) + tx);
		}

		if (last2.y >= 0)
			last1 = tex1Dfetch(pbaTexColor, __mul24(last2.y, size) + tx);

		s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty;
	}

	__syncthreads();

	for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y)
	{
		last1 = s_last1[col]; last2 = s_last2[col]; lasty = s_lasty[col];

		dx = last2.x - tx; dy = lasty - ty;
		best = dist = __mul24(dx, dx) + __mul24(dy, dy);

		while (last2.y >= 0) {
			dx = last1.x - tx; dy = last2.y - ty;
			dist = __mul24(dx, dx) + __mul24(dy, dy);

			if (dist > best)
				break;

			best = dist; lasty = last2.y; last2 = last1;

			if (last2.y >= 0)
				last1 = tex1Dfetch(pbaTexColor, __mul24(last2.y, size) + tx);
		}

		__syncthreads();

		output[TOID(tx, ty, size)] = make_short2(last2.x, lasty);

		if (tid == blockDim.y - 1) {
			s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty;
		}

		__syncthreads();
	}
}


// In-place transpose a squared texture. 
// Block orders are modified to optimize memory access. 
// Point coordinates are also swapped. 
void pba2DTranspose(short2 *texture)
{
	dim3 block(TILE_DIM, BLOCK_ROWS);
	dim3 grid(pbaTexSize / TILE_DIM, pbaTexSize / TILE_DIM);

	cudaBindTexture(0, pbaTexColor, texture);
	kernelTranspose << < grid, block >> >(texture, pbaTexSize);
	cudaUnbindTexture(pbaTexColor);
}

// Phase 1 of PBA. m1 must divides texture size
void pba2DPhase1(int m1, short xm, short ym, short xM, short yM)
{
	dim3 block = dim3(BLOCKSIZE);
	dim3 grid = dim3(pbaTexSize / block.x, m1);

	// Flood vertically in their own bands
	cudaBindTexture(0, pbaTexColor, pbaTextures[0]);
	kernelFloodDown << < grid, block >> >(pbaTextures[1], pbaTexSize, pbaTexSize / m1);
	cudaUnbindTexture(pbaTexColor);

	cudaBindTexture(0, pbaTexColor, pbaTextures[1]);
	kernelFloodUp << < grid, block >> >(pbaTextures[1], pbaTexSize, pbaTexSize / m1);

	// Passing information between bands
	grid = dim3(pbaTexSize / block.x, m1);
	kernelPropagateInterband << < grid, block >> >(pbaTextures[0], pbaTexSize, pbaTexSize / m1);

	cudaBindTexture(0, pbaTexLinks, pbaTextures[0]);
	kernelUpdateVertical << < grid, block >> >(pbaTextures[1], pbaTexSize, m1, pbaTexSize / m1);
	cudaUnbindTexture(pbaTexLinks);
	cudaUnbindTexture(pbaTexColor);
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2)
{
	// Compute proximate points locally in each band
	dim3 block = dim3(BLOCKSIZE);
	dim3 grid = dim3(pbaTexSize / block.x, m2);
	cudaBindTexture(0, pbaTexColor, pbaTextures[1]);
	kernelProximatePoints << < grid, block >> >(pbaTextures[0], pbaTexSize, pbaTexSize / m2);

	cudaBindTexture(0, pbaTexLinks, pbaTextures[0]);
	kernelCreateForwardPointers << < grid, block >> >(pbaTextures[0], pbaTexSize, pbaTexSize / m2);

	// Repeatly merging two bands into one
	for (int noBand = m2; noBand > 1; noBand /= 2) {
		grid = dim3(pbaTexSize / block.x, noBand / 2);
		kernelMergeBands << < grid, block >> >(pbaTextures[0], pbaTexSize, pbaTexSize / noBand);
	}

	// Replace the forward link with the X coordinate of the seed to remove
	// the need of looking at the other texture. We need it for coloring.
	grid = dim3(pbaTexSize / block.x, pbaTexSize);
	kernelDoubleToSingleList << < grid, block >> >(pbaTextures[0], pbaTexSize);
	cudaUnbindTexture(pbaTexLinks);
	cudaUnbindTexture(pbaTexColor);
}

// Phase 3 of PBA. m3 must divides texture size
void pba2DPhase3(int m3)
{
	dim3 block = dim3(BLOCKSIZE / m3, m3);
	dim3 grid = dim3(pbaTexSize / block.x);
	cudaBindTexture(0, pbaTexColor, pbaTextures[0]);
	kernelColor << < grid, block >> >(pbaTextures[1], pbaTexSize);
	cudaUnbindTexture(pbaTexColor);
}



void skel2DFTCompute(short xm, short ym, short xM, short yM, int floodBand, int maurerBand, int colorBand)
{
	pba2DPhase1(floodBand, xm, ym, xM, yM);										//Vertical sweep

	pba2DTranspose(pbaTextures[1]);											//

	pba2DPhase2(maurerBand);												//Horizontal coloring

	pba2DPhase3(colorBand);													//Row coloring

	pba2DTranspose(pbaTextures[1]);
}



__global__ void kernelSiteParamInit(short2* inputVoro, int size)							//Initialize the Voronoi textures from the sites' encoding texture (parameterization)
{																							//REMARK: we interpret 'inputVoro' as a 2D texture, as it's much easier/faster like this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx<size && ty<size)																	//Careful not to go outside the image..
	{
		int i = TOID(tx, ty, size);
		float param = tex1Dfetch(pbaTexParam, i);												//The sites-param has non-zero (parameter) values precisely on non-boundary points

		short2& v = inputVoro[i];
		v.x = v.y = MARKER;																	//Non-boundary points are marked as 0 in the parameterization. Here we will compute the FT.
		if (param)																			//These are points which define the 'sites' to compute the FT/skeleton (thus, have FT==identity)
		{																						//We could use an if-then-else here, but it's faster with an if-then
			v.x = tx; v.y = ty;
		}
	}
}


void skelft2DInitializeInput(float* sites, int size)										// Copy input sites from CPU to GPU; Also set up site param initialization in pbaTextures[0]
{
	pbaTexSize = size;																		// Size of the actual texture being used in this run; can be smaller than the max-tex-size
																							// which was used in skelft2DInitialization()
	pbaTextures = (short2 **)malloc(2 * sizeof(short2*));
	cudaMalloc((void **)&pbaTextures[0], pbaTexSize*pbaTexSize*sizeof(short2));
	cudaMalloc((void **)&pbaTextures[1], pbaTexSize*pbaTexSize * sizeof(short2));
	cudaMalloc((void **)&pbaTexSiteParam, pbaTexSize * pbaTexSize * sizeof(float));

	cudaMemcpy(pbaTexSiteParam, sites, pbaTexSize * pbaTexSize * sizeof(float), cudaMemcpyHostToDevice);
	// Pass sites parameterization to CUDA.  Must be done before calling the initialization
	// kernel, since we use the sites-param as a texture in that kernel
	cudaBindTexture(0, pbaTexParam, pbaTexSiteParam);										// Bind the sites-param as a 1D texture so we can quickly index it next
	dim3 block = dim3(BLOCKX, BLOCKY);
	dim3 grid = dim3(pbaTexSize / block.x, pbaTexSize / block.y);

	kernelSiteParamInit << <grid, block >> >(pbaTextures[0], pbaTexSize);							// Do the site param initialization. This sets up pbaTextures[0]
	cudaUnbindTexture(pbaTexParam);
}


// Compute 2D FT / Voronoi diagram of a set of sites
// siteParam:   Site parameterization. 0 = non-site points; >0 = site parameter value.
// output:		FT. The (x,y) at (i,j) are the coords of the closest site to (i,j)
// size:        Texture size (pow 2)
void skelft2DFT(short* output, float* siteParam, short xm, short ym, short xM, short yM, int size)
{
	skelft2DInitializeInput(siteParam, size);								    // Initialization of already-allocated data structures

	skel2DFTCompute(xm, ym, xM, yM, floodBand, maurerBand, colorBand);			// Compute FT

																				// Copy FT to CPU, if required
	memset(siteParam, 0, size*size * sizeof(float));
	if (output) cudaMemcpy(output, pbaTextures[1], size*size * sizeof(short2), cudaMemcpyDeviceToHost);

	for (int i = 0; i<2; ++i) cudaFree(pbaTextures[i]);
	cudaFree(pbaTexSiteParam);
	free(pbaTextures);
	//if (output) cudaMemcpy(siteParam, pbaTexSiteParam2,  size*size * sizeof(float), cudaMemcpyDeviceToHost);
}


//int 

int distancetransform(uint8_t* g_src_data0, uint8_t* g_dst_data0, uint8_t *voronoi, int width, int height)
{
	float *image = (float*)malloc(width*height * sizeof(float));
	float *dest = (float*)g_dst_data0;
	short *dft = (short*)voronoi;

	short xm=height, ym = width,  xM = 0, yM = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			image[i*width + j] = g_src_data0[i*width + j];
			if (g_src_data0[i*width + j])
			{
				xm = min(xm, i); ym = min(ym, j);
				xM = max(xM, i); yM = max(yM, j);
			}
		}
	}

	//short *dft = (short*)malloc(width*height *2* sizeof(short));

	skelft2DFT(dft, image, xm, ym, xM, yM, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			short courx = abs(dft[i*width * 2 + j * 2 + 0] - j);
			short coury = abs(dft[i*width * 2 + j * 2 + 1] - i);
			float distance = sqrt(courx*courx + coury*coury)*2.0;
			dest[i*width + j] = distance;
		}
	}

	free(image);
	//free(dft);

	return 0;
}
// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
int skeletonize(uint8_t* g_src_data0, uint8_t* g_dst_data0, int width, int height, int grid_dim_i, int block_dim_i) {
    // allocate memory on device
    uint8_t* g_src_data = NULL;
    uint8_t* g_dst_data = NULL;
	//FILE *f = fopen("G:/RECHERCHE/tools/skeletonization-master/test/gpu/test.raw", "rb");

	//fread(g_src_data0, 1, 512 * 512, f);


	dim3 grid_dim(grid_dim_i);
	dim3 block_dim(block_dim_i);

    int g_data_size = width * height * sizeof(uint8_t);
    gpuErrchk(cudaMalloc((void**) &g_src_data, g_data_size));
    gpuErrchk(cudaMalloc((void**) &g_dst_data, g_data_size));

    uint8_t* g_equ_data = NULL;
    int g_equ_size = ceil((width * height) / ((double) block_dim.x));
    gpuErrchk(cudaMalloc((void**) &g_equ_data, g_equ_size));

    // send data to device
    gpuErrchk(cudaMemcpy(g_src_data, g_src_data0, g_data_size, cudaMemcpyHostToDevice));

    uint8_t are_identical_bitmaps = 0;
    int iterations = 0;
    do {
        // copy g_src_data over g_dst_data (GPU <-> GPU transfer, so it has much
        // higher throughput than HOST <-> DEVICE transfers)
        gpuErrchk(cudaMemcpy(g_dst_data, g_src_data, g_data_size, cudaMemcpyDeviceToDevice));

        // set g_equ_data to 1 (GPU <-> GPU transfer, so it has very high
        // throughput)
        gpuErrchk(cudaMemset(g_equ_data, 1, g_equ_size));

        int skeletonize_pass_s_src_size = (block_dim.x + PAD_LEFT + PAD_RIGHT) * (1 + PAD_TOP + PAD_BOTTOM) * sizeof(uint8_t);
        int skeletonize_pass_s_equ_size = block_dim.x * sizeof(uint8_t);
        int skeletonize_pass_shared_mem_size = skeletonize_pass_s_src_size + skeletonize_pass_s_equ_size;
        skeletonize_pass<<<grid_dim, block_dim, skeletonize_pass_shared_mem_size>>>(g_src_data, g_dst_data, g_equ_data, width, height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        and_reduction(g_equ_data, g_equ_size, grid_dim, block_dim);

        // bring reduced bitmap equality information back from device
        gpuErrchk(cudaMemcpy(&are_identical_bitmaps, g_equ_data, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        swap_bitmaps((void**) &g_src_data, (void**) &g_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps);

    // bring dst_bitmap back from device
    gpuErrchk(cudaMemcpy(g_dst_data0, g_dst_data, g_data_size, cudaMemcpyDeviceToHost));

    // free memory on device
    gpuErrchk(cudaFree(g_src_data));
    gpuErrchk(cudaFree(g_dst_data));
    gpuErrchk(cudaFree(g_equ_data));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* g_src, uint8_t* g_dst, uint8_t* g_equ, int g_src_width, int g_src_height) {
    // shared memory for tile
    extern __shared__ uint8_t s_data[];
    uint8_t* s_src = &s_data[0];
    uint8_t* s_equ = &s_data[(blockDim.x + PAD_LEFT + PAD_RIGHT) * (1 + PAD_TOP + PAD_BOTTOM)];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int currentBlockIndex = blockIdx.x;

    int g_size = g_src_width * g_src_height;

    while (tid < g_size) {
        int g_src_row = (tid / g_src_width);
        int g_src_col = (tid % g_src_width);

        int s_src_row = PAD_TOP;
        int s_src_col = threadIdx.x + PAD_LEFT;
        int s_src_width = blockDim.x + PAD_LEFT + PAD_RIGHT;

        int s_equ_col = threadIdx.x;

        // load g_src into shared memory
        load_s_src(g_src, g_src_row, g_src_col, g_src_width, g_src_height, s_src, s_src_row, s_src_col, s_src_width);
        uint8_t is_src_white = is_white(s_src, s_src_row, s_src_col, s_src_width, s_equ, s_equ_col);

        if (!is_src_white) {
            uint8_t NZ = black_neighbors_around(s_src, s_src_row, s_src_col, s_src_width);
            uint8_t TR_P1 = wb_transitions_around(s_src, s_src_row, s_src_col, s_src_width);
            uint8_t TR_P2 = wb_transitions_around(s_src, s_src_row - 1, s_src_col, s_src_width);
            uint8_t TR_P4 = wb_transitions_around(s_src, s_src_row, s_src_col - 1, s_src_width);
            uint8_t P2 = P2_f(s_src, s_src_row, s_src_col, s_src_width);
            uint8_t P4 = P4_f(s_src, s_src_row, s_src_col, s_src_width);
            uint8_t P6 = P6_f(s_src, s_src_row, s_src_col, s_src_width);
            uint8_t P8 = P8_f(s_src, s_src_row, s_src_col, s_src_width);

            uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
            uint8_t thinning_cond_2 = (TR_P1 == 1);
            uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
            uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
            uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

            uint8_t g_dst_next = (thinning_cond_ok * BINARY_WHITE) + ((1 - thinning_cond_ok) * s_src[s_src_row * s_src_width + s_src_col]);
            __syncthreads();
            g_dst[g_src_row * g_src_width + g_src_col] = g_dst_next;

            // compute and write reduced value of s_equ to g_equ:
            //
            // do the first iteration of g_equ's reduction, since we already have
            // everything available in shared memory. This avoids the and_reduction
            // kernel to have to load (g_src_width * g_src_height) data, but only
            // ceil((g_src_width * g_src_height) / ((double) block_dim.x)), which is
            // much less.
            s_equ[s_equ_col] = (s_src[s_src_row * s_src_width + s_src_col] == g_dst_next);
            __syncthreads();
            uint8_t g_equ_next = block_and_reduce(s_equ);
            if (s_equ_col == 0) {
                g_equ[currentBlockIndex] = g_equ_next;
            }
        }

        currentBlockIndex += gridDim.x;
        tid += (gridDim.x * blockDim.x);
    }
}

// Computes the number of white to black transitions around a pixel.
__device__ uint8_t wb_transitions_around(uint8_t* s_data, int s_row, int s_col, int s_width) {
    uint8_t count = 0;

    count += ((P2_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P3_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P3_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P4_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P4_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P5_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P5_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P6_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P6_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P7_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P7_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P8_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P8_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P9_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));
    count += ((P9_f(s_data, s_row, s_col, s_width) == BINARY_WHITE) & (P2_f(s_data, s_row, s_col, s_width) == BINARY_BLACK));

    return count;
}

int main2(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;
    Padding padding_for_thread_count;
    dim3 grid_dim;
    dim3 block_dim;

    gpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap, &padding_for_thread_count, &grid_dim, &block_dim);

    int iterations = skeletonize(src_bitmap->data, dst_bitmap->data, src_bitmap->width, src_bitmap->height,  atoi(argv[3]), atoi(argv[4]));
    printf(" %u iterations\n", iterations);
    printf("\n");

    gpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap, padding_for_thread_count);

    return EXIT_SUCCESS;
}
