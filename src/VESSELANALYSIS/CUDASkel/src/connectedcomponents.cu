#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <list>
#include <algorithm>
#include <utility>
#include <cmath>
#include <functional>
#include <cstring>
#include <cmath>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "skeltools.cuh"

#define UNROLL_INNER

//#include <cutil_inline.h>

#define NOMINMAX



//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)


////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 1
#define      KERNEL_W (2 * KERNEL_RADIUS + 1)*2
#define KERNELW_2 (2 * KERNEL_RADIUS + 1)
__device__ __constant__ float d_Kernel[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations 
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48



const int KERNEL_SIZE = KERNEL_W * sizeof(float);

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float *data) {
	return
		data[KERNEL_RADIUS - i] * d_Kernel[i]
		+ convolutionRow<i - 1>(data);
}

template<> __device__ float convolutionRow<-1>(float *data) {
	return 0;
}

template<int i> __device__ float convolutionColumn(float *data) {
	return
		data[(KERNEL_RADIUS - i) * COLUMN_TILE_W] * d_Kernel[i + KERNELW_2]
		+ convolutionColumn<i - 1>(data);
}

template<> __device__ float convolutionColumn<-1>(float *data) {
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
	float *d_Result,
	float *d_Data,
	int dataW,
	int dataH
) {
	//Data cache
	__shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

	//Current tile and apron limits, relative to row start
	const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
	const int           tileEnd = tileStart + ROW_TILE_W - 1;
	const int        apronStart = tileStart - KERNEL_RADIUS;
	const int          apronEnd = tileEnd + KERNEL_RADIUS;

	//Clamp tile and apron limits by image borders
	const int    tileEndClamped = min(tileEnd, dataW - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int   apronEndClamped = min(apronEnd, dataW - 1);

	//Row start index in d_Data[]
	const int          rowStart = IMUL(blockIdx.y, dataW);

	//Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
	//of half-warp size, rowStart + apronStartAligned is also a 
	//multiple of half-warp size, thus having proper alignment 
	//for coalesced d_Data[] read.
	const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

	const int loadPos = apronStartAligned + threadIdx.x;
	//Set the entire data cache contents
	//Load global memory values, if indices are within the image borders,
	//or initialize with zeroes otherwise
	if (loadPos >= apronStart) {
		const int smemPos = loadPos - apronStart;

		data[smemPos] =
			((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
			d_Data[rowStart + loadPos] : 0;
	}


	//Ensure the completness of the loading stage
	//because results, emitted by each thread depend on the data,
	//loaded by another threads
	__syncthreads();
	const int writePos = tileStart + threadIdx.x;

	//Assuming dataW and ROW_TILE_W are multiples of half-warp size,
	//rowStart + tileStart is also a multiple of half-warp size,
	//thus having proper alignment for coalesced d_Result[] write.
	if (writePos <= tileEndClamped) {
		const int smemPos = writePos - apronStart;
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionRow< 2 * KERNEL_RADIUS >(data + smemPos);
#else
		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum += data[smemPos + k] * d_Kernel[k + KERNEL_RADIUS];
#endif

		d_Result[rowStart + writePos] = sum;
	}
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
	float *d_Result,
	float *d_Data,
	int dataW,
	int dataH,
	int smemStride,
	int gmemStride
) {
	//Data cache
	__shared__ float data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

	//Current tile and apron limits, in rows
	const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
	const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
	const int        apronStart = tileStart - KERNEL_RADIUS;
	const int          apronEnd = tileEnd + KERNEL_RADIUS;

	//Clamp tile and apron limits by image borders
	const int    tileEndClamped = min(tileEnd, dataH - 1);
	const int apronStartClamped = max(apronStart, 0);
	const int   apronEndClamped = min(apronEnd, dataH - 1);

	//Current column index
	const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

	//Shared and global memory indices for current column
	int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
	int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

	//Cycle through the entire data cache
	//Load global memory values, if indices are within the image borders,
	//or initialize with zero otherwise
	for (int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y) {
		data[smemPos] =
			((y >= apronStartClamped) && (y <= apronEndClamped)) ?
			d_Data[gmemPos] : 0;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}

	//Ensure the completness of the loading stage
	//because results, emitted by each thread depend on the data, 
	//loaded by another threads
	__syncthreads();

	//Shared and global memory indices for current column
	smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
	gmemPos = IMUL(tileStart + threadIdx.y, dataW) + columnStart;

	//Cycle through the tile body, clamped by image borders
	//Calculate and output the results
	for (int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y) {
		float sum = 0;

#ifdef UNROLL_INNER
		sum = convolutionColumn<2 * KERNEL_RADIUS>(data + smemPos);
#else
		for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
			sum +=
			data[smemPos + IMUL(k, COLUMN_TILE_W)] *
			d_Kernel[k + KERNEL_RADIUS + KERNEL_W];
#endif

		d_Result[gmemPos] = sum;
		smemPos += smemStride;
		gmemPos += gmemStride;
	}
}


#ifdef _MSC_VER
#include <ctime>
	inline double get_time()
{
	return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
#else
#include <sys/time.h>
	inline double get_time()
{
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}
#endif

using namespace std;

//const int BLOCK = 128;
const int BLOCK = 256;


int iDivUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void init_CCL(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[id] = id;
}

__device__ int diff(int d1, int d2)
{
	return abs(((d1 >> 16) & 0xff) - ((d2 >> 16) & 0xff)) + abs(((d1 >> 8) & 0xff) - ((d2 >> 8) & 0xff)) + abs((d1 & 0xff) - (d2 & 0xff));
}

__global__ void scanning(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - W >= 0 && diff(Did, D[id - W]) <= th) label = min(label, L[id - W]);
	if (id + W < N  && diff(Did, D[id + W]) <= th) label = min(label, L[id + W]);
	int r = id % W;
	if (r           && diff(Did, D[id - 1]) <= th) label = min(label, L[id - 1]);
	if (r + 1 != W  && diff(Did, D[id + 1]) <= th) label = min(label, L[id + 1]);

	if (label < L[id]) {
		//atomicMin(&R[L[id]], label);
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void scanning8(int D[], int L[], int R[], bool* m, int N, int W, int th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int Did = D[id];
	int label = N;
	if (id - W >= 0 && diff(Did, D[id - W]) <= th) label = min(label, L[id - W]);
	if (id + W < N  && diff(Did, D[id + W]) <= th) label = min(label, L[id + W]);
	int r = id % W;
	if (r) {
		if (diff(Did, D[id - 1]) <= th) label = min(label, L[id - 1]);
		if (id - W - 1 >= 0 && diff(Did, D[id - W - 1]) <= th) label = min(label, L[id - W - 1]);
		if (id + W - 1 < N  && diff(Did, D[id + W - 1]) <= th) label = min(label, L[id + W - 1]);
	}
	if (r + 1 != W) {
		if (diff(Did, D[id + 1]) <= th) label = min(label, L[id + 1]);
		if (id - W + 1 >= 0 && diff(Did, D[id - W + 1]) <= th) label = min(label, L[id - W + 1]);
		if (id + W + 1 < N  && diff(Did, D[id + W + 1]) <= th) label = min(label, L[id + W + 1]);
	}

	if (label < L[id]) {
		//atomicMin(&R[L[id]], label);
		R[L[id]] = label;
		*m = true;
	}
}

__global__ void analysis(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int label = L[id];
	int ref;
	if (label == id) {
		do { label = R[ref = label]; } while (ref ^ label);
		R[id] = label;
	}
}

__global__ void labeling(int D[], int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[R[L[id]]];
}

class CCL {
private:
	int* Dd;
	int* Ld;
	int* Rd;

public:
	int cuda_ccl(int *cc, int nrow, int ncol,  int degree_of_connectivity, int threshold);
	int cuda_removebifurcation(int *cc, int nrow, int ncol);
};


int CCL::cuda_removebifurcation(int *cc, int nrow, int ncol)
{
	float h_Kernel_p[KERNEL_W] = { 64,8,1,4,2,1 };
	float *h_Kernel = h_Kernel_p;
	float *h_Data, *h_Result;
	float *d_DataA, *d_DataB;
	int dw, dh;

	int data_size = nrow * ncol * sizeof(int);

	h_Data = (float*)malloc(nrow*ncol * sizeof(float));
	h_Result = (float*) malloc(nrow*ncol * sizeof(float));
	for (int i = 0; i < nrow*ncol; i++)
	{
		h_Data[i] = (float) cc[i];
	}

	cudaMalloc((void **)&d_DataA, data_size);
	cudaMalloc((void **)&d_DataB, data_size);

	cudaMemcpyToSymbol(d_Kernel, h_Kernel, KERNEL_SIZE);

	dw = ncol;
	dh = nrow;

	dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
	dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
	dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS);	// 16 128 8
	dim3 threadBlockColumns(COLUMN_TILE_W, 8);

	cudaMemcpy(d_DataA, h_Data, data_size, cudaMemcpyHostToDevice);

	//cudaThreadSynchronize();

	convolutionRowGPU << <blockGridRows, threadBlockRows >> >(
		d_DataB,
		d_DataA,
		dw,
		dh
		);

	convolutionColumnGPU << <blockGridColumns, threadBlockColumns >> >(
		d_DataA,
		d_DataB,
		dw,
		dh,
		COLUMN_TILE_W * threadBlockColumns.y,
		dw * threadBlockColumns.y
		);



	cudaMemcpy(h_Result, d_DataA, data_size, cudaMemcpyDeviceToHost);


	for (int i = 0; i < nrow*ncol; i++)
	{
		cc[i] = (int)h_Result[i];
	}

	cudaFree(d_DataB);
	cudaFree(d_DataA);

	return 0;
}

int CCL::cuda_ccl(int *cc, int nrow, int ncol, int degree_of_connectivity, int threshold)
{
	int* D = cc;// static_cast<int*>(&image[0]);
	int N = nrow*ncol;
	int W = ncol;

	cudaMalloc((void**)&Ld, sizeof(int) * N);
	cudaMalloc((void**)&Rd, sizeof(int) * N);
	cudaMalloc((void**)&Dd, sizeof(int) * N);
	cudaMemcpy(Dd, D, sizeof(int) * N, cudaMemcpyHostToDevice);

	bool* md;
	cudaMalloc((void**)&md, sizeof(bool));

	int width = static_cast<int>(sqrt(static_cast<double>(N) / BLOCK)) + 1;
	dim3 grid(width, width, 1);
	dim3 threads(BLOCK, 1, 1);

	init_CCL << <grid, threads >> >(Ld, Rd, N);

	for (;;) {
		bool m = false;
		cudaMemcpy(md, &m, sizeof(bool), cudaMemcpyHostToDevice);
		if (degree_of_connectivity == 4) scanning << <grid, threads >> >(Dd, Ld, Rd, md, N, W, threshold);
		else scanning8 << <grid, threads >> >(Dd, Ld, Rd, md, N, W, threshold);
		cudaMemcpy(&m, md, sizeof(bool), cudaMemcpyDeviceToHost);
		if (m) {
			analysis << <grid, threads >> >(Dd, Ld, Rd, N);
			//cudaThreadSynchronize();
			labeling << <grid, threads >> >(Dd, Ld, Rd, N);
		}
		else break;
	}

	cudaMemcpy(D, Ld, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(Dd);
	cudaFree(Ld);
	cudaFree(Rd);

	//result.swap(image);
	return 0;
}

void read_data(const string filename, vector<int>& image, int& W, int& degree_of_connectivity, int& threshold)
{
	fstream fs(filename.c_str(), ios_base::in);
	string line;
	stringstream ss;
	int data;

	getline(fs, line);
	ss.str(line);
	ss >> W >> degree_of_connectivity >> threshold;
	getline(fs, line);
	ss.str("");  ss.clear();
	for (ss.str(line); ss >> data; image.push_back(data));
}


int getconnectedcomponents(uint8_t* g_src_data0, uint8_t* g_dst_data0, int width, int height, int degree_of_connectivity, int *nbcomponents)
{
	int *cc = (int*)malloc(width * height * sizeof(int));
	int *cceof = (int*)malloc(width * height * sizeof(int));
	int *ccrembif = (int*)g_dst_data0;
 
	for (int i = 0; i < width * height; i++)
	{
		cc[i] = g_src_data0[i];
		cceof[i] = g_src_data0[i];
		ccrembif[i] = g_src_data0[i];
	}

	CCL ccl;

	//double start = get_time();
	ccl.cuda_removebifurcation(cc, width, height);
	//ccl.cuda_ccl(cc, W,W, degree_of_connectivity, threshold);
	//double end = get_time();
	//cerr << "Time: " << end - start << endl;


	short pattern[63] = { 58, 85, 113, 114, 115, 117, 121, 122, 125, 149, 151, 154, 156, 157, 158, 177, 178, 179, 181, 184, 185, 187, 189, 213, 215, 241, 242,
		243, 245, 277, 282, 284, 285, 286, 314,316, 317, 337, 338, 339, 340, 341, 342, 346, 348, 350, 369, 370, 371, 377, 378, 380, 381, 405,
		407, 410, 412, 413, 414, 466, 467, 470, 471
	};

	short patternsquare[4] = { 27, 54, 216, 432 };

	short patterneof[16] = { 17, 18, 19, 20, 22, 24, 25, 48, 52, 80, 88, 144, 208, 272, 304, 400 };

	for (int i = 0; i < width * height; i++)
	{
		unsigned char cour = 0;
		for (int j = 0; j < 63; j++) { cour = cour || cc[i] == pattern[j]; }
		for (int j = 0; j < 4; j++) { cour = cour || (cc[i] & patternsquare[j])== patternsquare[j]; }
		if (cour)
		{
			ccrembif[i] = 0;
			ccrembif[i + 1] = 0;
			ccrembif[i - 1] = 0;
			ccrembif[i + width] = 0;
			ccrembif[i - width] = 0;

			cceof[i] = 0;
			cceof[i + 1] = 0;
			cceof[i - 1] = 0;
			cceof[i + width] = 0;
			cceof[i - width] = 0;
		}
		//cour = 0;
		//for (int j = 0; j < 16; j++) { cour = cour || cc[i] == patterneof[j]; }
		//if (cour)
		//{
		//	ccrembif[i] = 0;
		//}
	}

	ccl.cuda_removebifurcation(cceof, width, height);


	ccl.cuda_ccl(ccrembif, width, height, degree_of_connectivity, 0);
	for (int i = 0; i < width * height; i++)
	{
		if (ccrembif[i] != 0)
		{
			*nbcomponents = max(ccrembif[i], *nbcomponents);
		}
	}

	int *tab = (int*)malloc(((*nbcomponents)+1) * sizeof(int));
	memset(tab, 0, ((*nbcomponents)+1) * sizeof(int));
	for (int i = 0; i < width * height; i++)
	{
		if (ccrembif[i] != 0)
		{
			tab[ccrembif[i]] = 1;
		}
	}
	for (int i = 1; i <= (*nbcomponents); i++)
	{
		tab[i] += tab[i - 1];
	}

	*nbcomponents = 0;
	for (int i = 0; i < width * height; i++)
	{
		if (ccrembif[i] != 0)
		{
			ccrembif[i] = tab[ccrembif[i]];
			*nbcomponents = max(ccrembif[i], *nbcomponents);
		}
	}

	for (int i = 0; i < width * height; i++)
	{
		unsigned char cour = 0;
		for (int j = 0; j < 16; j++) { cour = cour || cceof[i] == patterneof[j]; }
		if (cour)
		{
			ccrembif[i] = -ccrembif[i];
		}
	}


	free(cc);
	free(cceof);
	free(tab);

	return 0;
}

int main(int argc, char* argv[])
{
	ios_base::sync_with_stdio(false);

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " input_file" << endl;
		exit(1);
	}

	//cudaSetDevice(cutGetMaxGflopsDeviceId());

	int W=512, degree_of_connectivity=8, threshold=0;
	
	unsigned char rgbcolormap[3 * 11] = { 255,0,0,0,255,0,0,0,255,255,255,0,255,0,255,0,255,255, 128,128,128, 128,0,200,200,0,128,0,128,200,200,128,0 };
	
	FILE *fin = fopen(argv[1], "rb");
	unsigned char *imageuc = (unsigned char*)malloc(W*W);
	fread(imageuc, 1, W*W, fin);
	fclose(fin);
	int *cc = (int*)malloc(W * W * sizeof(int));
	int *ccrembif = (int*)malloc(W * W * sizeof(int));
	for (int i = 0; i < W*W; i++)
	{
		cc[i] = imageuc[i];
		ccrembif[i] = imageuc[i];
	}

	CCL ccl;

	double start = get_time();
	ccl.cuda_removebifurcation(cc, W, W);
	//ccl.cuda_ccl(cc, W,W, degree_of_connectivity, threshold);
	double end = get_time();
	cerr << "Time: " << end - start << endl;


	FILE *fout = fopen(argv[2], "wb");
	unsigned char *imageuc3c = (unsigned char*)malloc(W*W*3);
	unsigned char *ccimageuc3c = (unsigned char*)malloc(W*W * 3);

	memset(ccimageuc3c, 0, W*W * 3);

	short pattern[63] = {58, 85, 113, 114, 115, 117, 121, 122, 125, 149, 151, 154, 156, 157, 158, 177, 178, 179, 181, 184, 185, 187, 189, 213, 215, 241, 242,
						243, 245, 277, 282, 284, 285, 286, 314,316, 317, 337, 338, 339, 340, 341, 342, 346, 348, 350, 369, 370, 371, 377, 378, 380, 381, 405,
						407, 410, 412, 413, 414, 466, 467, 470, 471 
						};
	for (int i = 0; i < W*W; i++)
	{
		imageuc3c[i * 3 + 1] = imageuc[i];
		imageuc3c[i * 3 + 0] = imageuc[i];
		imageuc3c[i * 3 + 2] = imageuc[i];
	}
	for (int i = 0; i < W*W; i++)
	{
		unsigned char cour = 0;
		for (int j = 0; j < 63; j++) { cour = cour || cc[i] == pattern[j]; }
		if (cour)
		{
			imageuc3c[(i - 1) * 3 + 1] = 0;
			imageuc3c[(i + 1) * 3 + 1] = 0;
			imageuc3c[(i - W) * 3 + 1] = 0;
			imageuc3c[(i + W) * 3 + 1] = 0;
			imageuc3c[(i - 1) * 3 + 2] = 0;
			imageuc3c[(i + 1) * 3 + 2] = 0;
			imageuc3c[(i - W) * 3 + 2] = 0;
			imageuc3c[(i + W) * 3 + 2] = 0;
			imageuc3c[i * 3 + 1] = 0;
			imageuc3c[i * 3 + 2] = 0;
			ccrembif[i] = 0;
			ccrembif[i+1] = 0;
			ccrembif[i-1] = 0;
			ccrembif[i+W] = 0;
			ccrembif[i-W] = 0;
		}
	}
	fwrite(imageuc3c, 1, W*W*3, fout);
	fclose(fout);


	ccl.cuda_ccl(ccrembif, W, W, degree_of_connectivity, threshold);

	int nbcomponents=0;
	for (int i = 0; i < W * W; i++)
	{
		if (ccrembif[i] != 0)
		{
			nbcomponents = max(ccrembif[i], nbcomponents);
		}
	}

	int *tab = (int*)malloc((nbcomponents+1)*sizeof(int));
	memset(tab, 0, (nbcomponents+1) * sizeof(int));
	for (int i = 0; i < W * W; i++)
	{
		if (ccrembif[i] != 0)
		{
			tab[ccrembif[i]]=1;
		}
	}
	for (int i = 1; i <= nbcomponents; i++)
	{
		tab[i] += tab[i - 1];
	}

	nbcomponents = 0;
	for (int i = 0; i < W * W; i++)
	{
		if (ccrembif[i] != 0)
		{
			ccrembif[i] = tab[ccrembif[i]];
			nbcomponents = max(ccrembif[i], nbcomponents);
		}
	}


	for (int i = 0; i < W*W; i++)
	{
		int color = ccrembif[i] % 11;
		if (ccrembif[i] != 0)
		{
			ccimageuc3c[i * 3 + 0] = rgbcolormap[color * 3];
			ccimageuc3c[i * 3 + 1] = rgbcolormap[color * 3 + 1];
			ccimageuc3c[i * 3 + 2] = rgbcolormap[color * 3 + 2];
		}
	}

	FILE *fout2 = fopen(argv[3], "wb");
	fwrite(ccimageuc3c, 1, W*W * 3, fout2);
	fclose(fout2);

	free(imageuc);
	free(imageuc3c);
	free(ccimageuc3c);
	free(cc);
	free(ccrembif);

	return 0;
}
