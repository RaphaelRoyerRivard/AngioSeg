#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

#include "interface.h"



void va_thinningIteration(unsigned char *im, int row, int col, int iter)
{
	unsigned char *marker = (unsigned char*) malloc(row*col);

	const int NbRows = row;
	const int NbCols = col;
	const int JStart = 1;

	

	int i;

	memset(marker, 0, row*col);
#pragma omp parallel private( i ) shared( im, marker )
	{
		const unsigned char *p_p1;
		const unsigned char *p_p2;
		const unsigned char *p_p3;
		const unsigned char *p_p4;
		const unsigned char *p_p5;
		const unsigned char *p_p6;
		const unsigned char *p_p7;
		const unsigned char *p_p8;
		const unsigned char *p_p9;
		unsigned char * p_marker = marker+1*NbCols;

#pragma omp for schedule( guided ) nowait
		for (i = 1; i < NbRows-1; i++)
		{
			int j;
			p_p1 = im +i*NbCols+ JStart;
			p_p2 = im +(i-1)*NbCols + JStart;
			p_p3 = p_p2 + 1;
			p_p4 = p_p1 + 1;
			p_p6 = im+(i+1)*NbCols + JStart;
			p_p5 = p_p6 + 1;
			p_p7 = p_p6 - 1;
			p_p8 = p_p1 - 1;
			p_p9 = p_p2 - 1;
			p_marker = marker+i*NbCols + JStart;
			for (j = JStart; j < NbCols-1; j++, ++p_p1, ++p_p2, ++p_p3, ++p_p4, ++p_p5, ++p_p6, ++p_p7, ++p_p8, ++p_p9, ++p_marker )
			{
				int A  = (*p_p2 == 0 && *p_p3 != 0) + (*p_p3 == 0 && *p_p4 != 0) +
						 (*p_p4 == 0 && *p_p5 != 0) + (*p_p5 == 0 && *p_p6 != 0) +
						 (*p_p6 == 0 && *p_p7 != 0) + (*p_p7 == 0 && *p_p8 != 0) +
						 (*p_p8 == 0 && *p_p9 != 0) + (*p_p9 == 0 && *p_p2 != 0);
				int B  = *p_p2 + *p_p3 + *p_p4 + *p_p5 + *p_p6 + *p_p7 + *p_p8 + *p_p9;
				int m1 = iter == 0 ? ((*p_p2) * (*p_p4) * (*p_p6)) : ((*p_p2) * (*p_p4) * (*p_p8));
				int m2 = iter == 0 ? ((*p_p4) * (*p_p6) * (*p_p8)) : ((*p_p2) * (*p_p6) * (*p_p8));

				if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
					*p_marker = 1;
			}
		}
	}

	for (i=0; i<row*col; i++)
	{
		im[i]&=~marker[i];
	}

	free(marker);
}

void va_thinning(unsigned char *im, int row, int col)
{
	unsigned char *prev = (unsigned char*) malloc(row*col);
	int i=0, countnonzeros=0;

	for (i=0; i<row*col; i++)
	{
		if (im[i]>128)
			im[i] = 1;

		else
			im[i]=0;
	}

	memset(prev, 0, row*col);
	do {
		
		va_thinningIteration(im, row,col,0);
		va_thinningIteration(im, row, col,1);

		countnonzeros=0;
		for (i=0; i<row*col; i++)
		{
			if (im[i]-prev[i])
				countnonzeros++;
		}
		memcpy(prev, im, row*col);
	}
	while (countnonzeros > 0);

	free(prev);
}

int va_morpho_isinimage_kernelsz(int i, int j, int nrow, int ncol, int szrowk, int szcolk)
{
	int flag=0;
	if (i-szrowk>=0 && j-szcolk>=0 && i+szrowk<nrow && j+szcolk<ncol)
		flag=1;
	
	return flag;
}


int va_morpho_dilate(const unsigned char *src, int nrow, int ncol, unsigned char *dst, char *kernel, int nrowk, int ncolk)
{
	int i,j, ki, kj;
	int szrowk, szcolk;

	szrowk=(nrowk-1)/2;
	szcolk=(ncolk-1)/2;
    for (i=0; i<nrow; i++){
        for (j=0; j<ncol; j++){
			int val=src[i*ncol+j];
            if (va_morpho_isinimage_kernelsz(i,j,nrow,ncol,szrowk,szcolk))
			{
				val=0;
				for (ki=-szrowk; ki<=szrowk;ki++)
				{
					for (kj=-szcolk; kj<=szcolk;kj++)
					{
						if (kernel[(ki+szrowk)*ncolk+ kj+szcolk]==1)
						{
							if (val<src[(i+ki)*ncol+j+kj])
							{
								val=src[(i+ki)*ncol+j+kj];
							}
						}
					}
				}
				
            }
			dst[i*ncol+j]=val;
        }
    }

    return 0;

}

int va_morpho_erode(const unsigned char *src, int nrow, int ncol, unsigned char *dst, char *kernel, int nrowk, int ncolk)
{
	int i,j, ki, kj;
	int szrowk, szcolk;

	szrowk=(nrowk-1)/2;
	szcolk=(ncolk-1)/2;
    for (i=0; i<nrow; i++)
	{
        for (j=0; j<ncol; j++)
		{
			int val=src[i*ncol+j];
            if (va_morpho_isinimage_kernelsz(i,j,nrow,ncol,szrowk,szcolk))
			{
				val=1;
				for (ki=-szrowk; ki<=szrowk;ki++)
				{
					for (kj=-szcolk; kj<=szcolk;kj++)
					{
						if (kernel[(ki+szrowk)*ncolk+ kj+szcolk]==1)
						{
							if (val>src[(i+ki)*ncol+j+kj])
							{
								val=src[(i+ki)*ncol+j+kj];
							}
						}
					}
				}
				
            }
			dst[i*ncol+j]=val;
        }
    }

    return 0;

}

int va_morpho_image_and(unsigned char *src1, unsigned char *src2,int nrow, int ncol, unsigned char *dst)
{
	int i;
	for (i=0; i<nrow*ncol;i++)
	{
		dst[i]=src1[i]&src2[i];
	}
	return 0;
}

int va_morpho_image_or(unsigned char *src1, unsigned char *src2,int nrow, int ncol, unsigned char *dst)
{
	int i;
	for (i=0; i<nrow*ncol;i++)
	{
		dst[i]=src1[i]|src2[i];
	}
	return 0;
}

int va_morpho_image_inverse(const unsigned char *src, int nrow, int ncol, int value, unsigned char *dst)
{
	int i;
	for (i=0; i<nrow*ncol;i++)
	{
		dst[i]=value-src[i];
	}
	return 0;
}

int va_morpho_hitmiss(const unsigned char *src, int nrow, int ncol, unsigned char *dst, char *kernel, int nrowk, int ncolk)
{

	unsigned char *e1,*e2, *inv_src;
	char *k1, *k2;
	int i;

	k1=(char *) malloc(nrowk*ncolk);
	k2=(char *) malloc(nrowk*ncolk);
	e1=(unsigned char *) malloc(nrow*ncol);
	e2=(unsigned char *) malloc(nrow*ncol);
	inv_src=(unsigned char *) malloc(nrow*ncol);

	memset(k1,0, nrowk*ncolk);
	memset(k2,0, nrowk*ncolk);

	for (i=0; i<nrowk*ncolk; i++)
	{
		if (kernel[i]==1)
		{
			k1[i]=1;
		}
		else if (kernel[i]==-1)
		{
			k2[i]=1;
		}
	}

	va_morpho_image_inverse(src, nrow, ncol, 1, inv_src);

	va_morpho_erode(src, nrow, ncol, e1, k1, nrowk, ncolk);

	//vtatlas_write_bmp_void(e1, ncol, nrow, "uc","test.e1.bmp");
	va_morpho_erode(inv_src, nrow, ncol, e2, k2, nrowk, ncolk);

	//vtatlas_write_bmp_void(e2, ncol, nrow, "uc", "test.e2.bmp");
	va_morpho_image_and(e1,e2,nrow,ncol,dst);

	//vtatlas_write_bmp_void(dst, ncol, nrow, "uc","test.e1e2.bmp");

	free(e1);
	free(e2);
	free(k1);
	free(k2);
	free(inv_src);

	return 0;
}

#define CALL_morpho_cc_label_component(x,y,returnLabel) { STACK[SP] = x; STACK[SP+1] = y; STACK[SP+2] = returnLabel; SP += 3; goto START; }
#define RETURN { SP -= 3;                \
                 switch (STACK[SP+2])    \
                 {                       \
                 case 1 : goto RETURN1;  \
                 case 2 : goto RETURN2;  \
                 case 3 : goto RETURN3;  \
                 case 4 : goto RETURN4;  \
                 default: return;        \
                 }                       \
               }
#define X (STACK[SP-3])
#define Y (STACK[SP-2])

void vtatlas_morpho_cc_label_component(unsigned short* STACK, unsigned int ncol, unsigned int nrow, 
const unsigned char* input, unsigned char* output, unsigned short labelNo, unsigned short x, unsigned short y)
{
	int SP = 3;
	int index;

	STACK[0] = x;
	STACK[1] = y;
	STACK[2] = 0;  /* return - component is labelled */

	START: /* Recursive routine starts here */

		index = X + ncol*Y;
		if (input [index] == 0) RETURN;   /* This pixel is not part of a component */
		if (output[index] != 0) RETURN;   /* This pixel has already been labelled  */
		output[index] = labelNo;

		if (X > 0) CALL_morpho_cc_label_component(X-1, Y, 1);   /* left  pixel */
	RETURN1:

		if (X < ncol-1) CALL_morpho_cc_label_component(X+1, Y, 2);   /* right pixel */
	RETURN2:

		if (Y > 0) CALL_morpho_cc_label_component(X, Y-1, 3);   /* upper pixel */
	RETURN3:

		if (Y < nrow-1) CALL_morpho_cc_label_component(X, Y+1, 4);   /* lower pixel */
	RETURN4:

		RETURN;
}

int va_morpho_cc_connected_component(const unsigned char *src, int nrow, int ncol, unsigned char *dst, unsigned char *plabelNo)
{
	unsigned short* buffer = (unsigned short*) malloc(3*sizeof(unsigned short)*(nrow*ncol + 1));
	unsigned short x,y;
	unsigned short labelNo = 0;
	int index   = -1;
	for (y = 0; y < nrow; y++)
	{
		for (x = 0; x < ncol; x++)
		{
			index++;
			if (src[index] == 0) continue;   /* This pixel is not part of a component */
			if (dst[index] != 0) continue;   /* This pixel has already been labelled */
			/* New component found */
			labelNo++;
			vtatlas_morpho_cc_label_component(buffer, ncol, nrow, src, dst, labelNo, x, y);
		}
	}

	plabelNo[0] = labelNo;
	free(buffer);
	return 0;
}


int va_spur_pruning(unsigned char *img, int nrow, int ncol, int n, unsigned char *dest)
{
	int i=0, j=0;

	// on thin n fois, n etant la taille des barbules
	char element[8][9]={{0,-1 , -1 , 1 , 1 ,-1 , 0 , -1 , -1},{0, 1 , 0 , -1 , 1 ,-1 , -1 ,-1 , -1}, {-1,-1 , 0 ,-1 , 1 , 1 ,-1 ,-1 , 0  },{-1,-1 ,-1 ,-1 , 1 ,-1 , 0 , 1 , 0 },
						{ 1,-1 ,-1 ,-1 , 1 ,-1 ,-1 ,-1 ,-1}, {-1,-1 , 1 ,-1 , 1 ,-1 ,-1 ,-1 ,-1 }, {-1,-1 ,-1 ,-1 , 1 ,-1 ,-1 ,-1 , 1}, {-1,-1 ,-1 ,-1 , 1 ,-1 , 1 ,-1 ,-1}};

	char element_dilate[9] = {1,1,1,1,1,1,1,1,1};

	unsigned char *temp;
	unsigned char *eroded;
	unsigned char *inv_eroded;
	unsigned char *union_endpoint;

	temp = (unsigned char*) malloc(nrow*ncol);
	memcpy(temp, img, nrow*ncol);

	eroded = (unsigned char*) malloc(nrow*ncol);
	inv_eroded = (unsigned char*) malloc(nrow*ncol);

	for (i=0; i<n; i++)
	{
		for (j=0; j<8; j++)
		{
			// thinning A \inter (1-hitandmiss(A, Bi))
			va_morpho_hitmiss(temp, nrow, ncol, eroded, element[j], 3, 3);
			va_morpho_image_inverse(eroded, nrow, ncol, 1, inv_eroded);
			va_morpho_image_and(temp,inv_eroded,nrow,ncol,eroded);
			memcpy(temp, eroded, nrow*ncol);
		}
	}

	union_endpoint = (unsigned char*) malloc(nrow*ncol);
	memset(union_endpoint, 0, nrow*ncol);

	for (j=0; j<8; j++)
	{
		va_morpho_hitmiss(temp, nrow, ncol, eroded, element[j],3,3);
		va_morpho_image_or(union_endpoint,eroded,nrow,ncol,inv_eroded);
		memcpy(union_endpoint, inv_eroded, nrow*ncol);
	}


	//for (i=0; i<n; i++)
	//{
	//	va_morpho_dilate(union_endpoint, nrow, ncol, eroded, element_dilate, 3,3);
	//	va_morpho_image_and(eroded,img,nrow,ncol,union_endpoint);

	//}


	va_morpho_image_or(temp,union_endpoint,nrow,ncol,dest);

	free(union_endpoint);
	free(inv_eroded);
	free(eroded);
	free(temp);

	return 0;
}

/*
void spur_pruning_and_dilate(cv::Mat& img, int n)
{
	Mat dest;
	int i=0, j=0;
	// on thin n fois, n etant la taille des barbules
	char element[8][9]={{0,-1 , -1 , 1 , 1 ,-1 , 0 , -1 , -1},{0, 1 , 0 , -1 , 1 ,-1 , -1 ,-1 , -1}, {-1,-1 , 0 ,-1 , 1 , 1 ,-1 ,-1 , 0  },{-1,-1 ,-1 ,-1 , 1 ,-1 , 0 , 1 , 0 },
						{ 1,-1 ,-1 ,-1 , 1 ,-1 ,-1 ,-1 ,-1}, {-1,-1 , 1 ,-1 , 1 ,-1 ,-1 ,-1 ,-1 }, {-1,-1 ,-1 ,-1 , 1 ,-1 ,-1 ,-1 , 1}, {-1,-1 ,-1 ,-1 , 1 ,-1 , 1 ,-1 ,-1}};

	cv::Mat temp;
	cv::Mat eroded;

	img.copyTo(temp);
	for (i=0; i<n; i++)
	{
		for (j=0; j<8; j++)
		{
			// thinning A \inter (1-hitandmiss(A, Bi))
			hitmiss(temp, eroded, element[j]);
			eroded = 1- eroded;
			temp = temp & eroded;

		}

	}

	Mat union_endpoint(img.size(), CV_8UC1, cv::Scalar(0));
	for (j=0; j<8; j++)
	{
		hitmiss(temp, eroded, element[j]);
		union_endpoint |=eroded;
	}

	Mat kernel_dilate = (cv::Mat_<char>(3,3) << 1,1 ,1 , 1 , 1 ,1 , 1 ,1 ,1 );
	Mat temp2;
	for (i=0; i<n; i++)
	{
		dilate(union_endpoint, eroded, kernel_dilate, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		union_endpoint = eroded & img;
		temp2 = temp | union_endpoint;

	}


	img = temp | union_endpoint;


}*/


int va_turn_skeleton_into_4_connectivity(unsigned char *skeleton, int nrow, int ncol, unsigned char *skeleton4c)
{
	int j=0, i=0, k=0;
	//char kernel[4][9]={{0,-1 , 1 , 0 , 1 ,-1 , 0 , 0 , 0},{0, 0 , 0 , 0 , 1 ,-1 , 0 ,-1 , 1}, {0, 0 , 0 ,-1 , 1 , 0 , 1 ,-1 , 0 },{1,-1 , 0 ,-1 , 1 , 0 , 0 , 0 , 0 }};
	char kernel[4][9]={{0,1 , -1 , 0 , -1 ,1 , 0 , 0 , 0},{0, 0 , 0 , 0 , -1 ,1 , 0 ,1 , -1}, {0, 0 , 0 ,1 , -1 , 0 , -1 ,1 , 0 },{-1,1 , 0 ,1 , -1 , 0 , 0 , 0 , 0 }};

	char kernelspur1[4][9]={{-1,-1 , -1 , 0 , 1 ,0 , 1 , 1 , 1},{1, 1 , 1 , 0 , 1 ,0 , -1 ,-1 , -1}, {-1, 0 , 1 ,-1 , 1 , 1 , -1 ,0 , 1 },{1,0 , -1 ,1 , 1 , -1 , 1 , 0 , -1 }};
	char kernelspur2[4][9]={{-1,-1 , -1 , 0 , 1 ,0 , 1 , 1 , 1},{1, 1 , 1 , 0 , 1 ,0 , -1 ,-1 , -1}, {-1, 0 , 1 ,-1 , 1 , 1 , -1 ,0 , 1 },{1,0 , -1 ,1 , 1 , -1 , 1 , 0 , -1 }};
	int nrowk=3, ncolk=3;
	unsigned char *eroded = (unsigned char*)malloc(nrow*ncol);
	
	memcpy(skeleton4c, skeleton, nrow*ncol);

	for (k=0; k<4; k++)
	{
		va_morpho_hitmiss(skeleton4c, nrow, ncol, eroded, kernel[k], nrowk, ncolk );
		for (i=0; i<nrow; i++)
		{
			for (j=0; j<ncol; j++)
			{
				if (eroded[i*ncol+j])
				{
					skeleton4c[i*ncol+j]=1;		
				}
			}
		}

	}

	for (k=0; k<4; k++)
	{
		va_morpho_hitmiss(skeleton4c, nrow, ncol, eroded, kernelspur1[k], nrowk, ncolk );
		for (i=0; i<nrow; i++)
		{
			for (j=0; j<ncol; j++)
			{
				if (eroded[i*ncol+j])
				{
					skeleton4c[i*ncol+j]=0;		
				}
			}
		}

	}

	return 0;
}

int vesselanalysis_getskeleton(unsigned char *seg, int nrow, int ncol, unsigned char *skeleton)
{
	unsigned char *seg4c = (unsigned char*)malloc(nrow*ncol);
	int i=0;

	memcpy(skeleton, seg, nrow*ncol);

	va_thinning(skeleton, nrow, ncol);

	va_turn_skeleton_into_4_connectivity(skeleton, nrow, ncol, seg4c);

	//for (i=0; i<ncol*nrow; i++){skeleton[i] = seg4c[i]*255;}

	free(seg4c);

	return 0;
}

int va_get_connectivity(unsigned char *seg, int row, int col, int nrow, int ncol, int *connect)
{
	*connect=0;
	if (row-1 >= 0) {*connect+=(seg[(row-1)*ncol+col]>0);}
	if (col-1 >= 0) {*connect+=(seg[row*ncol+col-1]>0);}
	if (row+1 < nrow) {*connect+=(seg[(row+1)*ncol+col]>0);}
	if (col+1 < ncol) {*connect+=(seg[row*ncol+col+1]>0);}

	return 0;
}

int va_get_connectivity4_int(int *cc, unsigned char *seg, int row, int col, int nrow, int ncol, int *connect)
{
	*connect=0;
	if (row-1 >= 0) {*connect+=(seg[(row-1)*ncol+col]>0 && cc[(row-1)*ncol+col]==0);}
	if (col-1 >= 0) {*connect+=(seg[row*ncol+col-1]>0 && cc[row*ncol+col-1]==0);}
	if (row+1 < nrow) {*connect+=(seg[(row+1)*ncol+col]>0 && cc[(row+1)*ncol+col]==0);}
	if (col+1 < ncol) {*connect+=(seg[row*ncol+col+1]>0 && cc[row*ncol+col+1]==0);}

	return 0;
}

int va_get_connectivity8(unsigned char *seg, int row, int col, int nrow, int ncol, int *connect)
{
	*connect = 0;
	if (row - 1 >= 0) { *connect += (seg[(row - 1)*ncol + col]>0); }
	if (col - 1 >= 0) { *connect += (seg[row*ncol + col - 1]>0); }
	if (row + 1 < nrow) { *connect += (seg[(row + 1)*ncol + col]>0); }
	if (col + 1 < ncol) { *connect += (seg[row*ncol + col + 1]>0); }

	if (col + 1 < ncol && row - 1 >= 0) { *connect += (seg[(row-1)*ncol + col + 1]>0); }
	if (col + 1 < ncol && row + 1 < nrow) { *connect += (seg[(row+1)*ncol + col + 1]>0); }
	if (col - 1 >= 0 && row - 1 >= 0) { *connect += (seg[(row-1)*ncol + col - 1]>0); }
	if (col - 1 >= 0 && row + 1 < nrow) { *connect += (seg[(row+1)*ncol + col - 1]>0); }

	return 0;
}

int va_get_connectivity8_int(int *cc, unsigned char *seg, int row, int col, int nrow, int ncol, int *connect)
{
	*connect = 0;

	if (col + 1 < ncol && row - 1 >= 0) { *connect += (seg[(row-1)*ncol + col + 1]>0 && cc[(row-1)*ncol + col + 1]==0); }
	if (col + 1 < ncol && row + 1 < nrow) { *connect += (seg[(row+1)*ncol + col + 1]>0 && cc[(row+1)*ncol + col + 1]==0); }
	if (col - 1 >= 0 && row - 1 >= 0) { *connect += (seg[(row-1)*ncol + col - 1]>0 && cc[(row-1)*ncol + col - 1]==0); }
	if (col - 1 >= 0 && row + 1 < nrow) { *connect += (seg[(row+1)*ncol + col - 1]>0 && cc[(row+1)*ncol + col - 1]==0); }

	return 0;
}

int va_segment_detect_endline(unsigned char *seg, int nrow, int ncol)
{
	int i,j;
	int connect;

	for (i=0; i<nrow; i++)
	{
		for (j=0; j<ncol; j++)
		{
			if (seg[i*ncol+j]==1)
			{
				va_get_connectivity(seg, i, j, nrow, ncol, &connect);
			// on conserve les fins de ligne place est importante ici
				if (connect==1)
				{
					seg[i*ncol+j]=255;
				}
			}
		}
	}


	return 0;
}

int va_segment_delete_bifurcation(unsigned char *seg, int nrow, int ncol)
{
	int i,j;
	int connect;

	unsigned char *segcour=(unsigned char*) malloc(nrow*ncol);

	memcpy(segcour, seg, nrow*ncol);

	for (i=0; i<nrow; i++)
	{
		for (j=0; j<ncol; j++)
		{
			if (seg[i*ncol+j]==1)
			{
				va_get_connectivity(seg, i, j, nrow, ncol, &connect);
				if (connect>2)
				{
					segcour[i*ncol+j]=0;
				}
			}

		}
	}

	memcpy(seg, segcour, nrow*ncol);
	free(segcour);
	return 0;
}

int vesselanalysis_getcomponents(unsigned char *skel,  int nrow, int ncol, unsigned char *nbcomponents, unsigned char *vessels_component)
{
	unsigned char *vessels_seg4c = (unsigned char*) malloc(nrow*ncol);

	//memcpy(vessels_seg4c, skel, nrow*ncol);

	va_turn_skeleton_into_4_connectivity(skel, nrow, ncol, vessels_seg4c);
	//va_segment_detect_endline(vessels_seg4c, nrow, ncol);

	va_segment_delete_bifurcation(vessels_seg4c, nrow, ncol);

	va_morpho_cc_connected_component(vessels_seg4c, nrow, ncol, vessels_component, nbcomponents);

	free(vessels_seg4c);

	return 0;
}


int vesselanalysis_getstats(unsigned char *cc, unsigned char *dft, unsigned char *skel, int nrow, int ncol, int nbelem,  int nbcomponents, unsigned char *buffer)
{
	int *vessels_components = (int*)cc;
	float *stats = (float*)buffer;
	float *diam = (float*)dft;


	memset(stats, 0, nbcomponents * nbelem * sizeof(float));


	
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			int nco=0;
			if (vessels_components[i*ncol + j]<0)
			{
				nco = -vessels_components[i*ncol + j]-1;
				if (stats[nco * nbelem + 0]==0 && stats[nco * nbelem + 1]==0)
				{
					stats[nco * nbelem + 0]=i;
					stats[nco * nbelem + 1]=j;
				}
				else if (stats[nco * nbelem + 2]==0 && stats[nco * nbelem + 3]==0)
				{
					stats[nco * nbelem + 2]=i;
					stats[nco * nbelem + 3]=j;
				}
				vessels_components[i*ncol + j] = -vessels_components[i*ncol + j];
			}
		}
	}

	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			if (vessels_components[i*ncol + j])
			{
				int nco = vessels_components[i*ncol + j] - 1;
				//int connect=0;
				//va_get_connectivity4_int(vessels_components, skel,  i, j, nrow, ncol, &connect);
				//if (connect==1 && stats[nco * nbelem + 0]==0 && stats[nco * nbelem + 1]==0)
				//{
				//	stats[nco * nbelem + 0]=i;
				//	stats[nco * nbelem + 1]=j;
				//}
				//else if (connect==1 && stats[nco * nbelem + 2]==0 && stats[nco * nbelem + 3]==0)
				//{
				//	stats[nco * nbelem + 2]=i;
				//	stats[nco * nbelem + 3]=j;
				//}
				stats[nco * nbelem + 4] += diam[i*ncol + j];
				stats[nco * nbelem + 5]++;
			}
		}
	}

	//for (int i = 0; i < nrow; i++)
	//{
	//	for (int j = 0; j < ncol; j++)
	//	{
	//		if (vessels_components[i*ncol + j])
	//		{
	//			int nco = vessels_components[i*ncol + j] - 1;
	//			int connect=0;
	//			int connect4=0;
	//			// s'il en reste 
	//			if (stats[nco * nbelem + 5]>1 && ((stats[nco * nbelem + 0]==0 && stats[nco * nbelem + 1]==0) || (stats[nco * nbelem + 2]==0 && stats[nco * nbelem + 3]==0)))
	//			{
	//				va_get_connectivity8_int(vessels_components, skel,  i, j, nrow, ncol, &connect);
	//				va_get_connectivity4_int(vessels_components, skel,  i, j, nrow, ncol, &connect4);
	//				if (connect==1 && connect4!=1 && stats[nco * nbelem + 0]==0 && stats[nco * nbelem + 1]==0)
	//				{
	//					stats[nco * nbelem + 0]=i;
	//					stats[nco * nbelem + 1]=j;
	//				}
	//				else if (connect==1 && connect4!=1 && stats[nco * nbelem + 2]==0 && stats[nco * nbelem + 3]==0)
	//				{
	//					stats[nco * nbelem + 2]=i;
	//					stats[nco * nbelem + 3]=j;
	//				}
	//				else
	//				{
	//					printf("dmfsdfsdjfl\n"); fflush(stdout);
	//				}
	//			}

	//		}
	//	}
	//}

	return 0;
}

//int vesselanalysis_threshold_diameter(int * vessels_component, int nbcomponents, )
