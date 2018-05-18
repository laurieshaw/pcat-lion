#include <stdlib.h>
#include <stdbool.h>
#include "mkl_cblas.h"
#include "i_malloc.h"
#include <stdio.h>
#define max(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a > _b ? _a : _b; })
#define min(a,b) \
    ({ typeof (a) _a = (a);    \
	typeof (b) _b = (b);   \
        _a < _b ? _a : _b; })

// LU decomoposition of a general matrix
//void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
//void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);

void invert_matrix(double * O, double * WORK, int * IPIV, int N) {
  int INFO;
  int LWORK = N*N;
  
  dgetrf_(&N,&N,&O[0],&N,&IPIV[0],&INFO);
  dgetri_(&N,&O[0],&N,&IPIV[0],&WORK[0],&LWORK,&INFO);
}

void pcat_imag_acpt(int NX, int NY, float* image, float* image_acpt, int* reg_acpt, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                if (reg_acpt[j*NREGX+i] > 0) {
                    for (jj=y0 ; jj<y1; jj++)
                     for (ii=x0 ; ii<x1; ii++)
                        image_acpt[jj*NX+ii] = image[jj*NX+ii];
                }
        }
    }
}

void pcat_like_eval(int NX, int NY, float* image, float* ref, float* weight, double* diff2, int regsize, int margin, int offsetx, int offsety) {
    int NREGX = (NX / regsize) + 1;
    int NREGY = (NY / regsize) + 1;
    int y0, y1, x0, x1, i, j, ii, jj;
    for (j=0 ; j < NREGY ; j++) {
        y0 = max(j*regsize-offsety-margin, 0);
        y1 = min((j+1)*regsize-offsety+margin, NY);
        for (i=0 ; i < NREGX ; i++) {
                x0 = max(i*regsize-offsetx-margin, 0);
                x1 = min((i+1)*regsize-offsetx+margin, NX);
                diff2[j*NREGX+i] = 0.;
                for (jj=y0 ; jj<y1; jj++)
                 for (ii=x0 ; ii<x1; ii++)
                    diff2[j*NREGX+i] += (image[jj*NX+ii]-ref[jj*NX+ii])*(image[jj*NX+ii]-ref[jj*NX+ii]) * weight[jj*NX+ii];
        }
    }
}

void pcat_source_cov(int NX, int NY, int nstar, int nc, int* ix, int* iy, float* P, double* C) 
{
  int istar, jstar, i, j, i2, j2, rad, ixmin, ixmax, iymin, iymax, s1y_low, s1y_hi, s2y_low, s2y_hi;
  int xxi,xxj,yyi,yyj,xsep,ysep,rsep2;
  int npix_overlap;
  rad = nc/2;
  for (istar=0;istar<nstar;istar++) {
    for (jstar=istar;jstar<nstar;jstar++) {
      xsep = ix[istar]-ix[jstar]; ysep =  iy[istar]-iy[jstar];
      rsep2 = xsep*xsep + ysep*ysep;
      if ( rsep2 < 2.0*nc*nc ) { // postage stamps do overlap
	xxi = ix[istar];
	xxj = ix[jstar];
	yyi = iy[istar];
	yyj = iy[jstar];
	ixmin = max( max(xxi,xxj)-rad, 0);
	ixmax = min( min(xxi,xxj)+rad+1, NX);
	iymin = max( max(yyi,yyj)-rad, 0);
	iymax = min( min(yyi,yyj)+rad+1, NY);
	npix_overlap = (ixmax-ixmin)*(iymax-iymin);
	if (npix_overlap>0) { // they really should overlap give above condition. test.
	  s1y_low = iymin-yyi+rad;
	  s1y_hi = iymax-yyi+rad;
	  s2y_low = iymin-yyj+rad;
	  s2y_hi = iymax-yyj+rad;
	  //printf("%d,%d\n",s2y_low,s2y_hi);
	  for ( j = (istar*nc + s1y_low )*nc, j2 = (jstar*nc + s2y_low)*nc; j < (istar*nc + s1y_hi)*nc; j+=nc, j2+=nc  ) {
	    for ( i = ixmin-xxi+rad, i2 = ixmin-xxj+rad; i < ixmax-xxi+rad; i++,i2++ ){
	      C[istar*nstar+jstar] += (double) P[i+j]* (double) P[i2+j2];
	    }
	  }
	  if (istar!=jstar) C[jstar*nstar+istar] = C[istar*nstar+jstar];
	}
      }
    }
  }
}

void pcat_source_images(int nstar, int nc, int k, float* A, float* B, float* C)
{
    float    alpha, beta;
    int n = nc*nc;
    alpha = 1.0; beta = 0.0;
    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,nstar, n, k, alpha, A, k, B, n, beta, C, n);
}

void pcat_flux_estimate(int NX, int NY, int nstar, int nc, int k, float back, int* ix, int* iy, float* dd, float* cf, float* P, double* C, double* Cinv, double* WORK, int *IPIV, float* image, float* fluxes, float* freg) 
{
  // P is postage stamp images, C is covariance matrix
  // loop over stars, sum image*psf at each source position
  int istar, rad, xx, yy, imax, jmax, i, i2, j, j2;
  double alpha, beta;
  // cblas params
  alpha = 1.0; beta = 0.0;
  pcat_source_images(nstar,nc,k,dd,cf,P);
  pcat_source_cov(NX,NY,nstar,nc,ix,iy,P,C); 
  for (istar = 0 ; istar < nstar*nstar ; istar++ ) { // use memcpy() ?
    Cinv[istar] = C[istar];
  }
  invert_matrix(Cinv, WORK, IPIV, nstar);
  rad = nc/2;
  for (istar = 0 ; istar < nstar ; istar++) {
    xx = ix[istar];
    yy = iy[istar];
    imax = min(xx+rad,NX-1);
    jmax = min(yy+rad,NY-1);
    for (j = max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
      for (i = max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
	fluxes[istar] += (image[j*NX+i]-back)*P[i2+j2];
  }
  // now calculate fluxes
  for (i = 0; i<nstar; i++) {
    for (j = 0; j<nstar; j++) {
      freg[i] += Cinv[i*nstar+j]*fluxes[j];
    }
  }
}

void pcat_model_eval(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
	int* y, float* image, float* ref, float* weight, double* diff2, int regsize, int margin,
	int offsetx, int offsety)
{
    int      i,i2,imax,j,j2,jmax,rad,istar,xx,yy;
    float    alpha, beta;

    int n = nc*nc;
    rad = nc/2;

    // cblas params
    alpha = 1.0; beta = 0.0;

    // overwrite and shorten A matrix
    // save time if there are many sources per pixel
    int hash[NY*NX];
    for (i=0; i<NY*NX; i++) { hash[i] = -1; }
    int jstar = 0;
    for (istar = 0; istar < nstar; istar++)
    {
        xx = x[istar];
        yy = y[istar];
        int idx = yy*NX+xx;
        if (hash[idx] != -1) {
            for (i=0; i<k; i++) { A[hash[idx]*k+i] += A[istar*k+i]; }
        }
        else {
            hash[idx] = jstar;
            for (i=0; i<k; i++) { A[jstar*k+i] = A[istar*k+i]; }
            x[jstar] = x[istar];
            y[jstar] = y[istar];
            jstar++;
        }
    }
    nstar = jstar;

    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nstar, n, k, alpha, A, k, B, n, beta, C, n);

    //  loop over stars, insert psfs into image    
    for (istar = 0 ; istar < nstar ; istar++)
    {
	xx = x[istar];
	yy = y[istar];
	imax = min(xx+rad,NX-1);
	jmax = min(yy+rad,NY-1);
	for (j = max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
	    for (i = max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
		image[j*NX+i] += C[i2+j2];
    }

    pcat_like_eval(NX, NY, image, ref, weight, diff2, regsize, margin, offsetx, offsety);
}

void pcat_model_subtract(int NX, int NY, int nstar, int nc, int k, float* A, float* B, float* C, int* x,
			 int* y, float* subimage, float* data, float* model )
{
    int      i,i2,imax,j,j2,jmax,rad,istar,xx,yy;
    float    alpha, beta;

    int n = nc*nc;
    rad = nc/2;

    // cblas params
    alpha = 1.0; beta = 0.0;

    // overwrite and shorten A matrix
    // save time if there are many sources per pixel
    int hash[NY*NX];
    for (i=0; i<NY*NX; i++) { hash[i] = -1; }
    int jstar = 0;
    for (istar = 0; istar < nstar; istar++)
    {
        xx = x[istar];
        yy = y[istar];
        int idx = yy*NX+xx;
        if (hash[idx] != -1) {
            for (i=0; i<k; i++) { A[hash[idx]*k+i] += A[istar*k+i]; }
        }
        else {
            hash[idx] = jstar;
            for (i=0; i<k; i++) { A[jstar*k+i] = A[istar*k+i]; }
            x[jstar] = x[istar];
            y[jstar] = y[istar];
            jstar++;
        }
    }
    nstar = jstar;

    //  matrix multiplication
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        nstar, n, k, alpha, A, k, B, n, beta, C, n);

    //  loop over stars, insert psfs into image    
    for (istar = 0 ; istar < nstar ; istar++)
    {
	xx = x[istar];
	yy = y[istar];
	imax = min(xx+rad,NX-1);
	jmax = min(yy+rad,NY-1);
	for (j = max(yy-rad,0), j2 = (istar*nc+j-yy+rad)*nc ; j <= jmax ; j++, j2+=nc)
	    for (i = max(xx-rad,0), i2 = i-xx+rad ; i <= imax ; i++, i2++)
		subimage[j*NX+i] += C[i2+j2];
    }

    // now produce the final subtracted image
    // sugimage = data - (model - submodel)
    for (i=0; i<NY*NX; i++) {
        subimage[i] = data[i] -  model[i] + subimage[i];
    }
}

void position_sample_accept(float *x, float *y, int pn, int n, int istart, int *ifail, float min_sep2) {
  int i,j;
  float r2;
  ifail[0] = -1;
  for (i=istart;i<pn;i++) {
    for (j=0;j<n;j++) {
      r2 = (x[j]-x[i])*(x[j]-x[i]) + (y[j]-y[i])*(y[j]-y[i]);
      if (i!=j && r2<min_sep2) {
        ifail[0] = i;
	return;
      }
    }
  }
}
