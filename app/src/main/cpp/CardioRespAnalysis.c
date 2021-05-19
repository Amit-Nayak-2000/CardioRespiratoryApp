#include "CardioRespAnalysis.h"
#include "CardioRespAnalysis_emxutil.h"
#include <math.h>
#include <string.h>

static const double dv[8] = { 0.022785172947974931, -0.0089123507208401943,
  -0.070158812089422817, 0.21061726710176826, 0.56832912170437477,
  0.35186953432761287, -0.020955482562526946, -0.053574450708941054 };

static const double dv1[8] = { -0.053574450708941054, 0.020955482562526946,
  0.35186953432761287, -0.56832912170437477, 0.21061726710176826,
  0.070158812089422817, -0.0089123507208401943, -0.022785172947974931 };

static void FiltFiltM(const emxArray_real_T *X, emxArray_real_T *Y);
static void b_fft(const emxArray_real_T *x, emxArray_creal_T *y);
static boolean_T b_strcmp(const char a[8]);
static void bsxfun(const emxArray_real_T *a, const emxArray_real_T *b,
                   emxArray_real_T *c);
static void c_FFTImplementationCallback_doH(const double x[8], emxArray_creal_T *
  y, int unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T
  *sintab);
static void c_FFTImplementationCallback_gen(int nRows, boolean_T useRadix2,
  emxArray_real_T *costab, emxArray_real_T *sintab, emxArray_real_T *sintabinv);
static void c_FFTImplementationCallback_get(int nfft, boolean_T useRadix2, int
  *n2blue, int *nRows);
static void c_FFTImplementationCallback_r2b(const emxArray_creal_T *x, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab,
  emxArray_creal_T *y);
static void c_fft(double varargin_1, emxArray_creal_T *y);
static void computeperiodogram(const emxArray_real_T *x, const emxArray_real_T
  *win, emxArray_real_T *Pxx, emxArray_real_T *F);
static void computepsd(const emxArray_real_T *Sxx1, const emxArray_real_T *w2,
  const char range[8], emxArray_real_T *varargout_1, emxArray_real_T
  *varargout_2);
static void d_FFTImplementationCallback_doH(const double x[8], emxArray_creal_T *
  y, int nRows, int nfft, const emxArray_creal_T *wwc, const emxArray_real_T
  *costab, const emxArray_real_T *sintab, const emxArray_real_T *costabinv,
  const emxArray_real_T *sintabinv);
static void d_FFTImplementationCallback_gen(int nRows, emxArray_real_T *costab,
  emxArray_real_T *sintab, emxArray_real_T *sintabinv);
static void d_FFTImplementationCallback_get(int nRowsM1, int nfftLen,
  emxArray_int32_T *bitrevIndex);
static void d_FFTImplementationCallback_r2b(const emxArray_real_T *x, const
  emxArray_real_T *costab, const emxArray_real_T *sintab, emxArray_creal_T *y);
static void d_fft(const emxArray_real_T *x, emxArray_creal_T *y);
static int div_s32(int numerator, int denominator);
static int div_s32_floor(int numerator, int denominator);
static void e_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int unsigned_nRows, const emxArray_real_T *costab, const
  emxArray_real_T *sintab);
static void e_FFTImplementationCallback_gen(emxArray_real_T *costab,
  emxArray_real_T *sintab, int sintabinv_size[2]);
static void e_FFTImplementationCallback_get(const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv, emxArray_real_T *hcostab, emxArray_real_T *hsintab,
  emxArray_real_T *hcostabinv, emxArray_real_T *hsintabinv);
static void f_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int nrowsx, int nRows, int nfft, const emxArray_creal_T
  *wwc, const emxArray_real_T *costab, const emxArray_real_T *sintab, const
  emxArray_real_T *costabinv, const emxArray_real_T *sintabinv);
static void f_FFTImplementationCallback_get(emxArray_creal_T *y, const
  emxArray_creal_T *reconVar1, const emxArray_creal_T *reconVar2, const
  emxArray_int32_T *wrapIndex, int hnRows);
static void fft(const double x[8], double varargin_1, emxArray_creal_T *y);
static void filter(const double b[7], const double a[7], const double x[18],
                   const double zi[6], double y[18], double zf[6]);
static void g_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab);
static void g_FFTImplementationCallback_get(emxArray_creal_T *y, const
  emxArray_creal_T *reconVar1, const emxArray_creal_T *reconVar2, const
  emxArray_int32_T *wrapIndex);
static void h_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv);
static void heartRateFilt(const emxArray_real_T *Y, emxArray_real_T *Y_Filt,
  double HeartRatePxx_data[], int HeartRatePxx_size[1]);
static void i_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab);
static void ifft(const emxArray_creal_T *x, emxArray_creal_T *y);
static void imodwtrec(const emxArray_real_T *Vin, const emxArray_real_T *Win,
                      const emxArray_creal_T *G, const emxArray_creal_T *H, int
                      J, emxArray_real_T *Vout);
static void j_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv);
static void k_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int nChan, const emxArray_real_T *costab, const
  emxArray_real_T *sintab);
static void localComputeSpectra(const emxArray_real_T *Sxx, const
  emxArray_real_T *x, const emxArray_real_T *xStart, const emxArray_real_T *xEnd,
  const emxArray_real_T *win, const char options_range[8], double k,
  emxArray_real_T *Pxx, emxArray_real_T *w);
static void modwtmra(const emxArray_real_T *w, emxArray_real_T *mra);
static void psdfreqvec(emxArray_real_T *w);
static void pwelch(const emxArray_real_T *x, emxArray_real_T *varargout_1,
                   emxArray_real_T *varargout_2);
static double rt_hypotd(double u0, double u1);
static void FiltFiltM(const emxArray_real_T *X, emxArray_real_T *Y)
{
  int i;
  double K[36];
  static const signed char iv[36] = { 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };

  static const double b_dv[6] = { -5.4717209152107378, 12.556021271069568,
    -15.468678357547901, 10.791973285377862, -4.0430635516484248,
    0.63548519162935935 };

  int j;
  signed char ipiv[6];
  int jA;
  int mmj_tmp;
  int naxpy;
  double IC[6];
  static const double b_IC[6] = { 0.0063915463899826986, -0.018171071430416595,
    0.018069045707200686, -0.0091018455609738272, 0.0047227241024250712,
    -0.0019104189768469207 };

  int jj;
  int jp1j;
  int k;
  int iy;
  double smax;
  int ix;
  double s;
  double Xi[18];
  double Xf[18];
  emxArray_real_T *Ys;
  double c_IC[6];
  static const double b[7] = { 0.0011681053345054487, -0.0,
    -0.0035043160035163464, -0.0, 0.0035043160035163464, -0.0,
    -0.0011681053345054487 };

  static const double a[7] = { 1.0, -5.4717209152107378, 12.556021271069568,
    -15.468678357547901, 10.791973285377862, -4.0430635516484248,
    0.63548519162935935 };

  double dum[18];
  double Zi[6];
  emxArray_real_T *b_Y;
  if (X->size[0] == 0) {
    Y->size[0] = 0;
    Y->size[1] = 0;
  } else {
    for (i = 0; i < 36; i++) {
      K[i] = iv[i];
    }

    for (i = 0; i < 6; i++) {
      K[i] = b_dv[i];
    }

    K[0]++;
    for (i = 0; i < 5; i++) {
      K[7 * i + 6] = -1.0;
    }

    for (i = 0; i < 6; i++) {
      ipiv[i] = (signed char)(i + 1);
    }

    for (j = 0; j < 5; j++) {
      mmj_tmp = 4 - j;
      naxpy = j * 7;
      jj = j * 7;
      jp1j = naxpy + 2;
      iy = 6 - j;
      jA = 0;
      ix = naxpy;
      smax = fabs(K[jj]);
      for (k = 2; k <= iy; k++) {
        ix++;
        s = fabs(K[ix]);
        if (s > smax) {
          jA = k - 1;
          smax = s;
        }
      }

      if (K[jj + jA] != 0.0) {
        if (jA != 0) {
          iy = j + jA;
          ipiv[j] = (signed char)(iy + 1);
          ix = j;
          for (k = 0; k < 6; k++) {
            smax = K[ix];
            K[ix] = K[iy];
            K[iy] = smax;
            ix += 6;
            iy += 6;
          }
        }

        i = (jj - j) + 6;
        for (jA = jp1j; jA <= i; jA++) {
          K[jA - 1] /= K[jj];
        }
      }

      iy = naxpy + 6;
      jA = jj;
      for (jp1j = 0; jp1j <= mmj_tmp; jp1j++) {
        smax = K[iy];
        if (K[iy] != 0.0) {
          ix = jj + 1;
          i = jA + 8;
          naxpy = (jA - j) + 12;
          for (k = i; k <= naxpy; k++) {
            K[k - 1] += K[ix] * -smax;
            ix++;
          }
        }

        iy += 6;
        jA += 6;
      }
    }

    for (jA = 0; jA < 6; jA++) {
      IC[jA] = b_IC[jA];
    }

    for (jA = 0; jA < 5; jA++) {
      if (ipiv[jA] != jA + 1) {
        smax = IC[jA];
        iy = ipiv[jA] - 1;
        IC[jA] = IC[iy];
        IC[iy] = smax;
      }
    }

    for (k = 0; k < 6; k++) {
      iy = 6 * k;
      if (IC[k] != 0.0) {
        i = k + 2;
        for (jA = i; jA < 7; jA++) {
          IC[jA - 1] -= IC[k] * K[(jA + iy) - 1];
        }
      }
    }

    for (k = 5; k >= 0; k--) {
      iy = 6 * k;
      if (IC[k] != 0.0) {
        IC[k] /= K[k + iy];
        for (jA = 0; jA < k; jA++) {
          IC[jA] -= IC[k] * K[jA + iy];
        }
      }
    }

    smax = 2.0 * X->data[0];
    s = 2.0 * X->data[X->size[0] - 1];
    for (i = 0; i < 18; i++) {
      Xi[i] = smax - X->data[18 - i];
      Xf[i] = s - X->data[(X->size[0] - i) - 2];
    }

    for (i = 0; i < 6; i++) {
      c_IC[i] = IC[i] * Xi[0];
    }

    emxInit_real_T(&Ys, 1);
    filter(b, a, Xi, c_IC, dum, Zi);
    i = Ys->size[0];
    Ys->size[0] = X->size[0];
    emxEnsureCapacity_real_T(Ys, i);
    jp1j = X->size[0];
    for (jA = 0; jA < 6; jA++) {
      c_IC[jA] = 0.0;
    }

    if (X->size[0] < 6) {
      iy = X->size[0] - 1;
    } else {
      iy = 5;
    }

    for (k = 0; k <= iy; k++) {
      Ys->data[k] = Zi[k];
    }

    i = iy + 2;
    for (k = i; k <= jp1j; k++) {
      Ys->data[k - 1] = 0.0;
    }

    for (k = 0; k < jp1j; k++) {
      jA = jp1j - k;
      if (jA < 7) {
        naxpy = jA;
      } else {
        naxpy = 7;
      }

      for (j = 0; j < naxpy; j++) {
        iy = k + j;
        Ys->data[iy] += X->data[k] * b[j];
      }

      jA -= 2;
      if (jA + 1 < 6) {
        naxpy = jA;
      } else {
        naxpy = 5;
      }

      smax = -Ys->data[k];
      for (j = 0; j <= naxpy; j++) {
        iy = (k + j) + 1;
        Ys->data[iy] += smax * a[j + 1];
      }
    }

    if (X->size[0] < 6) {
      iy = 5 - X->size[0];
      for (k = 0; k <= iy; k++) {
        c_IC[k] = Zi[k + jp1j];
      }
    }

    if (X->size[0] >= 7) {
      jA = X->size[0] - 6;
    } else {
      jA = 0;
    }

    i = X->size[0] - 1;
    for (k = jA; k <= i; k++) {
      iy = jp1j - k;
      naxpy = 6 - iy;
      for (j = 0; j <= naxpy; j++) {
        c_IC[j] += X->data[k] * b[iy + j];
      }
    }

    if (X->size[0] >= 7) {
      jA = X->size[0] - 6;
    } else {
      jA = 0;
    }

    i = X->size[0] - 1;
    for (k = jA; k <= i; k++) {
      iy = jp1j - k;
      naxpy = 6 - iy;
      for (j = 0; j <= naxpy; j++) {
        c_IC[j] += -Ys->data[k] * a[iy + j];
      }
    }

    for (k = 0; k < 6; k++) {
      Xi[k] = c_IC[k];
    }

    memset(&Xi[6], 0, 12U * sizeof(double));
    for (k = 0; k < 18; k++) {
      if (18 - k < 7) {
        naxpy = 17 - k;
      } else {
        naxpy = 6;
      }

      for (j = 0; j <= naxpy; j++) {
        iy = k + j;
        Xi[iy] += Xf[k] * b[j];
      }

      if (17 - k < 6) {
        naxpy = 16 - k;
      } else {
        naxpy = 5;
      }

      smax = -Xi[k];
      for (j = 0; j <= naxpy; j++) {
        iy = (k + j) + 1;
        Xi[iy] += smax * a[j + 1];
      }
    }

    for (i = 0; i < 18; i++) {
      Xf[i] = Xi[17 - i];
    }

    memcpy(&Xi[0], &Xf[0], 18U * sizeof(double));
    for (i = 0; i < 6; i++) {
      c_IC[i] = IC[i] * Xi[0];
    }

    emxInit_real_T(&b_Y, 1);
    filter(b, a, Xi, c_IC, dum, IC);
    i = b_Y->size[0];
    b_Y->size[0] = div_s32_floor(1 - X->size[0], -1) + 1;
    emxEnsureCapacity_real_T(b_Y, i);
    jp1j = div_s32_floor(1 - X->size[0], -1);
    for (k = 0; k < 6; k++) {
      b_Y->data[k] = IC[k];
    }

    for (k = 7; k <= jp1j + 1; k++) {
      b_Y->data[k - 1] = 0.0;
    }

    for (k = 0; k <= jp1j; k++) {
      jA = jp1j - k;
      if (jA + 1 < 7) {
        naxpy = jA;
      } else {
        naxpy = 6;
      }

      for (j = 0; j <= naxpy; j++) {
        iy = k + j;
        b_Y->data[iy] += Ys->data[(X->size[0] - k) - 1] * b[j];
      }

      if (jA < 6) {
        naxpy = jA;
      } else {
        naxpy = 6;
      }

      smax = -b_Y->data[k];
      for (j = 0; j < naxpy; j++) {
        iy = (k + j) + 1;
        b_Y->data[iy] += smax * a[j + 1];
      }
    }

    emxFree_real_T(&Ys);
    i = Y->size[0] * Y->size[1];
    Y->size[0] = div_s32_floor(1 - X->size[0], -1) + 1;
    Y->size[1] = 1;
    emxEnsureCapacity_real_T(Y, i);
    iy = div_s32_floor(1 - X->size[0], -1) + 1;
    for (i = 0; i < iy; i++) {
      Y->data[i] = b_Y->data[(X->size[0] - i) - 1];
    }

    if (X->size[0] == 1) {
      i = Y->size[0] * Y->size[1];
      Y->size[0] = 1;
      Y->size[1] = 1;
      emxEnsureCapacity_real_T(Y, i);
      Y->data[0] = b_Y->data[X->size[0] - 1];
    }

    emxFree_real_T(&b_Y);
  }
}

static void b_fft(const emxArray_real_T *x, emxArray_creal_T *y)
{
  emxArray_real_T *costab;
  emxArray_real_T *sintab;
  emxArray_real_T *sintabinv;
  int len;
  boolean_T useRadix2;
  int N2blue;
  int minNrowsNx;
  emxArray_creal_T *yCol;
  emxArray_creal_T *wwc;
  int i;
  int nInt2m1;
  emxArray_real_T b_x;
  int c_x[1];
  int idx;
  int rt;
  int nInt2;
  int k;
  int b_y;
  double nt_im;
  double nt_re;
  int d_x[1];
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  emxInit_real_T(&costab, 2);
  emxInit_real_T(&sintab, 2);
  emxInit_real_T(&sintabinv, 2);
  len = x->size[1];
  useRadix2 = ((x->size[1] & (x->size[1] - 1)) == 0);
  c_FFTImplementationCallback_get(x->size[1], useRadix2, &N2blue, &minNrowsNx);
  c_FFTImplementationCallback_gen(minNrowsNx, useRadix2, costab, sintab,
    sintabinv);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = x->size[1];
    emxEnsureCapacity_creal_T(yCol, i);
    minNrowsNx = x->size[1];
    b_x = *x;
    c_x[0] = minNrowsNx;
    b_x.size = &c_x[0];
    b_x.numDimensions = 1;
    e_FFTImplementationCallback_doH(&b_x, yCol, x->size[1], costab, sintab);
  } else {
    emxInit_creal_T(&wwc, 1);
    if ((x->size[1] & 1) == 0) {
      minNrowsNx = x->size[1] / 2;
      nInt2m1 = (minNrowsNx + minNrowsNx) - 1;
      i = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, i);
      idx = minNrowsNx;
      rt = 0;
      wwc->data[minNrowsNx - 1].re = 1.0;
      wwc->data[minNrowsNx - 1].im = 0.0;
      nInt2 = minNrowsNx << 1;
      for (k = 0; k <= minNrowsNx - 2; k++) {
        b_y = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= b_y) {
          rt += b_y - nInt2;
        } else {
          rt += b_y;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)minNrowsNx;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      i = nInt2m1 - 1;
      for (k = i; k >= minNrowsNx; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    } else {
      nInt2m1 = (x->size[1] + x->size[1]) - 1;
      i = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, i);
      idx = x->size[1];
      rt = 0;
      wwc->data[x->size[1] - 1].re = 1.0;
      wwc->data[x->size[1] - 1].im = 0.0;
      nInt2 = x->size[1] << 1;
      i = x->size[1];
      for (k = 0; k <= i - 2; k++) {
        b_y = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= b_y) {
          rt += b_y - nInt2;
        } else {
          rt += b_y;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)len;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      i = nInt2m1 - 1;
      for (k = i; k >= len; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    }

    i = yCol->size[0];
    yCol->size[0] = x->size[1];
    emxEnsureCapacity_creal_T(yCol, i);
    if ((N2blue != 1) && ((x->size[1] & 1) == 0)) {
      minNrowsNx = x->size[1];
      b_x = *x;
      d_x[0] = minNrowsNx;
      b_x.size = &d_x[0];
      b_x.numDimensions = 1;
      f_FFTImplementationCallback_doH(&b_x, yCol, x->size[1], x->size[1], N2blue,
        wwc, costab, sintab, costab, sintabinv);
    } else {
      minNrowsNx = x->size[1];
      nInt2m1 = 0;
      for (k = 0; k < minNrowsNx; k++) {
        i = (len + k) - 1;
        yCol->data[k].re = wwc->data[i].re * x->data[nInt2m1];
        yCol->data[k].im = wwc->data[i].im * -x->data[nInt2m1];
        nInt2m1++;
      }

      i = x->size[1] + 1;
      for (k = i; k <= len; k++) {
        yCol->data[k - 1].re = 0.0;
        yCol->data[k - 1].im = 0.0;
      }

      emxInit_creal_T(&fv, 1);
      emxInit_creal_T(&b_fv, 1);
      c_FFTImplementationCallback_r2b(yCol, N2blue, costab, sintab, fv);
      c_FFTImplementationCallback_r2b(wwc, N2blue, costab, sintab, b_fv);
      i = b_fv->size[0];
      b_fv->size[0] = fv->size[0];
      emxEnsureCapacity_creal_T(b_fv, i);
      minNrowsNx = fv->size[0];
      for (i = 0; i < minNrowsNx; i++) {
        nt_re = fv->data[i].re * b_fv->data[i].im + fv->data[i].im * b_fv->
          data[i].re;
        b_fv->data[i].re = fv->data[i].re * b_fv->data[i].re - fv->data[i].im *
          b_fv->data[i].im;
        b_fv->data[i].im = nt_re;
      }

      c_FFTImplementationCallback_r2b(b_fv, N2blue, costab, sintabinv, fv);
      emxFree_creal_T(&b_fv);
      if (fv->size[0] > 1) {
        nt_re = 1.0 / (double)fv->size[0];
        minNrowsNx = fv->size[0];
        for (i = 0; i < minNrowsNx; i++) {
          fv->data[i].re *= nt_re;
          fv->data[i].im *= nt_re;
        }
      }

      idx = 0;
      i = x->size[1];
      minNrowsNx = wwc->size[0];
      for (k = i; k <= minNrowsNx; k++) {
        yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re +
          wwc->data[k - 1].im * fv->data[k - 1].im;
        yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im -
          wwc->data[k - 1].im * fv->data[k - 1].re;
        idx++;
      }

      emxFree_creal_T(&fv);
    }

    emxFree_creal_T(&wwc);
  }

  emxFree_real_T(&sintabinv);
  emxFree_real_T(&sintab);
  emxFree_real_T(&costab);
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = x->size[1];
  emxEnsureCapacity_creal_T(y, i);
  minNrowsNx = x->size[1];
  for (i = 0; i < minNrowsNx; i++) {
    y->data[i] = yCol->data[i];
  }

  emxFree_creal_T(&yCol);
}

static boolean_T b_strcmp(const char a[8])
{
  int ret;
  static const char b[8] = { 'o', 'n', 'e', 's', 'i', 'd', 'e', 'd' };

  ret = memcmp(&a[0], &b[0], 8);
  return ret == 0;
}

static void bsxfun(const emxArray_real_T *a, const emxArray_real_T *b,
                   emxArray_real_T *c)
{
  int i;
  int acoef;
  int bcoef;
  int k;
  i = c->size[0] * c->size[1];
  acoef = b->size[0];
  bcoef = a->size[0];
  if (acoef < bcoef) {
    bcoef = acoef;
  }

  if (b->size[0] == 1) {
    c->size[0] = a->size[0];
  } else if (a->size[0] == 1) {
    c->size[0] = b->size[0];
  } else if (a->size[0] == b->size[0]) {
    c->size[0] = a->size[0];
  } else {
    c->size[0] = bcoef;
  }

  c->size[1] = a->size[1];
  emxEnsureCapacity_real_T(c, i);
  acoef = b->size[0];
  bcoef = a->size[0];
  if (acoef < bcoef) {
    bcoef = acoef;
  }

  if (b->size[0] == 1) {
    bcoef = a->size[0];
  } else if (a->size[0] == 1) {
    bcoef = b->size[0];
  } else {
    if (a->size[0] == b->size[0]) {
      bcoef = a->size[0];
    }
  }

  if ((bcoef != 0) && (a->size[1] != 0)) {
    acoef = (a->size[0] != 1);
    bcoef = (b->size[0] != 1);
    i = c->size[0] - 1;
    for (k = 0; k <= i; k++) {
      c->data[k] = a->data[acoef * k] * b->data[bcoef * k];
    }
  }
}

static void c_FFTImplementationCallback_doH(const double x[8], emxArray_creal_T *
  y, int unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T
  *sintab)
{
  emxArray_real_T *hcostab;
  emxArray_real_T *hsintab;
  int nRows;
  int istart;
  int nRowsD2;
  int k;
  int hszCostab;
  int iDelta;
  int i;
  emxArray_int32_T *wrapIndex;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  emxArray_int32_T *bitrevIndex;
  double z;
  double temp_re;
  double temp_im;
  int temp_re_tmp;
  int j;
  double twid_re;
  double twid_im;
  int ihi;
  emxInit_real_T(&hcostab, 2);
  emxInit_real_T(&hsintab, 2);
  nRows = unsigned_nRows / 2;
  istart = nRows - 2;
  nRowsD2 = nRows / 2;
  k = nRowsD2 / 2;
  hszCostab = costab->size[1] / 2;
  iDelta = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostab, iDelta);
  iDelta = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintab, iDelta);
  for (i = 0; i < hszCostab; i++) {
    iDelta = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[iDelta];
    hsintab->data[i] = sintab->data[iDelta];
  }

  emxInit_int32_T(&wrapIndex, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  iDelta = reconVar1->size[0];
  reconVar1->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar1, iDelta);
  iDelta = reconVar2->size[0];
  reconVar2->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar2, iDelta);
  iDelta = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = nRows;
  emxEnsureCapacity_int32_T(wrapIndex, iDelta);
  for (i = 0; i < nRows; i++) {
    z = sintab->data[i];
    temp_re = costab->data[i];
    reconVar1->data[i].re = z + 1.0;
    reconVar1->data[i].im = -temp_re;
    reconVar2->data[i].re = 1.0 - z;
    reconVar2->data[i].im = temp_re;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (nRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxInit_int32_T(&bitrevIndex, 1);
  z = (double)unsigned_nRows / 2.0;
  iDelta = y->size[0];
  if (iDelta >= nRows) {
    iDelta = nRows;
  }

  d_FFTImplementationCallback_get(iDelta - 1, (int)z, bitrevIndex);
  hszCostab = 0;
  if (8 < unsigned_nRows) {
    iDelta = 8;
  } else {
    iDelta = unsigned_nRows;
  }

  iDelta = (int)((double)iDelta / 2.0);
  for (i = 0; i < iDelta; i++) {
    y->data[bitrevIndex->data[i] - 1].re = x[hszCostab];
    y->data[bitrevIndex->data[i] - 1].im = x[hszCostab + 1];
    hszCostab += 2;
  }

  emxFree_int32_T(&bitrevIndex);
  if (nRows > 1) {
    for (i = 0; i <= istart; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }
  }

  iDelta = 2;
  hszCostab = 4;
  nRows = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < nRows; i += hszCostab) {
      temp_re_tmp = i + iDelta;
      temp_re = y->data[temp_re_tmp].re;
      temp_im = y->data[temp_re_tmp].im;
      y->data[temp_re_tmp].re = y->data[i].re - temp_re;
      y->data[temp_re_tmp].im = y->data[i].im - temp_im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      twid_re = hcostab->data[j];
      twid_im = hsintab->data[j];
      i = istart;
      ihi = istart + nRows;
      while (i < ihi) {
        temp_re_tmp = i + iDelta;
        temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
          data[temp_re_tmp].im;
        temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
          data[temp_re_tmp].re;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
        i += hszCostab;
      }

      istart++;
    }

    k /= 2;
    iDelta = hszCostab;
    hszCostab += hszCostab;
    nRows -= iDelta;
  }

  emxFree_real_T(&hsintab);
  emxFree_real_T(&hcostab);
  f_FFTImplementationCallback_get(y, reconVar1, reconVar2, wrapIndex, (int)z);
  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
}

static void c_FFTImplementationCallback_gen(int nRows, boolean_T useRadix2,
  emxArray_real_T *costab, emxArray_real_T *sintab, emxArray_real_T *sintabinv)
{
  emxArray_real_T *costab1q;
  double e;
  int n;
  int i;
  int nd2;
  int k;
  emxInit_real_T(&costab1q, 2);
  e = 6.2831853071795862 / (double)nRows;
  n = nRows / 2 / 2;
  i = costab1q->size[0] * costab1q->size[1];
  costab1q->size[0] = 1;
  costab1q->size[1] = n + 1;
  emxEnsureCapacity_real_T(costab1q, i);
  costab1q->data[0] = 1.0;
  nd2 = n / 2 - 1;
  for (k = 0; k <= nd2; k++) {
    costab1q->data[k + 1] = cos(e * ((double)k + 1.0));
  }

  i = nd2 + 2;
  nd2 = n - 1;
  for (k = i; k <= nd2; k++) {
    costab1q->data[k] = sin(e * (double)(n - k));
  }

  costab1q->data[n] = 0.0;
  if (!useRadix2) {
    n = costab1q->size[1] - 1;
    nd2 = (costab1q->size[1] - 1) << 1;
    i = costab->size[0] * costab->size[1];
    costab->size[0] = 1;
    costab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(costab, i);
    i = sintab->size[0] * sintab->size[1];
    sintab->size[0] = 1;
    sintab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintab, i);
    costab->data[0] = 1.0;
    sintab->data[0] = 0.0;
    i = sintabinv->size[0] * sintabinv->size[1];
    sintabinv->size[0] = 1;
    sintabinv->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintabinv, i);
    for (k = 0; k < n; k++) {
      sintabinv->data[k + 1] = costab1q->data[(n - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      sintabinv->data[k] = costab1q->data[k - n];
    }

    for (k = 0; k < n; k++) {
      costab->data[k + 1] = costab1q->data[k + 1];
      sintab->data[k + 1] = -costab1q->data[(n - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      costab->data[k] = -costab1q->data[nd2 - k];
      sintab->data[k] = -costab1q->data[k - n];
    }
  } else {
    n = costab1q->size[1] - 1;
    nd2 = (costab1q->size[1] - 1) << 1;
    i = costab->size[0] * costab->size[1];
    costab->size[0] = 1;
    costab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(costab, i);
    i = sintab->size[0] * sintab->size[1];
    sintab->size[0] = 1;
    sintab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintab, i);
    costab->data[0] = 1.0;
    sintab->data[0] = 0.0;
    for (k = 0; k < n; k++) {
      costab->data[k + 1] = costab1q->data[k + 1];
      sintab->data[k + 1] = -costab1q->data[(n - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      costab->data[k] = -costab1q->data[nd2 - k];
      sintab->data[k] = -costab1q->data[k - n];
    }

    sintabinv->size[0] = 1;
    sintabinv->size[1] = 0;
  }

  emxFree_real_T(&costab1q);
}

static void c_FFTImplementationCallback_get(int nfft, boolean_T useRadix2, int
  *n2blue, int *nRows)
{
  int n;
  int pmax;
  int pmin;
  boolean_T exitg1;
  int k;
  int pow2p;
  *n2blue = 1;
  if (useRadix2) {
    *nRows = nfft;
  } else {
    n = (nfft + nfft) - 1;
    pmax = 31;
    pmin = 0;
    exitg1 = false;
    while ((!exitg1) && (pmax - pmin > 1)) {
      k = (pmin + pmax) >> 1;
      pow2p = 1 << k;
      if (pow2p == n) {
        pmax = k;
        exitg1 = true;
      } else if (pow2p > n) {
        pmax = k;
      } else {
        pmin = k;
      }
    }

    *n2blue = 1 << pmax;
    *nRows = *n2blue;
  }
}

static void c_FFTImplementationCallback_r2b(const emxArray_creal_T *x, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab,
  emxArray_creal_T *y)
{
  int iy;
  int iDelta2;
  int iheight;
  int nRowsD2;
  int k;
  int ix;
  int ju;
  int i;
  boolean_T tst;
  double temp_re;
  double temp_im;
  double twid_re;
  double twid_im;
  int temp_re_tmp;
  int ihi;
  iy = y->size[0];
  y->size[0] = unsigned_nRows;
  emxEnsureCapacity_creal_T(y, iy);
  if (unsigned_nRows > x->size[0]) {
    iy = y->size[0];
    y->size[0] = unsigned_nRows;
    emxEnsureCapacity_creal_T(y, iy);
    for (iy = 0; iy < unsigned_nRows; iy++) {
      y->data[iy].re = 0.0;
      y->data[iy].im = 0.0;
    }
  }

  iDelta2 = x->size[0];
  if (iDelta2 >= unsigned_nRows) {
    iDelta2 = unsigned_nRows;
  }

  iheight = unsigned_nRows - 2;
  nRowsD2 = unsigned_nRows / 2;
  k = nRowsD2 / 2;
  ix = 0;
  iy = 0;
  ju = 0;
  for (i = 0; i <= iDelta2 - 2; i++) {
    y->data[iy] = x->data[ix];
    iy = unsigned_nRows;
    tst = true;
    while (tst) {
      iy >>= 1;
      ju ^= iy;
      tst = ((ju & iy) == 0);
    }

    iy = ju;
    ix++;
  }

  y->data[iy] = x->data[ix];
  if (unsigned_nRows > 1) {
    for (i = 0; i <= iheight; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      twid_re = y->data[i].re;
      twid_im = y->data[i].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      twid_re += temp_re;
      twid_im += temp_im;
      y->data[i].re = twid_re;
      y->data[i].im = twid_im;
    }
  }

  iy = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += iDelta2) {
      temp_re_tmp = i + iy;
      temp_re = y->data[temp_re_tmp].re;
      temp_im = y->data[temp_re_tmp].im;
      y->data[temp_re_tmp].re = y->data[i].re - y->data[temp_re_tmp].re;
      y->data[temp_re_tmp].im = y->data[i].im - y->data[temp_re_tmp].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    ix = 1;
    for (ju = k; ju < nRowsD2; ju += k) {
      twid_re = costab->data[ju];
      twid_im = sintab->data[ju];
      i = ix;
      ihi = ix + iheight;
      while (i < ihi) {
        temp_re_tmp = i + iy;
        temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
          data[temp_re_tmp].im;
        temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
          data[temp_re_tmp].re;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
        i += iDelta2;
      }

      ix++;
    }

    k /= 2;
    iy = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iy;
  }
}

static void c_fft(double varargin_1, emxArray_creal_T *y)
{
  emxArray_real_T *costab;
  emxArray_real_T *sintab;
  emxArray_real_T *sintabinv;
  int len_tmp;
  int nInt2;
  boolean_T useRadix2;
  int N2blue;
  int nRows;
  emxArray_creal_T *yCol;
  int i;
  emxArray_creal_T *wwc;
  int nInt2m1;
  int xidx;
  int idx;
  int rt;
  int k;
  double nt_im;
  double nt_re;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  emxInit_real_T(&costab, 2);
  emxInit_real_T(&sintab, 2);
  emxInit_real_T(&sintabinv, 2);
  len_tmp = (int)varargin_1;
  nInt2 = len_tmp - 1;
  useRadix2 = ((len_tmp & nInt2) == 0);
  c_FFTImplementationCallback_get((int)varargin_1, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, costab, sintab, sintabinv);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, i);
    if (len_tmp > 8) {
      i = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, i);
      for (i = 0; i < len_tmp; i++) {
        yCol->data[i].re = 0.0;
        yCol->data[i].im = 0.0;
      }
    }

    i_FFTImplementationCallback_doH(yCol, (int)varargin_1, costab, sintab);
  } else {
    i = len_tmp & 1;
    emxInit_creal_T(&wwc, 1);
    if (i == 0) {
      nRows = len_tmp / 2;
      nInt2m1 = (nRows + nRows) - 1;
      xidx = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, xidx);
      idx = nRows;
      rt = 0;
      wwc->data[nRows - 1].re = 1.0;
      wwc->data[nRows - 1].im = 0.0;
      nInt2 = nRows << 1;
      for (k = 0; k <= nRows - 2; k++) {
        xidx = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= xidx) {
          rt += xidx - nInt2;
        } else {
          rt += xidx;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)nRows;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      xidx = nInt2m1 - 1;
      for (k = xidx; k >= nRows; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    } else {
      nInt2m1 = (len_tmp + len_tmp) - 1;
      xidx = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, xidx);
      idx = len_tmp;
      rt = 0;
      wwc->data[nInt2].re = 1.0;
      wwc->data[nInt2].im = 0.0;
      nInt2 = len_tmp << 1;
      for (k = 0; k <= len_tmp - 2; k++) {
        xidx = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= xidx) {
          rt += xidx - nInt2;
        } else {
          rt += xidx;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)len_tmp;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      xidx = nInt2m1 - 1;
      for (k = xidx; k >= len_tmp; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    }

    xidx = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, xidx);
    if (len_tmp > 8) {
      xidx = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, xidx);
      for (xidx = 0; xidx < len_tmp; xidx++) {
        yCol->data[xidx].re = 0.0;
        yCol->data[xidx].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      j_FFTImplementationCallback_doH(yCol, (int)varargin_1, N2blue, wwc, costab,
        sintab, costab, sintabinv);
    } else {
      if ((int)varargin_1 < 8) {
        nInt2 = (int)varargin_1 - 1;
      } else {
        nInt2 = 7;
      }

      xidx = 0;
      for (k = 0; k <= nInt2; k++) {
        rt = (len_tmp + k) - 1;
        yCol->data[k].re = wwc->data[rt].re * dv1[xidx];
        yCol->data[k].im = wwc->data[rt].im * -dv1[xidx];
        xidx++;
      }

      i = nInt2 + 2;
      for (k = i; k <= len_tmp; k++) {
        yCol->data[k - 1].re = 0.0;
        yCol->data[k - 1].im = 0.0;
      }

      emxInit_creal_T(&fv, 1);
      emxInit_creal_T(&b_fv, 1);
      c_FFTImplementationCallback_r2b(yCol, N2blue, costab, sintab, fv);
      c_FFTImplementationCallback_r2b(wwc, N2blue, costab, sintab, b_fv);
      i = b_fv->size[0];
      b_fv->size[0] = fv->size[0];
      emxEnsureCapacity_creal_T(b_fv, i);
      nInt2 = fv->size[0];
      for (i = 0; i < nInt2; i++) {
        nt_re = fv->data[i].re * b_fv->data[i].im + fv->data[i].im * b_fv->
          data[i].re;
        b_fv->data[i].re = fv->data[i].re * b_fv->data[i].re - fv->data[i].im *
          b_fv->data[i].im;
        b_fv->data[i].im = nt_re;
      }

      c_FFTImplementationCallback_r2b(b_fv, N2blue, costab, sintabinv, fv);
      emxFree_creal_T(&b_fv);
      if (fv->size[0] > 1) {
        nt_re = 1.0 / (double)fv->size[0];
        nInt2 = fv->size[0];
        for (i = 0; i < nInt2; i++) {
          fv->data[i].re *= nt_re;
          fv->data[i].im *= nt_re;
        }
      }

      idx = 0;
      i = wwc->size[0];
      for (k = len_tmp; k <= i; k++) {
        yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re +
          wwc->data[k - 1].im * fv->data[k - 1].im;
        yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im -
          wwc->data[k - 1].im * fv->data[k - 1].re;
        idx++;
      }

      emxFree_creal_T(&fv);
    }

    emxFree_creal_T(&wwc);
  }

  emxFree_real_T(&sintabinv);
  emxFree_real_T(&sintab);
  emxFree_real_T(&costab);
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = len_tmp;
  emxEnsureCapacity_creal_T(y, i);
  for (i = 0; i < len_tmp; i++) {
    y->data[i] = yCol->data[i];
  }

  emxFree_creal_T(&yCol);
}

static void computeperiodogram(const emxArray_real_T *x, const emxArray_real_T
  *win, emxArray_real_T *Pxx, emxArray_real_T *F)
{
  emxArray_real_T *xw;
  int i;
  int offset;
  emxArray_real_T *xin;
  int nFullPasses;
  emxArray_real_T *wrappedData;
  int b_remainder;
  int j;
  int i1;
  emxArray_creal_T *Xx;
  double b_win;
  int k;
  int b_j;
  double Xx_re;
  double Xx_im;
  emxInit_real_T(&xw, 2);
  if ((x->size[0] == 1) || (x->size[1] == 1)) {
    i = xw->size[0] * xw->size[1];
    xw->size[0] = x->size[0] * x->size[1];
    xw->size[1] = 1;
    emxEnsureCapacity_real_T(xw, i);
    offset = x->size[0] * x->size[1];
    for (i = 0; i < offset; i++) {
      xw->data[i] = x->data[i];
    }
  } else {
    i = xw->size[0] * xw->size[1];
    xw->size[0] = x->size[0];
    xw->size[1] = x->size[1];
    emxEnsureCapacity_real_T(xw, i);
    offset = x->size[0] * x->size[1];
    for (i = 0; i < offset; i++) {
      xw->data[i] = x->data[i];
    }
  }

  emxInit_real_T(&xin, 2);
  bsxfun(xw, win, xin);
  i = xw->size[0] * xw->size[1];
  xw->size[0] = 65536;
  xw->size[1] = xin->size[1];
  emxEnsureCapacity_real_T(xw, i);
  offset = xin->size[1] << 16;
  for (i = 0; i < offset; i++) {
    xw->data[i] = 0.0;
  }

  if (xin->size[0] > 65536) {
    i = xin->size[1];
    if (0 <= xin->size[1] - 1) {
      nFullPasses = xin->size[0] / 65536;
      b_remainder = (xin->size[0] - (nFullPasses << 16)) - 1;
      i1 = b_remainder + 2;
    }

    emxInit_real_T(&wrappedData, 2);
    for (j = 0; j < i; j++) {
      offset = wrappedData->size[0] * wrappedData->size[1];
      wrappedData->size[0] = 65536;
      wrappedData->size[1] = 1;
      emxEnsureCapacity_real_T(wrappedData, offset);
      for (offset = 0; offset < 65536; offset++) {
        wrappedData->data[offset] = 0.0;
      }

      offset = nFullPasses << 16;
      for (k = 0; k <= b_remainder; k++) {
        wrappedData->data[k] = xin->data[offset + k];
      }

      for (k = i1; k < 65537; k++) {
        wrappedData->data[k - 1] = 0.0;
      }

      for (b_j = 0; b_j < nFullPasses; b_j++) {
        offset = b_j << 16;
        for (k = 0; k < 65536; k++) {
          wrappedData->data[k] += xin->data[offset + k];
        }
      }

      for (offset = 0; offset < 65536; offset++) {
        xw->data[offset] = wrappedData->data[offset];
      }
    }

    emxFree_real_T(&wrappedData);
  } else {
    i = xw->size[0] * xw->size[1];
    xw->size[0] = xin->size[0];
    xw->size[1] = xin->size[1];
    emxEnsureCapacity_real_T(xw, i);
    offset = xin->size[0] * xin->size[1];
    for (i = 0; i < offset; i++) {
      xw->data[i] = xin->data[i];
    }
  }

  emxFree_real_T(&xin);
  emxInit_creal_T(&Xx, 2);
  d_fft(xw, Xx);
  psdfreqvec(F);
  b_win = 0.0;
  offset = win->size[0];
  emxFree_real_T(&xw);
  for (i = 0; i < offset; i++) {
    b_win += win->data[i] * win->data[i];
  }

  i = Pxx->size[0] * Pxx->size[1];
  Pxx->size[0] = 65536;
  Pxx->size[1] = Xx->size[1];
  emxEnsureCapacity_real_T(Pxx, i);
  offset = Xx->size[0] * Xx->size[1];
  for (i = 0; i < offset; i++) {
    Xx_re = Xx->data[i].re * Xx->data[i].re - Xx->data[i].im * -Xx->data[i].im;
    Xx_im = Xx->data[i].re * -Xx->data[i].im + Xx->data[i].im * Xx->data[i].re;
    if (Xx_im == 0.0) {
      Xx_re /= b_win;
    } else if (Xx_re == 0.0) {
      Xx_re = 0.0;
    } else {
      Xx_re /= b_win;
    }

    Pxx->data[i] = Xx_re;
  }

  emxFree_creal_T(&Xx);
}

static void computepsd(const emxArray_real_T *Sxx1, const emxArray_real_T *w2,
  const char range[8], emxArray_real_T *varargout_1, emxArray_real_T
  *varargout_2)
{
  emxArray_real_T *Sxx;
  int i;
  emxArray_real_T *Sxx_unscaled;
  int loop_ub;
  emxArray_real_T *y;
  int i1;
  int result;
  boolean_T empty_non_axis_sizes;
  signed char input_sizes_idx_0;
  unsigned short b_input_sizes_idx_0;
  signed char sizes_idx_0;
  int b_loop_ub;
  int c_input_sizes_idx_0;
  double Sxx_unscaled_data[1];
  int d_input_sizes_idx_0;
  double b_Sxx_unscaled_data[1];
  emxInit_real_T(&Sxx, 2);
  if (b_strcmp(range)) {
    emxInit_real_T(&Sxx_unscaled, 2);
    loop_ub = Sxx1->size[1];
    i = Sxx_unscaled->size[0] * Sxx_unscaled->size[1];
    Sxx_unscaled->size[0] = 32769;
    Sxx_unscaled->size[1] = Sxx1->size[1];
    emxEnsureCapacity_real_T(Sxx_unscaled, i);
    for (i = 0; i < loop_ub; i++) {
      for (i1 = 0; i1 < 32769; i1++) {
        Sxx_unscaled->data[i1 + Sxx_unscaled->size[0] * i] = Sxx1->data[i1 +
          Sxx1->size[0] * i];
      }
    }

    emxInit_real_T(&y, 2);
    loop_ub = Sxx1->size[1] - 1;
    i = y->size[0] * y->size[1];
    y->size[0] = 32767;
    y->size[1] = Sxx1->size[1];
    emxEnsureCapacity_real_T(y, i);
    for (i = 0; i <= loop_ub; i++) {
      for (i1 = 0; i1 < 32767; i1++) {
        y->data[i1 + 32767 * i] = 2.0 * Sxx_unscaled->data[(i1 +
          Sxx_unscaled->size[0] * i) + 1];
      }
    }

    if (Sxx1->size[1] != 0) {
      result = Sxx1->size[1];
    } else if (y->size[1] != 0) {
      result = 1;
    } else if (Sxx1->size[1] != 0) {
      result = Sxx1->size[1];
    } else {
      if (Sxx1->size[1] > 0) {
        result = Sxx1->size[1];
      } else {
        result = 0;
      }

      if (y->size[1] > result) {
        result = 1;
      }

      if (Sxx1->size[1] > result) {
        result = Sxx1->size[1];
      }
    }

    empty_non_axis_sizes = (result == 0);
    if (empty_non_axis_sizes || (Sxx1->size[1] != 0)) {
      input_sizes_idx_0 = 1;
    } else {
      input_sizes_idx_0 = 0;
    }

    if (empty_non_axis_sizes || (y->size[1] != 0)) {
      b_input_sizes_idx_0 = 32767U;
    } else {
      b_input_sizes_idx_0 = 0U;
    }

    if (empty_non_axis_sizes || (Sxx1->size[1] != 0)) {
      sizes_idx_0 = 1;
    } else {
      sizes_idx_0 = 0;
    }

    loop_ub = Sxx1->size[1] - 1;
    b_loop_ub = Sxx1->size[1] - 1;
    for (i = 0; i <= loop_ub; i++) {
      Sxx_unscaled_data[i] = Sxx_unscaled->data[Sxx_unscaled->size[0] * i];
    }

    c_input_sizes_idx_0 = input_sizes_idx_0;
    d_input_sizes_idx_0 = b_input_sizes_idx_0;
    for (i = 0; i <= b_loop_ub; i++) {
      b_Sxx_unscaled_data[i] = Sxx_unscaled->data[Sxx_unscaled->size[0] * i +
        32768];
    }

    emxFree_real_T(&Sxx_unscaled);
    loop_ub = sizes_idx_0;
    i = Sxx->size[0] * Sxx->size[1];
    i1 = input_sizes_idx_0 + b_input_sizes_idx_0;
    Sxx->size[0] = i1 + sizes_idx_0;
    Sxx->size[1] = result;
    emxEnsureCapacity_real_T(Sxx, i);
    for (i = 0; i < result; i++) {
      for (b_loop_ub = 0; b_loop_ub < c_input_sizes_idx_0; b_loop_ub++) {
        Sxx->data[Sxx->size[0] * i] = Sxx_unscaled_data[input_sizes_idx_0 * i];
      }
    }

    for (i = 0; i < result; i++) {
      for (b_loop_ub = 0; b_loop_ub < d_input_sizes_idx_0; b_loop_ub++) {
        Sxx->data[(b_loop_ub + input_sizes_idx_0) + Sxx->size[0] * i] = y->
          data[b_loop_ub + 32767 * i];
      }
    }

    emxFree_real_T(&y);
    for (i = 0; i < result; i++) {
      for (b_loop_ub = 0; b_loop_ub < loop_ub; b_loop_ub++) {
        Sxx->data[i1 + Sxx->size[0] * i] = b_Sxx_unscaled_data[sizes_idx_0 * i];
      }
    }

    i = varargout_2->size[0];
    varargout_2->size[0] = 32769;
    emxEnsureCapacity_real_T(varargout_2, i);
    for (i = 0; i < 32769; i++) {
      varargout_2->data[i] = w2->data[i];
    }
  } else {
    i = Sxx->size[0] * Sxx->size[1];
    Sxx->size[0] = Sxx1->size[0];
    Sxx->size[1] = Sxx1->size[1];
    emxEnsureCapacity_real_T(Sxx, i);
    loop_ub = Sxx1->size[0] * Sxx1->size[1];
    for (i = 0; i < loop_ub; i++) {
      Sxx->data[i] = Sxx1->data[i];
    }

    i = varargout_2->size[0];
    varargout_2->size[0] = 65536;
    emxEnsureCapacity_real_T(varargout_2, i);
    for (i = 0; i < 65536; i++) {
      varargout_2->data[i] = w2->data[i];
    }
  }

  i = varargout_1->size[0] * varargout_1->size[1];
  varargout_1->size[0] = Sxx->size[0];
  varargout_1->size[1] = Sxx->size[1];
  emxEnsureCapacity_real_T(varargout_1, i);
  loop_ub = Sxx->size[0] * Sxx->size[1];
  for (i = 0; i < loop_ub; i++) {
    varargout_1->data[i] = Sxx->data[i] / 50.0;
  }

  emxFree_real_T(&Sxx);
}

static void d_FFTImplementationCallback_doH(const double x[8], emxArray_creal_T *
  y, int nRows, int nfft, const emxArray_creal_T *wwc, const emxArray_real_T
  *costab, const emxArray_real_T *sintab, const emxArray_real_T *costabinv,
  const emxArray_real_T *sintabinv)
{
  emxArray_creal_T *ytmp;
  int hnRows;
  int ix;
  emxArray_real_T *unusedU0;
  emxArray_int32_T *wrapIndex;
  emxArray_real_T *costable;
  emxArray_real_T *sintable;
  emxArray_real_T *hsintab;
  emxArray_real_T *hcostabinv;
  emxArray_real_T *hsintabinv;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  int idx;
  int i;
  int xidx;
  int temp_re_tmp;
  double twid_re;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double temp_re;
  double temp_im;
  double twid_im;
  emxArray_creal_T *fv;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 8) {
    ix = ytmp->size[0];
    ytmp->size[0] = hnRows;
    emxEnsureCapacity_creal_T(ytmp, ix);
    for (ix = 0; ix < hnRows; ix++) {
      ytmp->data[ix].re = 0.0;
      ytmp->data[ix].im = 0.0;
    }
  }

  emxInit_real_T(&unusedU0, 2);
  emxInit_int32_T(&wrapIndex, 2);
  emxInit_real_T(&costable, 2);
  emxInit_real_T(&sintable, 2);
  emxInit_real_T(&hsintab, 2);
  emxInit_real_T(&hcostabinv, 2);
  emxInit_real_T(&hsintabinv, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  d_FFTImplementationCallback_gen(nRows << 1, costable, sintable, unusedU0);
  e_FFTImplementationCallback_get(costab, sintab, costabinv, sintabinv, unusedU0,
    hsintab, hcostabinv, hsintabinv);
  ix = reconVar1->size[0];
  reconVar1->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar1, ix);
  ix = reconVar2->size[0];
  reconVar2->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar2, ix);
  idx = 0;
  ix = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = hnRows;
  emxEnsureCapacity_int32_T(wrapIndex, ix);
  for (i = 0; i < hnRows; i++) {
    reconVar1->data[i].re = sintable->data[idx] + 1.0;
    reconVar1->data[i].im = -costable->data[idx];
    reconVar2->data[i].re = 1.0 - sintable->data[idx];
    reconVar2->data[i].im = costable->data[idx];
    idx += 2;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (hnRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxFree_real_T(&sintable);
  emxFree_real_T(&costable);
  xidx = 1;
  if (8 < nRows) {
    idx = 8;
  } else {
    idx = nRows;
  }

  ix = (int)((double)idx / 2.0);
  for (idx = 0; idx < ix; idx++) {
    temp_re_tmp = (hnRows + idx) - 1;
    twid_re = x[xidx - 1];
    ytmp->data[idx].re = wwc->data[temp_re_tmp].re * twid_re + wwc->
      data[temp_re_tmp].im * x[xidx];
    ytmp->data[idx].im = wwc->data[temp_re_tmp].re * x[xidx] - wwc->
      data[temp_re_tmp].im * twid_re;
    xidx += 2;
  }

  ix++;
  if (ix <= hnRows) {
    for (i = ix; i <= hnRows; i++) {
      ytmp->data[i - 1].re = 0.0;
      ytmp->data[i - 1].im = 0.0;
    }
  }

  emxInit_creal_T(&fy, 1);
  loop_ub_tmp = (int)((double)nfft / 2.0);
  ix = fy->size[0];
  fy->size[0] = loop_ub_tmp;
  emxEnsureCapacity_creal_T(fy, ix);
  if (loop_ub_tmp > ytmp->size[0]) {
    ix = fy->size[0];
    fy->size[0] = loop_ub_tmp;
    emxEnsureCapacity_creal_T(fy, ix);
    for (ix = 0; ix < loop_ub_tmp; ix++) {
      fy->data[ix].re = 0.0;
      fy->data[ix].im = 0.0;
    }
  }

  xidx = ytmp->size[0];
  if (xidx >= loop_ub_tmp) {
    xidx = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= xidx - 2; i++) {
    fy->data[idx] = ytmp->data[ix];
    idx = loop_ub_tmp;
    tst = true;
    while (tst) {
      idx >>= 1;
      ju ^= idx;
      tst = ((ju & idx) == 0);
    }

    idx = ju;
    ix++;
  }

  fy->data[idx] = ytmp->data[ix];
  if (loop_ub_tmp > 1) {
    for (i = 0; i <= iheight; i += 2) {
      temp_re = fy->data[i + 1].re;
      temp_im = fy->data[i + 1].im;
      twid_re = fy->data[i].re;
      twid_im = fy->data[i].im;
      fy->data[i + 1].re = fy->data[i].re - fy->data[i + 1].re;
      fy->data[i + 1].im = fy->data[i].im - fy->data[i + 1].im;
      twid_re += temp_re;
      twid_im += temp_im;
      fy->data[i].re = twid_re;
      fy->data[i].im = twid_im;
    }
  }

  idx = 2;
  xidx = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += xidx) {
      temp_re_tmp = i + idx;
      temp_re = fy->data[temp_re_tmp].re;
      temp_im = fy->data[temp_re_tmp].im;
      fy->data[temp_re_tmp].re = fy->data[i].re - fy->data[temp_re_tmp].re;
      fy->data[temp_re_tmp].im = fy->data[i].im - fy->data[temp_re_tmp].im;
      fy->data[i].re += temp_re;
      fy->data[i].im += temp_im;
    }

    ix = 1;
    for (ju = k; ju < nRowsD2; ju += k) {
      twid_re = unusedU0->data[ju];
      twid_im = hsintab->data[ju];
      i = ix;
      ihi = ix + iheight;
      while (i < ihi) {
        temp_re_tmp = i + idx;
        temp_re = twid_re * fy->data[temp_re_tmp].re - twid_im * fy->
          data[temp_re_tmp].im;
        temp_im = twid_re * fy->data[temp_re_tmp].im + twid_im * fy->
          data[temp_re_tmp].re;
        fy->data[temp_re_tmp].re = fy->data[i].re - temp_re;
        fy->data[temp_re_tmp].im = fy->data[i].im - temp_im;
        fy->data[i].re += temp_re;
        fy->data[i].im += temp_im;
        i += xidx;
      }

      ix++;
    }

    k /= 2;
    idx = xidx;
    xidx += xidx;
    iheight -= idx;
  }

  emxInit_creal_T(&fv, 1);
  c_FFTImplementationCallback_r2b(wwc, loop_ub_tmp, unusedU0, hsintab, fv);
  idx = fy->size[0];
  emxFree_real_T(&hsintab);
  emxFree_real_T(&unusedU0);
  for (ix = 0; ix < idx; ix++) {
    twid_im = fy->data[ix].re * fv->data[ix].im + fy->data[ix].im * fv->data[ix]
      .re;
    fy->data[ix].re = fy->data[ix].re * fv->data[ix].re - fy->data[ix].im *
      fv->data[ix].im;
    fy->data[ix].im = twid_im;
  }

  c_FFTImplementationCallback_r2b(fy, loop_ub_tmp, hcostabinv, hsintabinv, fv);
  emxFree_creal_T(&fy);
  emxFree_real_T(&hsintabinv);
  emxFree_real_T(&hcostabinv);
  if (fv->size[0] > 1) {
    twid_re = 1.0 / (double)fv->size[0];
    idx = fv->size[0];
    for (ix = 0; ix < idx; ix++) {
      fv->data[ix].re *= twid_re;
      fv->data[ix].im *= twid_re;
    }
  }

  idx = 0;
  ix = wwc->size[0];
  for (k = hnRows; k <= ix; k++) {
    ytmp->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re + wwc->data[k
      - 1].im * fv->data[k - 1].im;
    ytmp->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im - wwc->data[k
      - 1].im * fv->data[k - 1].re;
    idx++;
  }

  emxFree_creal_T(&fv);
  for (i = 0; i < hnRows; i++) {
    ix = wrapIndex->data[i];
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    y->data[i].re = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].re -
      ytmp->data[i].im * reconVar1->data[i].im) + (twid_re * reconVar2->data[i].
      re - twid_im * reconVar2->data[i].im));
    y->data[i].im = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].im +
      ytmp->data[i].im * reconVar1->data[i].re) + (twid_re * reconVar2->data[i].
      im + twid_im * reconVar2->data[i].re));
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    ix = hnRows + i;
    y->data[ix].re = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].re -
      ytmp->data[i].im * reconVar2->data[i].im) + (twid_re * reconVar1->data[i].
      re - twid_im * reconVar1->data[i].im));
    y->data[ix].im = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].im +
      ytmp->data[i].im * reconVar2->data[i].re) + (twid_re * reconVar1->data[i].
      im + twid_im * reconVar1->data[i].re));
  }

  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
  emxFree_creal_T(&ytmp);
}

static void d_FFTImplementationCallback_gen(int nRows, emxArray_real_T *costab,
  emxArray_real_T *sintab, emxArray_real_T *sintabinv)
{
  emxArray_real_T *costab1q;
  double e;
  int n;
  int i;
  int nd2;
  int k;
  emxInit_real_T(&costab1q, 2);
  e = 6.2831853071795862 / (double)nRows;
  n = nRows / 2 / 2;
  i = costab1q->size[0] * costab1q->size[1];
  costab1q->size[0] = 1;
  costab1q->size[1] = n + 1;
  emxEnsureCapacity_real_T(costab1q, i);
  costab1q->data[0] = 1.0;
  nd2 = n / 2 - 1;
  for (k = 0; k <= nd2; k++) {
    costab1q->data[k + 1] = cos(e * ((double)k + 1.0));
  }

  i = nd2 + 2;
  nd2 = n - 1;
  for (k = i; k <= nd2; k++) {
    costab1q->data[k] = sin(e * (double)(n - k));
  }

  costab1q->data[n] = 0.0;
  n = costab1q->size[1] - 1;
  nd2 = (costab1q->size[1] - 1) << 1;
  i = costab->size[0] * costab->size[1];
  costab->size[0] = 1;
  costab->size[1] = nd2 + 1;
  emxEnsureCapacity_real_T(costab, i);
  i = sintab->size[0] * sintab->size[1];
  sintab->size[0] = 1;
  sintab->size[1] = nd2 + 1;
  emxEnsureCapacity_real_T(sintab, i);
  costab->data[0] = 1.0;
  sintab->data[0] = 0.0;
  i = sintabinv->size[0] * sintabinv->size[1];
  sintabinv->size[0] = 1;
  sintabinv->size[1] = nd2 + 1;
  emxEnsureCapacity_real_T(sintabinv, i);
  for (k = 0; k < n; k++) {
    sintabinv->data[k + 1] = costab1q->data[(n - k) - 1];
  }

  i = costab1q->size[1];
  for (k = i; k <= nd2; k++) {
    sintabinv->data[k] = costab1q->data[k - n];
  }

  for (k = 0; k < n; k++) {
    costab->data[k + 1] = costab1q->data[k + 1];
    sintab->data[k + 1] = -costab1q->data[(n - k) - 1];
  }

  i = costab1q->size[1];
  for (k = i; k <= nd2; k++) {
    costab->data[k] = -costab1q->data[nd2 - k];
    sintab->data[k] = -costab1q->data[k - n];
  }

  emxFree_real_T(&costab1q);
}

static void d_FFTImplementationCallback_get(int nRowsM1, int nfftLen,
  emxArray_int32_T *bitrevIndex)
{
  int ju;
  int iy;
  int b_j1;
  boolean_T tst;
  ju = 0;
  iy = 1;
  b_j1 = bitrevIndex->size[0];
  bitrevIndex->size[0] = nfftLen;
  emxEnsureCapacity_int32_T(bitrevIndex, b_j1);
  for (b_j1 = 0; b_j1 < nfftLen; b_j1++) {
    bitrevIndex->data[b_j1] = 0;
  }

  for (b_j1 = 0; b_j1 < nRowsM1; b_j1++) {
    bitrevIndex->data[b_j1] = iy;
    iy = nfftLen;
    tst = true;
    while (tst) {
      iy >>= 1;
      ju ^= iy;
      tst = ((ju & iy) == 0);
    }

    iy = ju + 1;
  }

  bitrevIndex->data[nRowsM1] = iy;
}

static void d_FFTImplementationCallback_r2b(const emxArray_real_T *x, const
  emxArray_real_T *costab, const emxArray_real_T *sintab, emxArray_creal_T *y)
{
  int i;
  int loop_ub;
  int i1;
  int i2;
  i = y->size[0] * y->size[1];
  y->size[0] = 65536;
  y->size[1] = x->size[1];
  emxEnsureCapacity_creal_T(y, i);
  if (65536 > x->size[0]) {
    i = y->size[0] * y->size[1];
    y->size[0] = 65536;
    emxEnsureCapacity_creal_T(y, i);
    loop_ub = x->size[1];
    for (i = 0; i < loop_ub; i++) {
      for (i1 = 0; i1 < 65536; i1++) {
        i2 = i1 + 65536 * i;
        y->data[i2].re = 0.0;
        y->data[i2].im = 0.0;
      }
    }
  }

  k_FFTImplementationCallback_doH(x, y, x->size[1], costab, sintab);
}

static void d_fft(const emxArray_real_T *x, emxArray_creal_T *y)
{
  emxArray_real_T *costab;
  unsigned int ySize_idx_1;
  emxArray_real_T *sintab;
  int i;
  int sintabinv_size[2];
  int loop_ub;
  if ((x->size[0] == 0) || (x->size[1] == 0)) {
    ySize_idx_1 = (unsigned int)x->size[1];
    i = y->size[0] * y->size[1];
    y->size[0] = 65536;
    y->size[1] = (int)ySize_idx_1;
    emxEnsureCapacity_creal_T(y, i);
    loop_ub = (int)ySize_idx_1 << 16;
    for (i = 0; i < loop_ub; i++) {
      y->data[i].re = 0.0;
      y->data[i].im = 0.0;
    }
  } else {
    emxInit_real_T(&costab, 2);
    emxInit_real_T(&sintab, 2);
    e_FFTImplementationCallback_gen(costab, sintab, sintabinv_size);
    d_FFTImplementationCallback_r2b(x, costab, sintab, y);
    emxFree_real_T(&sintab);
    emxFree_real_T(&costab);
  }
}

static int div_s32(int numerator, int denominator)
{
  int quotient;
  unsigned int b_numerator;
  unsigned int b_denominator;
  if (denominator == 0) {
    if (numerator >= 0) {
      quotient = MAX_int32_T;
    } else {
      quotient = MIN_int32_T;
    }
  } else {
    if (numerator < 0) {
      b_numerator = ~(unsigned int)numerator + 1U;
    } else {
      b_numerator = (unsigned int)numerator;
    }

    if (denominator < 0) {
      b_denominator = ~(unsigned int)denominator + 1U;
    } else {
      b_denominator = (unsigned int)denominator;
    }

    b_numerator /= b_denominator;
    if ((numerator < 0) != (denominator < 0)) {
      quotient = -(int)b_numerator;
    } else {
      quotient = (int)b_numerator;
    }
  }

  return quotient;
}

static int div_s32_floor(int numerator, int denominator)
{
  int quotient;
  unsigned int absNumerator;
  unsigned int absDenominator;
  boolean_T quotientNeedsNegation;
  unsigned int tempAbsQuotient;
  if (denominator == 0) {
    if (numerator >= 0) {
      quotient = MAX_int32_T;
    } else {
      quotient = MIN_int32_T;
    }
  } else {
    if (numerator < 0) {
      absNumerator = ~(unsigned int)numerator + 1U;
    } else {
      absNumerator = (unsigned int)numerator;
    }

    if (denominator < 0) {
      absDenominator = ~(unsigned int)denominator + 1U;
    } else {
      absDenominator = (unsigned int)denominator;
    }

    quotientNeedsNegation = ((numerator < 0) != (denominator < 0));
    tempAbsQuotient = absNumerator / absDenominator;
    if (quotientNeedsNegation) {
      absNumerator %= absDenominator;
      if (absNumerator > 0U) {
        tempAbsQuotient++;
      }

      quotient = -(int)tempAbsQuotient;
    } else {
      quotient = (int)tempAbsQuotient;
    }
  }

  return quotient;
}

static void e_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int unsigned_nRows, const emxArray_real_T *costab, const
  emxArray_real_T *sintab)
{
  emxArray_real_T *hcostab;
  emxArray_real_T *hsintab;
  int nRows;
  int istart;
  int nRowsD2;
  int k;
  int hszCostab;
  int iDelta;
  int i;
  emxArray_int32_T *wrapIndex;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  emxArray_int32_T *bitrevIndex;
  double z;
  double temp_re;
  boolean_T nxeven;
  double temp_im;
  int temp_re_tmp;
  int j;
  double twid_re;
  double twid_im;
  int ihi;
  emxInit_real_T(&hcostab, 2);
  emxInit_real_T(&hsintab, 2);
  nRows = unsigned_nRows / 2;
  istart = nRows - 2;
  nRowsD2 = nRows / 2;
  k = nRowsD2 / 2;
  hszCostab = costab->size[1] / 2;
  iDelta = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostab, iDelta);
  iDelta = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintab, iDelta);
  for (i = 0; i < hszCostab; i++) {
    iDelta = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[iDelta];
    hsintab->data[i] = sintab->data[iDelta];
  }

  emxInit_int32_T(&wrapIndex, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  iDelta = reconVar1->size[0];
  reconVar1->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar1, iDelta);
  iDelta = reconVar2->size[0];
  reconVar2->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar2, iDelta);
  iDelta = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = nRows;
  emxEnsureCapacity_int32_T(wrapIndex, iDelta);
  for (i = 0; i < nRows; i++) {
    z = sintab->data[i];
    temp_re = costab->data[i];
    reconVar1->data[i].re = z + 1.0;
    reconVar1->data[i].im = -temp_re;
    reconVar2->data[i].re = 1.0 - z;
    reconVar2->data[i].im = temp_re;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (nRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxInit_int32_T(&bitrevIndex, 1);
  z = (double)unsigned_nRows / 2.0;
  iDelta = y->size[0];
  if (iDelta >= nRows) {
    iDelta = nRows;
  }

  d_FFTImplementationCallback_get(iDelta - 1, (int)z, bitrevIndex);
  if ((x->size[0] & 1) == 0) {
    nxeven = true;
    iDelta = x->size[0];
  } else if (x->size[0] >= unsigned_nRows) {
    nxeven = true;
    iDelta = unsigned_nRows;
  } else {
    nxeven = false;
    iDelta = x->size[0] - 1;
  }

  hszCostab = 0;
  if (iDelta >= unsigned_nRows) {
    iDelta = unsigned_nRows;
  }

  iDelta = (int)((double)iDelta / 2.0);
  for (i = 0; i < iDelta; i++) {
    y->data[bitrevIndex->data[i] - 1].re = x->data[hszCostab];
    y->data[bitrevIndex->data[i] - 1].im = x->data[hszCostab + 1];
    hszCostab += 2;
  }

  if (!nxeven) {
    iDelta = bitrevIndex->data[iDelta] - 1;
    y->data[iDelta].re = x->data[hszCostab];
    y->data[iDelta].im = 0.0;
  }

  emxFree_int32_T(&bitrevIndex);
  if (nRows > 1) {
    for (i = 0; i <= istart; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }
  }

  iDelta = 2;
  hszCostab = 4;
  nRows = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < nRows; i += hszCostab) {
      temp_re_tmp = i + iDelta;
      temp_re = y->data[temp_re_tmp].re;
      temp_im = y->data[temp_re_tmp].im;
      y->data[temp_re_tmp].re = y->data[i].re - temp_re;
      y->data[temp_re_tmp].im = y->data[i].im - temp_im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      twid_re = hcostab->data[j];
      twid_im = hsintab->data[j];
      i = istart;
      ihi = istart + nRows;
      while (i < ihi) {
        temp_re_tmp = i + iDelta;
        temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
          data[temp_re_tmp].im;
        temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
          data[temp_re_tmp].re;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
        i += hszCostab;
      }

      istart++;
    }

    k /= 2;
    iDelta = hszCostab;
    hszCostab += hszCostab;
    nRows -= iDelta;
  }

  emxFree_real_T(&hsintab);
  emxFree_real_T(&hcostab);
  f_FFTImplementationCallback_get(y, reconVar1, reconVar2, wrapIndex, (int)z);
  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
}

static void e_FFTImplementationCallback_gen(emxArray_real_T *costab,
  emxArray_real_T *sintab, int sintabinv_size[2])
{
  emxArray_real_T *costab1q;
  int i;
  int k;
  emxArray_real_T *b_costab;
  emxArray_real_T *b_sintab;
  int n;
  int n2;
  emxInit_real_T(&costab1q, 2);
  i = costab1q->size[0] * costab1q->size[1];
  costab1q->size[0] = 1;
  costab1q->size[1] = 16385;
  emxEnsureCapacity_real_T(costab1q, i);
  costab1q->data[0] = 1.0;
  for (k = 0; k < 8192; k++) {
    costab1q->data[k + 1] = cos(9.5873799242852573E-5 * ((double)k + 1.0));
  }

  for (k = 0; k < 8191; k++) {
    costab1q->data[k + 8193] = sin(9.5873799242852573E-5 * (16384.0 - ((double)k
      + 8193.0)));
  }

  emxInit_real_T(&b_costab, 2);
  emxInit_real_T(&b_sintab, 2);
  costab1q->data[16384] = 0.0;
  n = costab1q->size[1] - 1;
  n2 = (costab1q->size[1] - 1) << 1;
  i = b_costab->size[0] * b_costab->size[1];
  b_costab->size[0] = 1;
  b_costab->size[1] = (unsigned short)(n2 + 1);
  emxEnsureCapacity_real_T(b_costab, i);
  i = b_sintab->size[0] * b_sintab->size[1];
  b_sintab->size[0] = 1;
  b_sintab->size[1] = (unsigned short)(n2 + 1);
  emxEnsureCapacity_real_T(b_sintab, i);
  b_costab->data[0] = 1.0;
  b_sintab->data[0] = 0.0;
  for (k = 0; k < n; k++) {
    b_costab->data[k + 1] = costab1q->data[k + 1];
    b_sintab->data[k + 1] = -costab1q->data[(n - k) - 1];
  }

  i = costab1q->size[1];
  for (k = i; k <= n2; k++) {
    b_costab->data[k] = -costab1q->data[n2 - k];
    b_sintab->data[k] = -costab1q->data[k - n];
  }

  emxFree_real_T(&costab1q);
  i = costab->size[0] * costab->size[1];
  costab->size[0] = 1;
  costab->size[1] = b_costab->size[1];
  emxEnsureCapacity_real_T(costab, i);
  n = b_costab->size[0] * b_costab->size[1];
  for (i = 0; i < n; i++) {
    costab->data[i] = b_costab->data[i];
  }

  emxFree_real_T(&b_costab);
  i = sintab->size[0] * sintab->size[1];
  sintab->size[0] = 1;
  sintab->size[1] = b_sintab->size[1];
  emxEnsureCapacity_real_T(sintab, i);
  n = b_sintab->size[0] * b_sintab->size[1];
  for (i = 0; i < n; i++) {
    sintab->data[i] = b_sintab->data[i];
  }

  emxFree_real_T(&b_sintab);
  sintabinv_size[0] = 1;
  sintabinv_size[1] = 0;
}

static void e_FFTImplementationCallback_get(const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv, emxArray_real_T *hcostab, emxArray_real_T *hsintab,
  emxArray_real_T *hcostabinv, emxArray_real_T *hsintabinv)
{
  int hszCostab;
  int hcostab_tmp;
  int i;
  hszCostab = costab->size[1] / 2;
  hcostab_tmp = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostab, hcostab_tmp);
  hcostab_tmp = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintab, hcostab_tmp);
  hcostab_tmp = hcostabinv->size[0] * hcostabinv->size[1];
  hcostabinv->size[0] = 1;
  hcostabinv->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostabinv, hcostab_tmp);
  hcostab_tmp = hsintabinv->size[0] * hsintabinv->size[1];
  hsintabinv->size[0] = 1;
  hsintabinv->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintabinv, hcostab_tmp);
  for (i = 0; i < hszCostab; i++) {
    hcostab_tmp = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[hcostab_tmp];
    hsintab->data[i] = sintab->data[hcostab_tmp];
    hcostabinv->data[i] = costabinv->data[hcostab_tmp];
    hsintabinv->data[i] = sintabinv->data[hcostab_tmp];
  }
}

static void f_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int nrowsx, int nRows, int nfft, const emxArray_creal_T
  *wwc, const emxArray_real_T *costab, const emxArray_real_T *sintab, const
  emxArray_real_T *costabinv, const emxArray_real_T *sintabinv)
{
  emxArray_creal_T *ytmp;
  int hnRows;
  int ju;
  boolean_T tst;
  int j;
  emxArray_real_T *unusedU0;
  emxArray_int32_T *wrapIndex;
  emxArray_real_T *costable;
  emxArray_real_T *sintable;
  emxArray_real_T *hsintab;
  emxArray_real_T *hcostabinv;
  emxArray_real_T *hsintabinv;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  int idx;
  int i;
  int ix;
  int temp_re_tmp;
  double twid_im;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iDelta2;
  int nRowsD2;
  int k;
  double temp_re;
  double temp_im;
  double twid_re;
  emxArray_creal_T *fv;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ju = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ju);
  if (hnRows > nrowsx) {
    ju = ytmp->size[0];
    ytmp->size[0] = hnRows;
    emxEnsureCapacity_creal_T(ytmp, ju);
    for (ju = 0; ju < hnRows; ju++) {
      ytmp->data[ju].re = 0.0;
      ytmp->data[ju].im = 0.0;
    }
  }

  if ((x->size[0] & 1) == 0) {
    tst = true;
    j = x->size[0];
  } else if (x->size[0] >= nRows) {
    tst = true;
    j = nRows;
  } else {
    tst = false;
    j = x->size[0] - 1;
  }

  emxInit_real_T(&unusedU0, 2);
  emxInit_int32_T(&wrapIndex, 2);
  emxInit_real_T(&costable, 2);
  emxInit_real_T(&sintable, 2);
  emxInit_real_T(&hsintab, 2);
  emxInit_real_T(&hcostabinv, 2);
  emxInit_real_T(&hsintabinv, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  if (j >= nRows) {
    j = nRows;
  }

  d_FFTImplementationCallback_gen(nRows << 1, costable, sintable, unusedU0);
  e_FFTImplementationCallback_get(costab, sintab, costabinv, sintabinv, unusedU0,
    hsintab, hcostabinv, hsintabinv);
  ju = reconVar1->size[0];
  reconVar1->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar1, ju);
  ju = reconVar2->size[0];
  reconVar2->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar2, ju);
  idx = 0;
  ju = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = hnRows;
  emxEnsureCapacity_int32_T(wrapIndex, ju);
  for (i = 0; i < hnRows; i++) {
    reconVar1->data[i].re = sintable->data[idx] + 1.0;
    reconVar1->data[i].im = -costable->data[idx];
    reconVar2->data[i].re = 1.0 - sintable->data[idx];
    reconVar2->data[i].im = costable->data[idx];
    idx += 2;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (hnRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxFree_real_T(&sintable);
  emxFree_real_T(&costable);
  idx = 0;
  ju = (int)((double)j / 2.0);
  for (ix = 0; ix < ju; ix++) {
    temp_re_tmp = (hnRows + ix) - 1;
    twid_im = x->data[idx + 1];
    ytmp->data[ix].re = wwc->data[temp_re_tmp].re * x->data[idx] + wwc->
      data[temp_re_tmp].im * twid_im;
    ytmp->data[ix].im = wwc->data[temp_re_tmp].re * twid_im - wwc->
      data[temp_re_tmp].im * x->data[idx];
    idx += 2;
  }

  if (!tst) {
    temp_re_tmp = (hnRows + ju) - 1;
    ytmp->data[ju].re = wwc->data[temp_re_tmp].re * x->data[idx];
    ytmp->data[ju].im = 0.0 - wwc->data[temp_re_tmp].im * x->data[idx];
    if (ju + 2 <= hnRows) {
      ju = (int)((double)j / 2.0) + 2;
      for (i = ju; i <= hnRows; i++) {
        ytmp->data[i - 1].re = 0.0;
        ytmp->data[i - 1].im = 0.0;
      }
    }
  } else {
    if (ju + 1 <= hnRows) {
      ju = (int)((double)j / 2.0) + 1;
      for (i = ju; i <= hnRows; i++) {
        ytmp->data[i - 1].re = 0.0;
        ytmp->data[i - 1].im = 0.0;
      }
    }
  }

  emxInit_creal_T(&fy, 1);
  loop_ub_tmp = (int)((double)nfft / 2.0);
  ju = fy->size[0];
  fy->size[0] = loop_ub_tmp;
  emxEnsureCapacity_creal_T(fy, ju);
  if (loop_ub_tmp > ytmp->size[0]) {
    ju = fy->size[0];
    fy->size[0] = loop_ub_tmp;
    emxEnsureCapacity_creal_T(fy, ju);
    for (ju = 0; ju < loop_ub_tmp; ju++) {
      fy->data[ju].re = 0.0;
      fy->data[ju].im = 0.0;
    }
  }

  j = ytmp->size[0];
  if (j >= loop_ub_tmp) {
    j = loop_ub_tmp;
  }

  iDelta2 = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= j - 2; i++) {
    fy->data[idx] = ytmp->data[ix];
    idx = loop_ub_tmp;
    tst = true;
    while (tst) {
      idx >>= 1;
      ju ^= idx;
      tst = ((ju & idx) == 0);
    }

    idx = ju;
    ix++;
  }

  fy->data[idx] = ytmp->data[ix];
  if (loop_ub_tmp > 1) {
    for (i = 0; i <= iDelta2; i += 2) {
      temp_re = fy->data[i + 1].re;
      temp_im = fy->data[i + 1].im;
      twid_re = fy->data[i].re;
      twid_im = fy->data[i].im;
      fy->data[i + 1].re = fy->data[i].re - fy->data[i + 1].re;
      fy->data[i + 1].im = fy->data[i].im - fy->data[i + 1].im;
      twid_re += temp_re;
      twid_im += temp_im;
      fy->data[i].re = twid_re;
      fy->data[i].im = twid_im;
    }
  }

  idx = 2;
  iDelta2 = 4;
  ix = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < ix; i += iDelta2) {
      temp_re_tmp = i + idx;
      temp_re = fy->data[temp_re_tmp].re;
      temp_im = fy->data[temp_re_tmp].im;
      fy->data[temp_re_tmp].re = fy->data[i].re - fy->data[temp_re_tmp].re;
      fy->data[temp_re_tmp].im = fy->data[i].im - fy->data[temp_re_tmp].im;
      fy->data[i].re += temp_re;
      fy->data[i].im += temp_im;
    }

    ju = 1;
    for (j = k; j < nRowsD2; j += k) {
      twid_re = unusedU0->data[j];
      twid_im = hsintab->data[j];
      i = ju;
      ihi = ju + ix;
      while (i < ihi) {
        temp_re_tmp = i + idx;
        temp_re = twid_re * fy->data[temp_re_tmp].re - twid_im * fy->
          data[temp_re_tmp].im;
        temp_im = twid_re * fy->data[temp_re_tmp].im + twid_im * fy->
          data[temp_re_tmp].re;
        fy->data[temp_re_tmp].re = fy->data[i].re - temp_re;
        fy->data[temp_re_tmp].im = fy->data[i].im - temp_im;
        fy->data[i].re += temp_re;
        fy->data[i].im += temp_im;
        i += iDelta2;
      }

      ju++;
    }

    k /= 2;
    idx = iDelta2;
    iDelta2 += iDelta2;
    ix -= idx;
  }

  emxInit_creal_T(&fv, 1);
  c_FFTImplementationCallback_r2b(wwc, loop_ub_tmp, unusedU0, hsintab, fv);
  idx = fy->size[0];
  emxFree_real_T(&hsintab);
  emxFree_real_T(&unusedU0);
  for (ju = 0; ju < idx; ju++) {
    twid_im = fy->data[ju].re * fv->data[ju].im + fy->data[ju].im * fv->data[ju]
      .re;
    fy->data[ju].re = fy->data[ju].re * fv->data[ju].re - fy->data[ju].im *
      fv->data[ju].im;
    fy->data[ju].im = twid_im;
  }

  c_FFTImplementationCallback_r2b(fy, loop_ub_tmp, hcostabinv, hsintabinv, fv);
  emxFree_creal_T(&fy);
  emxFree_real_T(&hsintabinv);
  emxFree_real_T(&hcostabinv);
  if (fv->size[0] > 1) {
    twid_re = 1.0 / (double)fv->size[0];
    idx = fv->size[0];
    for (ju = 0; ju < idx; ju++) {
      fv->data[ju].re *= twid_re;
      fv->data[ju].im *= twid_re;
    }
  }

  idx = 0;
  ju = wwc->size[0];
  for (k = hnRows; k <= ju; k++) {
    ytmp->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re + wwc->data[k
      - 1].im * fv->data[k - 1].im;
    ytmp->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im - wwc->data[k
      - 1].im * fv->data[k - 1].re;
    idx++;
  }

  emxFree_creal_T(&fv);
  for (i = 0; i < hnRows; i++) {
    ju = wrapIndex->data[i];
    twid_re = ytmp->data[ju - 1].re;
    twid_im = -ytmp->data[ju - 1].im;
    y->data[i].re = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].re -
      ytmp->data[i].im * reconVar1->data[i].im) + (twid_re * reconVar2->data[i].
      re - twid_im * reconVar2->data[i].im));
    y->data[i].im = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].im +
      ytmp->data[i].im * reconVar1->data[i].re) + (twid_re * reconVar2->data[i].
      im + twid_im * reconVar2->data[i].re));
    twid_re = ytmp->data[ju - 1].re;
    twid_im = -ytmp->data[ju - 1].im;
    ju = hnRows + i;
    y->data[ju].re = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].re -
      ytmp->data[i].im * reconVar2->data[i].im) + (twid_re * reconVar1->data[i].
      re - twid_im * reconVar1->data[i].im));
    y->data[ju].im = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].im +
      ytmp->data[i].im * reconVar2->data[i].re) + (twid_re * reconVar1->data[i].
      im + twid_im * reconVar1->data[i].re));
  }

  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
  emxFree_creal_T(&ytmp);
}

static void f_FFTImplementationCallback_get(emxArray_creal_T *y, const
  emxArray_creal_T *reconVar1, const emxArray_creal_T *reconVar2, const
  emxArray_int32_T *wrapIndex, int hnRows)
{
  int iterVar;
  double temp1_re;
  double temp1_im;
  double y_im;
  double y_re;
  double b_y_im;
  int i;
  int b_i;
  double temp2_re;
  double temp2_im;
  int i1;
  iterVar = hnRows / 2;
  temp1_re = y->data[0].re;
  temp1_im = y->data[0].im;
  y_im = y->data[0].re * reconVar1->data[0].im + y->data[0].im * reconVar1->
    data[0].re;
  y_re = y->data[0].re;
  b_y_im = -y->data[0].im;
  y->data[0].re = 0.5 * ((y->data[0].re * reconVar1->data[0].re - y->data[0].im *
    reconVar1->data[0].im) + (y_re * reconVar2->data[0].re - b_y_im *
    reconVar2->data[0].im));
  y->data[0].im = 0.5 * (y_im + (y_re * reconVar2->data[0].im + b_y_im *
    reconVar2->data[0].re));
  y->data[hnRows].re = 0.5 * ((temp1_re * reconVar2->data[0].re - temp1_im *
    reconVar2->data[0].im) + (temp1_re * reconVar1->data[0].re - -temp1_im *
    reconVar1->data[0].im));
  y->data[hnRows].im = 0.5 * ((temp1_re * reconVar2->data[0].im + temp1_im *
    reconVar2->data[0].re) + (temp1_re * reconVar1->data[0].im + -temp1_im *
    reconVar1->data[0].re));
  for (i = 2; i <= iterVar; i++) {
    temp1_re = y->data[i - 1].re;
    temp1_im = y->data[i - 1].im;
    b_i = wrapIndex->data[i - 1];
    temp2_re = y->data[b_i - 1].re;
    temp2_im = y->data[b_i - 1].im;
    y_im = y->data[i - 1].re * reconVar1->data[i - 1].im + y->data[i - 1].im *
      reconVar1->data[i - 1].re;
    y_re = y->data[b_i - 1].re;
    b_y_im = -y->data[b_i - 1].im;
    y->data[i - 1].re = 0.5 * ((y->data[i - 1].re * reconVar1->data[i - 1].re -
      y->data[i - 1].im * reconVar1->data[i - 1].im) + (y_re * reconVar2->data[i
      - 1].re - b_y_im * reconVar2->data[i - 1].im));
    y->data[i - 1].im = 0.5 * (y_im + (y_re * reconVar2->data[i - 1].im + b_y_im
      * reconVar2->data[i - 1].re));
    i1 = (hnRows + i) - 1;
    y->data[i1].re = 0.5 * ((temp1_re * reconVar2->data[i - 1].re - temp1_im *
      reconVar2->data[i - 1].im) + (temp2_re * reconVar1->data[i - 1].re -
      -temp2_im * reconVar1->data[i - 1].im));
    y->data[i1].im = 0.5 * ((temp1_re * reconVar2->data[i - 1].im + temp1_im *
      reconVar2->data[i - 1].re) + (temp2_re * reconVar1->data[i - 1].im +
      -temp2_im * reconVar1->data[i - 1].re));
    y->data[b_i - 1].re = 0.5 * ((temp2_re * reconVar1->data[b_i - 1].re -
      temp2_im * reconVar1->data[b_i - 1].im) + (temp1_re * reconVar2->data[b_i
      - 1].re - -temp1_im * reconVar2->data[b_i - 1].im));
    y->data[b_i - 1].im = 0.5 * ((temp2_re * reconVar1->data[b_i - 1].im +
      temp2_im * reconVar1->data[b_i - 1].re) + (temp1_re * reconVar2->data[b_i
      - 1].im + -temp1_im * reconVar2->data[b_i - 1].re));
    i1 = (b_i + hnRows) - 1;
    y->data[i1].re = 0.5 * ((temp2_re * reconVar2->data[b_i - 1].re - temp2_im *
      reconVar2->data[b_i - 1].im) + (temp1_re * reconVar1->data[b_i - 1].re -
      -temp1_im * reconVar1->data[b_i - 1].im));
    y->data[i1].im = 0.5 * ((temp2_re * reconVar2->data[b_i - 1].im + temp2_im *
      reconVar2->data[b_i - 1].re) + (temp1_re * reconVar1->data[b_i - 1].im +
      -temp1_im * reconVar1->data[b_i - 1].re));
  }

  if (iterVar != 0) {
    temp1_re = y->data[iterVar].re;
    temp1_im = y->data[iterVar].im;
    y_im = y->data[iterVar].re * reconVar1->data[iterVar].im + y->data[iterVar].
      im * reconVar1->data[iterVar].re;
    y_re = y->data[iterVar].re;
    b_y_im = -y->data[iterVar].im;
    y->data[iterVar].re = 0.5 * ((y->data[iterVar].re * reconVar1->data[iterVar]
      .re - y->data[iterVar].im * reconVar1->data[iterVar].im) + (y_re *
      reconVar2->data[iterVar].re - b_y_im * reconVar2->data[iterVar].im));
    y->data[iterVar].im = 0.5 * (y_im + (y_re * reconVar2->data[iterVar].im +
      b_y_im * reconVar2->data[iterVar].re));
    b_i = hnRows + iterVar;
    y->data[b_i].re = 0.5 * ((temp1_re * reconVar2->data[iterVar].re - temp1_im *
      reconVar2->data[iterVar].im) + (temp1_re * reconVar1->data[iterVar].re -
      -temp1_im * reconVar1->data[iterVar].im));
    y->data[b_i].im = 0.5 * ((temp1_re * reconVar2->data[iterVar].im + temp1_im *
      reconVar2->data[iterVar].re) + (temp1_re * reconVar1->data[iterVar].im +
      -temp1_im * reconVar1->data[iterVar].re));
  }
}

static void fft(const double x[8], double varargin_1, emxArray_creal_T *y)
{
  emxArray_real_T *costab;
  emxArray_real_T *sintab;
  emxArray_real_T *sintabinv;
  int len_tmp;
  int nInt2;
  boolean_T useRadix2;
  int N2blue;
  int nRows;
  emxArray_creal_T *yCol;
  int i;
  emxArray_creal_T *wwc;
  int nInt2m1;
  int xidx;
  int idx;
  int rt;
  int k;
  double nt_im;
  double nt_re;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  emxInit_real_T(&costab, 2);
  emxInit_real_T(&sintab, 2);
  emxInit_real_T(&sintabinv, 2);
  len_tmp = (int)varargin_1;
  nInt2 = len_tmp - 1;
  useRadix2 = ((len_tmp & nInt2) == 0);
  c_FFTImplementationCallback_get((int)varargin_1, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, costab, sintab, sintabinv);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, i);
    if (len_tmp > 8) {
      i = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, i);
      for (i = 0; i < len_tmp; i++) {
        yCol->data[i].re = 0.0;
        yCol->data[i].im = 0.0;
      }
    }

    c_FFTImplementationCallback_doH(x, yCol, (int)varargin_1, costab, sintab);
  } else {
    i = len_tmp & 1;
    emxInit_creal_T(&wwc, 1);
    if (i == 0) {
      nRows = len_tmp / 2;
      nInt2m1 = (nRows + nRows) - 1;
      xidx = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, xidx);
      idx = nRows;
      rt = 0;
      wwc->data[nRows - 1].re = 1.0;
      wwc->data[nRows - 1].im = 0.0;
      nInt2 = nRows << 1;
      for (k = 0; k <= nRows - 2; k++) {
        xidx = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= xidx) {
          rt += xidx - nInt2;
        } else {
          rt += xidx;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)nRows;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      xidx = nInt2m1 - 1;
      for (k = xidx; k >= nRows; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    } else {
      nInt2m1 = (len_tmp + len_tmp) - 1;
      xidx = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, xidx);
      idx = len_tmp;
      rt = 0;
      wwc->data[nInt2].re = 1.0;
      wwc->data[nInt2].im = 0.0;
      nInt2 = len_tmp << 1;
      for (k = 0; k <= len_tmp - 2; k++) {
        xidx = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= xidx) {
          rt += xidx - nInt2;
        } else {
          rt += xidx;
        }

        nt_im = -3.1415926535897931 * (double)rt / (double)len_tmp;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      xidx = nInt2m1 - 1;
      for (k = xidx; k >= len_tmp; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    }

    xidx = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, xidx);
    if (len_tmp > 8) {
      xidx = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, xidx);
      for (xidx = 0; xidx < len_tmp; xidx++) {
        yCol->data[xidx].re = 0.0;
        yCol->data[xidx].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      d_FFTImplementationCallback_doH(x, yCol, (int)varargin_1, N2blue, wwc,
        costab, sintab, costab, sintabinv);
    } else {
      if ((int)varargin_1 < 8) {
        nInt2 = (int)varargin_1 - 1;
      } else {
        nInt2 = 7;
      }

      xidx = 0;
      for (k = 0; k <= nInt2; k++) {
        rt = (len_tmp + k) - 1;
        yCol->data[k].re = wwc->data[rt].re * x[xidx];
        yCol->data[k].im = wwc->data[rt].im * -x[xidx];
        xidx++;
      }

      i = nInt2 + 2;
      for (k = i; k <= len_tmp; k++) {
        yCol->data[k - 1].re = 0.0;
        yCol->data[k - 1].im = 0.0;
      }

      emxInit_creal_T(&fv, 1);
      emxInit_creal_T(&b_fv, 1);
      c_FFTImplementationCallback_r2b(yCol, N2blue, costab, sintab, fv);
      c_FFTImplementationCallback_r2b(wwc, N2blue, costab, sintab, b_fv);
      i = b_fv->size[0];
      b_fv->size[0] = fv->size[0];
      emxEnsureCapacity_creal_T(b_fv, i);
      nInt2 = fv->size[0];
      for (i = 0; i < nInt2; i++) {
        nt_re = fv->data[i].re * b_fv->data[i].im + fv->data[i].im * b_fv->
          data[i].re;
        b_fv->data[i].re = fv->data[i].re * b_fv->data[i].re - fv->data[i].im *
          b_fv->data[i].im;
        b_fv->data[i].im = nt_re;
      }

      c_FFTImplementationCallback_r2b(b_fv, N2blue, costab, sintabinv, fv);
      emxFree_creal_T(&b_fv);
      if (fv->size[0] > 1) {
        nt_re = 1.0 / (double)fv->size[0];
        nInt2 = fv->size[0];
        for (i = 0; i < nInt2; i++) {
          fv->data[i].re *= nt_re;
          fv->data[i].im *= nt_re;
        }
      }

      idx = 0;
      i = wwc->size[0];
      for (k = len_tmp; k <= i; k++) {
        yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re +
          wwc->data[k - 1].im * fv->data[k - 1].im;
        yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im -
          wwc->data[k - 1].im * fv->data[k - 1].re;
        idx++;
      }

      emxFree_creal_T(&fv);
    }

    emxFree_creal_T(&wwc);
  }

  emxFree_real_T(&sintabinv);
  emxFree_real_T(&sintab);
  emxFree_real_T(&costab);
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = len_tmp;
  emxEnsureCapacity_creal_T(y, i);
  for (i = 0; i < len_tmp; i++) {
    y->data[i] = yCol->data[i];
  }

  emxFree_creal_T(&yCol);
}

static void filter(const double b[7], const double a[7], const double x[18],
                   const double zi[6], double y[18], double zf[6])
{
  int k;
  int naxpy;
  int j;
  int y_tmp;
  double as;
  for (k = 0; k < 6; k++) {
    zf[k] = 0.0;
    y[k] = zi[k];
  }

  memset(&y[6], 0, 12U * sizeof(double));
  for (k = 0; k < 18; k++) {
    if (18 - k < 7) {
      naxpy = 17 - k;
    } else {
      naxpy = 6;
    }

    for (j = 0; j <= naxpy; j++) {
      y_tmp = k + j;
      y[y_tmp] += x[k] * b[j];
    }

    if (17 - k < 6) {
      naxpy = 16 - k;
    } else {
      naxpy = 5;
    }

    as = -y[k];
    for (j = 0; j <= naxpy; j++) {
      y_tmp = (k + j) + 1;
      y[y_tmp] += as * a[j + 1];
    }
  }

  for (k = 0; k < 6; k++) {
    for (j = 0; j <= k; j++) {
      zf[j] += x[k + 12] * b[(j - k) + 6];
    }
  }

  for (k = 0; k < 6; k++) {
    for (j = 0; j <= k; j++) {
      zf[j] += -y[k + 12] * a[(j - k) + 6];
    }
  }
}

static void g_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab)
{
  emxArray_real_T *hcostab;
  emxArray_real_T *hsintab;
  int nRows;
  int istart;
  int nRowsD2;
  int k;
  int hszCostab;
  int iDelta;
  int i;
  emxArray_int32_T *wrapIndex;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  emxArray_int32_T *bitrevIndex;
  double z;
  double temp_re;
  double temp_im;
  int temp_re_tmp;
  int j;
  double twid_re;
  double twid_im;
  int ihi;
  emxInit_real_T(&hcostab, 2);
  emxInit_real_T(&hsintab, 2);
  nRows = unsigned_nRows / 2;
  istart = nRows - 2;
  nRowsD2 = nRows / 2;
  k = nRowsD2 / 2;
  hszCostab = costab->size[1] / 2;
  iDelta = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostab, iDelta);
  iDelta = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintab, iDelta);
  for (i = 0; i < hszCostab; i++) {
    iDelta = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[iDelta];
    hsintab->data[i] = sintab->data[iDelta];
  }

  emxInit_int32_T(&wrapIndex, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  iDelta = reconVar1->size[0];
  reconVar1->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar1, iDelta);
  iDelta = reconVar2->size[0];
  reconVar2->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar2, iDelta);
  iDelta = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = nRows;
  emxEnsureCapacity_int32_T(wrapIndex, iDelta);
  for (i = 0; i < nRows; i++) {
    z = sintab->data[i];
    temp_re = costab->data[i];
    reconVar1->data[i].re = z + 1.0;
    reconVar1->data[i].im = -temp_re;
    reconVar2->data[i].re = 1.0 - z;
    reconVar2->data[i].im = temp_re;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (nRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxInit_int32_T(&bitrevIndex, 1);
  z = (double)unsigned_nRows / 2.0;
  iDelta = y->size[0];
  if (iDelta >= nRows) {
    iDelta = nRows;
  }

  d_FFTImplementationCallback_get(iDelta - 1, (int)z, bitrevIndex);
  hszCostab = 0;
  if (8 < unsigned_nRows) {
    iDelta = 8;
  } else {
    iDelta = unsigned_nRows;
  }

  iDelta = (int)((double)iDelta / 2.0);
  for (i = 0; i < iDelta; i++) {
    y->data[bitrevIndex->data[i] - 1].re = dv[hszCostab];
    y->data[bitrevIndex->data[i] - 1].im = dv[hszCostab + 1];
    hszCostab += 2;
  }

  emxFree_int32_T(&bitrevIndex);
  if (nRows > 1) {
    for (i = 0; i <= istart; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }
  }

  iDelta = 2;
  hszCostab = 4;
  nRows = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < nRows; i += hszCostab) {
      temp_re_tmp = i + iDelta;
      temp_re = y->data[temp_re_tmp].re;
      temp_im = y->data[temp_re_tmp].im;
      y->data[temp_re_tmp].re = y->data[i].re - temp_re;
      y->data[temp_re_tmp].im = y->data[i].im - temp_im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      twid_re = hcostab->data[j];
      twid_im = hsintab->data[j];
      i = istart;
      ihi = istart + nRows;
      while (i < ihi) {
        temp_re_tmp = i + iDelta;
        temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
          data[temp_re_tmp].im;
        temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
          data[temp_re_tmp].re;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
        i += hszCostab;
      }

      istart++;
    }

    k /= 2;
    iDelta = hszCostab;
    hszCostab += hszCostab;
    nRows -= iDelta;
  }

  emxFree_real_T(&hsintab);
  emxFree_real_T(&hcostab);
  f_FFTImplementationCallback_get(y, reconVar1, reconVar2, wrapIndex, (int)z);
  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
}

static void g_FFTImplementationCallback_get(emxArray_creal_T *y, const
  emxArray_creal_T *reconVar1, const emxArray_creal_T *reconVar2, const
  emxArray_int32_T *wrapIndex)
{
  double temp1_re;
  double temp1_im;
  double y_im;
  double y_re;
  double b_y_im;
  int i;
  int temp2_re_tmp_tmp;
  int temp2_re_tmp;
  double temp2_re;
  double temp2_im;
  temp1_re = y->data[0].re;
  temp1_im = y->data[0].im;
  y_im = y->data[0].re * reconVar1->data[0].im + y->data[0].im * reconVar1->
    data[0].re;
  y_re = y->data[0].re;
  b_y_im = -y->data[0].im;
  y->data[0].re = 0.5 * ((y->data[0].re * reconVar1->data[0].re - y->data[0].im *
    reconVar1->data[0].im) + (y_re * reconVar2->data[0].re - b_y_im *
    reconVar2->data[0].im));
  y->data[0].im = 0.5 * (y_im + (y_re * reconVar2->data[0].im + b_y_im *
    reconVar2->data[0].re));
  y->data[32768].re = 0.5 * ((temp1_re * reconVar2->data[0].re - temp1_im *
    reconVar2->data[0].im) + (temp1_re * reconVar1->data[0].re - -temp1_im *
    reconVar1->data[0].im));
  y->data[32768].im = 0.5 * ((temp1_re * reconVar2->data[0].im + temp1_im *
    reconVar2->data[0].re) + (temp1_re * reconVar1->data[0].im + -temp1_im *
    reconVar1->data[0].re));
  for (i = 0; i < 16383; i++) {
    temp1_re = y->data[i + 1].re;
    temp1_im = y->data[i + 1].im;
    temp2_re_tmp_tmp = wrapIndex->data[i + 1];
    temp2_re_tmp = temp2_re_tmp_tmp - 1;
    temp2_re = y->data[temp2_re_tmp].re;
    temp2_im = y->data[temp2_re_tmp].im;
    y_im = y->data[i + 1].re * reconVar1->data[i + 1].im + y->data[i + 1].im *
      reconVar1->data[i + 1].re;
    y_re = y->data[temp2_re_tmp].re;
    b_y_im = -y->data[temp2_re_tmp].im;
    y->data[i + 1].re = 0.5 * ((y->data[i + 1].re * reconVar1->data[i + 1].re -
      y->data[i + 1].im * reconVar1->data[i + 1].im) + (y_re * reconVar2->data[i
      + 1].re - b_y_im * reconVar2->data[i + 1].im));
    y->data[i + 1].im = 0.5 * (y_im + (y_re * reconVar2->data[i + 1].im + b_y_im
      * reconVar2->data[i + 1].re));
    y->data[i + 32769].re = 0.5 * ((temp1_re * reconVar2->data[i + 1].re -
      temp1_im * reconVar2->data[i + 1].im) + (temp2_re * reconVar1->data[i + 1]
      .re - -temp2_im * reconVar1->data[i + 1].im));
    y->data[i + 32769].im = 0.5 * ((temp1_re * reconVar2->data[i + 1].im +
      temp1_im * reconVar2->data[i + 1].re) + (temp2_re * reconVar1->data[i + 1]
      .im + -temp2_im * reconVar1->data[i + 1].re));
    y->data[temp2_re_tmp].re = 0.5 * ((temp2_re * reconVar1->data[temp2_re_tmp].
      re - temp2_im * reconVar1->data[temp2_re_tmp].im) + (temp1_re *
      reconVar2->data[temp2_re_tmp].re - -temp1_im * reconVar2->
      data[temp2_re_tmp].im));
    y->data[temp2_re_tmp].im = 0.5 * ((temp2_re * reconVar1->data[temp2_re_tmp].
      im + temp2_im * reconVar1->data[temp2_re_tmp].re) + (temp1_re *
      reconVar2->data[temp2_re_tmp].im + -temp1_im * reconVar2->
      data[temp2_re_tmp].re));
    temp2_re_tmp_tmp += 32767;
    y->data[temp2_re_tmp_tmp].re = 0.5 * ((temp2_re * reconVar2->
      data[temp2_re_tmp].re - temp2_im * reconVar2->data[temp2_re_tmp].im) +
      (temp1_re * reconVar1->data[temp2_re_tmp].re - -temp1_im * reconVar1->
       data[temp2_re_tmp].im));
    y->data[temp2_re_tmp_tmp].im = 0.5 * ((temp2_re * reconVar2->
      data[temp2_re_tmp].im + temp2_im * reconVar2->data[temp2_re_tmp].re) +
      (temp1_re * reconVar1->data[temp2_re_tmp].im + -temp1_im * reconVar1->
       data[temp2_re_tmp].re));
  }

  temp1_re = y->data[16384].re;
  temp1_im = y->data[16384].im;
  y_im = y->data[16384].re * reconVar1->data[16384].im + y->data[16384].im *
    reconVar1->data[16384].re;
  y_re = y->data[16384].re;
  b_y_im = -y->data[16384].im;
  y->data[16384].re = 0.5 * ((y->data[16384].re * reconVar1->data[16384].re -
    y->data[16384].im * reconVar1->data[16384].im) + (y_re * reconVar2->data
    [16384].re - b_y_im * reconVar2->data[16384].im));
  y->data[16384].im = 0.5 * (y_im + (y_re * reconVar2->data[16384].im + b_y_im *
    reconVar2->data[16384].re));
  y->data[49152].re = 0.5 * ((temp1_re * reconVar2->data[16384].re - temp1_im *
    reconVar2->data[16384].im) + (temp1_re * reconVar1->data[16384].re -
    -temp1_im * reconVar1->data[16384].im));
  y->data[49152].im = 0.5 * ((temp1_re * reconVar2->data[16384].im + temp1_im *
    reconVar2->data[16384].re) + (temp1_re * reconVar1->data[16384].im +
    -temp1_im * reconVar1->data[16384].re));
}

static void h_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv)
{
  emxArray_creal_T *ytmp;
  int hnRows;
  int ix;
  emxArray_real_T *unusedU0;
  emxArray_int32_T *wrapIndex;
  emxArray_real_T *costable;
  emxArray_real_T *sintable;
  emxArray_real_T *hsintab;
  emxArray_real_T *hcostabinv;
  emxArray_real_T *hsintabinv;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  int idx;
  int i;
  int xidx;
  int temp_re_tmp;
  double twid_re;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double temp_re;
  double temp_im;
  double twid_im;
  emxArray_creal_T *fv;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 8) {
    ix = ytmp->size[0];
    ytmp->size[0] = hnRows;
    emxEnsureCapacity_creal_T(ytmp, ix);
    for (ix = 0; ix < hnRows; ix++) {
      ytmp->data[ix].re = 0.0;
      ytmp->data[ix].im = 0.0;
    }
  }

  emxInit_real_T(&unusedU0, 2);
  emxInit_int32_T(&wrapIndex, 2);
  emxInit_real_T(&costable, 2);
  emxInit_real_T(&sintable, 2);
  emxInit_real_T(&hsintab, 2);
  emxInit_real_T(&hcostabinv, 2);
  emxInit_real_T(&hsintabinv, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  d_FFTImplementationCallback_gen(nRows << 1, costable, sintable, unusedU0);
  e_FFTImplementationCallback_get(costab, sintab, costabinv, sintabinv, unusedU0,
    hsintab, hcostabinv, hsintabinv);
  ix = reconVar1->size[0];
  reconVar1->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar1, ix);
  ix = reconVar2->size[0];
  reconVar2->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar2, ix);
  idx = 0;
  ix = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = hnRows;
  emxEnsureCapacity_int32_T(wrapIndex, ix);
  for (i = 0; i < hnRows; i++) {
    reconVar1->data[i].re = sintable->data[idx] + 1.0;
    reconVar1->data[i].im = -costable->data[idx];
    reconVar2->data[i].re = 1.0 - sintable->data[idx];
    reconVar2->data[i].im = costable->data[idx];
    idx += 2;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (hnRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxFree_real_T(&sintable);
  emxFree_real_T(&costable);
  xidx = 1;
  if (8 < nRows) {
    idx = 8;
  } else {
    idx = nRows;
  }

  ix = (int)((double)idx / 2.0);
  for (idx = 0; idx < ix; idx++) {
    temp_re_tmp = (hnRows + idx) - 1;
    twid_re = dv[xidx - 1];
    ytmp->data[idx].re = wwc->data[temp_re_tmp].re * twid_re + wwc->
      data[temp_re_tmp].im * dv[xidx];
    ytmp->data[idx].im = wwc->data[temp_re_tmp].re * dv[xidx] - wwc->
      data[temp_re_tmp].im * twid_re;
    xidx += 2;
  }

  ix++;
  if (ix <= hnRows) {
    for (i = ix; i <= hnRows; i++) {
      ytmp->data[i - 1].re = 0.0;
      ytmp->data[i - 1].im = 0.0;
    }
  }

  emxInit_creal_T(&fy, 1);
  loop_ub_tmp = (int)((double)nfft / 2.0);
  ix = fy->size[0];
  fy->size[0] = loop_ub_tmp;
  emxEnsureCapacity_creal_T(fy, ix);
  if (loop_ub_tmp > ytmp->size[0]) {
    ix = fy->size[0];
    fy->size[0] = loop_ub_tmp;
    emxEnsureCapacity_creal_T(fy, ix);
    for (ix = 0; ix < loop_ub_tmp; ix++) {
      fy->data[ix].re = 0.0;
      fy->data[ix].im = 0.0;
    }
  }

  xidx = ytmp->size[0];
  if (xidx >= loop_ub_tmp) {
    xidx = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= xidx - 2; i++) {
    fy->data[idx] = ytmp->data[ix];
    idx = loop_ub_tmp;
    tst = true;
    while (tst) {
      idx >>= 1;
      ju ^= idx;
      tst = ((ju & idx) == 0);
    }

    idx = ju;
    ix++;
  }

  fy->data[idx] = ytmp->data[ix];
  if (loop_ub_tmp > 1) {
    for (i = 0; i <= iheight; i += 2) {
      temp_re = fy->data[i + 1].re;
      temp_im = fy->data[i + 1].im;
      twid_re = fy->data[i].re;
      twid_im = fy->data[i].im;
      fy->data[i + 1].re = fy->data[i].re - fy->data[i + 1].re;
      fy->data[i + 1].im = fy->data[i].im - fy->data[i + 1].im;
      twid_re += temp_re;
      twid_im += temp_im;
      fy->data[i].re = twid_re;
      fy->data[i].im = twid_im;
    }
  }

  idx = 2;
  xidx = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += xidx) {
      temp_re_tmp = i + idx;
      temp_re = fy->data[temp_re_tmp].re;
      temp_im = fy->data[temp_re_tmp].im;
      fy->data[temp_re_tmp].re = fy->data[i].re - fy->data[temp_re_tmp].re;
      fy->data[temp_re_tmp].im = fy->data[i].im - fy->data[temp_re_tmp].im;
      fy->data[i].re += temp_re;
      fy->data[i].im += temp_im;
    }

    ix = 1;
    for (ju = k; ju < nRowsD2; ju += k) {
      twid_re = unusedU0->data[ju];
      twid_im = hsintab->data[ju];
      i = ix;
      ihi = ix + iheight;
      while (i < ihi) {
        temp_re_tmp = i + idx;
        temp_re = twid_re * fy->data[temp_re_tmp].re - twid_im * fy->
          data[temp_re_tmp].im;
        temp_im = twid_re * fy->data[temp_re_tmp].im + twid_im * fy->
          data[temp_re_tmp].re;
        fy->data[temp_re_tmp].re = fy->data[i].re - temp_re;
        fy->data[temp_re_tmp].im = fy->data[i].im - temp_im;
        fy->data[i].re += temp_re;
        fy->data[i].im += temp_im;
        i += xidx;
      }

      ix++;
    }

    k /= 2;
    idx = xidx;
    xidx += xidx;
    iheight -= idx;
  }

  emxInit_creal_T(&fv, 1);
  c_FFTImplementationCallback_r2b(wwc, loop_ub_tmp, unusedU0, hsintab, fv);
  idx = fy->size[0];
  emxFree_real_T(&hsintab);
  emxFree_real_T(&unusedU0);
  for (ix = 0; ix < idx; ix++) {
    twid_im = fy->data[ix].re * fv->data[ix].im + fy->data[ix].im * fv->data[ix]
      .re;
    fy->data[ix].re = fy->data[ix].re * fv->data[ix].re - fy->data[ix].im *
      fv->data[ix].im;
    fy->data[ix].im = twid_im;
  }

  c_FFTImplementationCallback_r2b(fy, loop_ub_tmp, hcostabinv, hsintabinv, fv);
  emxFree_creal_T(&fy);
  emxFree_real_T(&hsintabinv);
  emxFree_real_T(&hcostabinv);
  if (fv->size[0] > 1) {
    twid_re = 1.0 / (double)fv->size[0];
    idx = fv->size[0];
    for (ix = 0; ix < idx; ix++) {
      fv->data[ix].re *= twid_re;
      fv->data[ix].im *= twid_re;
    }
  }

  idx = 0;
  ix = wwc->size[0];
  for (k = hnRows; k <= ix; k++) {
    ytmp->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re + wwc->data[k
      - 1].im * fv->data[k - 1].im;
    ytmp->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im - wwc->data[k
      - 1].im * fv->data[k - 1].re;
    idx++;
  }

  emxFree_creal_T(&fv);
  for (i = 0; i < hnRows; i++) {
    ix = wrapIndex->data[i];
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    y->data[i].re = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].re -
      ytmp->data[i].im * reconVar1->data[i].im) + (twid_re * reconVar2->data[i].
      re - twid_im * reconVar2->data[i].im));
    y->data[i].im = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].im +
      ytmp->data[i].im * reconVar1->data[i].re) + (twid_re * reconVar2->data[i].
      im + twid_im * reconVar2->data[i].re));
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    ix = hnRows + i;
    y->data[ix].re = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].re -
      ytmp->data[i].im * reconVar2->data[i].im) + (twid_re * reconVar1->data[i].
      re - twid_im * reconVar1->data[i].im));
    y->data[ix].im = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].im +
      ytmp->data[i].im * reconVar2->data[i].re) + (twid_re * reconVar1->data[i].
      im + twid_im * reconVar1->data[i].re));
  }

  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
  emxFree_creal_T(&ytmp);
}

static void heartRateFilt(const emxArray_real_T *Y, emxArray_real_T *Y_Filt,
  double HeartRatePxx_data[], int HeartRatePxx_size[1])
{
  emxArray_real_T *pxx;
  emxArray_real_T *fxx;
  int m;
  int idx_size_idx_1;
  int loop_ub;
  int i;
  int idx_data[1];
  double ex_data[1];
  double d;
  emxInit_real_T(&pxx, 2);
  emxInit_real_T(&fxx, 1);
  FiltFiltM(Y, Y_Filt);
  pwelch(Y_Filt, pxx, fxx);
  m = pxx->size[0];
  idx_size_idx_1 = pxx->size[1];
  loop_ub = pxx->size[1];
  for (i = 0; i < loop_ub; i++) {
    idx_data[i] = 1;
  }

  if (pxx->size[1] >= 1) {
    ex_data[0] = pxx->data[0];
    for (loop_ub = 2; loop_ub <= m; loop_ub++) {
      d = pxx->data[loop_ub - 1];
      if (ex_data[0] < d) {
        ex_data[0] = d;
        idx_data[0] = loop_ub;
      }
    }
  }

  emxFree_real_T(&pxx);
  for (i = 0; i < idx_size_idx_1; i++) {
    ex_data[i] = idx_data[i];
  }

  HeartRatePxx_size[0] = idx_size_idx_1;
  for (i = 0; i < idx_size_idx_1; i++) {
    HeartRatePxx_data[i] = fxx->data[(int)ex_data[i] - 1] * 60.0;
  }

  emxFree_real_T(&fxx);
}

static void i_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab)
{
  emxArray_real_T *hcostab;
  emxArray_real_T *hsintab;
  int nRows;
  int istart;
  int nRowsD2;
  int k;
  int hszCostab;
  int iDelta;
  int i;
  emxArray_int32_T *wrapIndex;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  emxArray_int32_T *bitrevIndex;
  double z;
  double temp_re;
  double temp_im;
  int temp_re_tmp;
  int j;
  double twid_re;
  double twid_im;
  int ihi;
  emxInit_real_T(&hcostab, 2);
  emxInit_real_T(&hsintab, 2);
  nRows = unsigned_nRows / 2;
  istart = nRows - 2;
  nRowsD2 = nRows / 2;
  k = nRowsD2 / 2;
  hszCostab = costab->size[1] / 2;
  iDelta = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hcostab, iDelta);
  iDelta = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = hszCostab;
  emxEnsureCapacity_real_T(hsintab, iDelta);
  for (i = 0; i < hszCostab; i++) {
    iDelta = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[iDelta];
    hsintab->data[i] = sintab->data[iDelta];
  }

  emxInit_int32_T(&wrapIndex, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  iDelta = reconVar1->size[0];
  reconVar1->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar1, iDelta);
  iDelta = reconVar2->size[0];
  reconVar2->size[0] = nRows;
  emxEnsureCapacity_creal_T(reconVar2, iDelta);
  iDelta = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = nRows;
  emxEnsureCapacity_int32_T(wrapIndex, iDelta);
  for (i = 0; i < nRows; i++) {
    z = sintab->data[i];
    temp_re = costab->data[i];
    reconVar1->data[i].re = z + 1.0;
    reconVar1->data[i].im = -temp_re;
    reconVar2->data[i].re = 1.0 - z;
    reconVar2->data[i].im = temp_re;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (nRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxInit_int32_T(&bitrevIndex, 1);
  z = (double)unsigned_nRows / 2.0;
  iDelta = y->size[0];
  if (iDelta >= nRows) {
    iDelta = nRows;
  }

  d_FFTImplementationCallback_get(iDelta - 1, (int)z, bitrevIndex);
  hszCostab = 0;
  if (8 < unsigned_nRows) {
    iDelta = 8;
  } else {
    iDelta = unsigned_nRows;
  }

  iDelta = (int)((double)iDelta / 2.0);
  for (i = 0; i < iDelta; i++) {
    y->data[bitrevIndex->data[i] - 1].re = dv1[hszCostab];
    y->data[bitrevIndex->data[i] - 1].im = dv1[hszCostab + 1];
    hszCostab += 2;
  }

  emxFree_int32_T(&bitrevIndex);
  if (nRows > 1) {
    for (i = 0; i <= istart; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }
  }

  iDelta = 2;
  hszCostab = 4;
  nRows = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < nRows; i += hszCostab) {
      temp_re_tmp = i + iDelta;
      temp_re = y->data[temp_re_tmp].re;
      temp_im = y->data[temp_re_tmp].im;
      y->data[temp_re_tmp].re = y->data[i].re - temp_re;
      y->data[temp_re_tmp].im = y->data[i].im - temp_im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    istart = 1;
    for (j = k; j < nRowsD2; j += k) {
      twid_re = hcostab->data[j];
      twid_im = hsintab->data[j];
      i = istart;
      ihi = istart + nRows;
      while (i < ihi) {
        temp_re_tmp = i + iDelta;
        temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
          data[temp_re_tmp].im;
        temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
          data[temp_re_tmp].re;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
        i += hszCostab;
      }

      istart++;
    }

    k /= 2;
    iDelta = hszCostab;
    hszCostab += hszCostab;
    nRows -= iDelta;
  }

  emxFree_real_T(&hsintab);
  emxFree_real_T(&hcostab);
  f_FFTImplementationCallback_get(y, reconVar1, reconVar2, wrapIndex, (int)z);
  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
}

static void ifft(const emxArray_creal_T *x, emxArray_creal_T *y)
{
  emxArray_real_T *costab1q;
  int len;
  boolean_T useRadix2;
  int N2blue;
  int nd2;
  double nt_im;
  int rt;
  int i;
  int k;
  emxArray_real_T *costab;
  emxArray_real_T *sintab;
  emxArray_real_T *sintabinv;
  emxArray_creal_T *yCol;
  emxArray_creal_T *wwc;
  emxArray_creal_T b_x;
  int c_x[1];
  int idx;
  int nInt2;
  int b_y;
  double nt_re;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  double re;
  emxInit_real_T(&costab1q, 2);
  len = x->size[1];
  useRadix2 = ((x->size[1] & (x->size[1] - 1)) == 0);
  c_FFTImplementationCallback_get(x->size[1], useRadix2, &N2blue, &nd2);
  nt_im = 6.2831853071795862 / (double)nd2;
  rt = nd2 / 2 / 2;
  i = costab1q->size[0] * costab1q->size[1];
  costab1q->size[0] = 1;
  costab1q->size[1] = rt + 1;
  emxEnsureCapacity_real_T(costab1q, i);
  costab1q->data[0] = 1.0;
  nd2 = rt / 2 - 1;
  for (k = 0; k <= nd2; k++) {
    costab1q->data[k + 1] = cos(nt_im * ((double)k + 1.0));
  }

  i = nd2 + 2;
  nd2 = rt - 1;
  for (k = i; k <= nd2; k++) {
    costab1q->data[k] = sin(nt_im * (double)(rt - k));
  }

  costab1q->data[rt] = 0.0;
  emxInit_real_T(&costab, 2);
  emxInit_real_T(&sintab, 2);
  emxInit_real_T(&sintabinv, 2);
  if (!useRadix2) {
    rt = costab1q->size[1] - 1;
    nd2 = (costab1q->size[1] - 1) << 1;
    i = costab->size[0] * costab->size[1];
    costab->size[0] = 1;
    costab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(costab, i);
    i = sintab->size[0] * sintab->size[1];
    sintab->size[0] = 1;
    sintab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintab, i);
    costab->data[0] = 1.0;
    sintab->data[0] = 0.0;
    i = sintabinv->size[0] * sintabinv->size[1];
    sintabinv->size[0] = 1;
    sintabinv->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintabinv, i);
    for (k = 0; k < rt; k++) {
      sintabinv->data[k + 1] = costab1q->data[(rt - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      sintabinv->data[k] = costab1q->data[k - rt];
    }

    for (k = 0; k < rt; k++) {
      costab->data[k + 1] = costab1q->data[k + 1];
      sintab->data[k + 1] = -costab1q->data[(rt - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      costab->data[k] = -costab1q->data[nd2 - k];
      sintab->data[k] = -costab1q->data[k - rt];
    }
  } else {
    rt = costab1q->size[1] - 1;
    nd2 = (costab1q->size[1] - 1) << 1;
    i = costab->size[0] * costab->size[1];
    costab->size[0] = 1;
    costab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(costab, i);
    i = sintab->size[0] * sintab->size[1];
    sintab->size[0] = 1;
    sintab->size[1] = nd2 + 1;
    emxEnsureCapacity_real_T(sintab, i);
    costab->data[0] = 1.0;
    sintab->data[0] = 0.0;
    for (k = 0; k < rt; k++) {
      costab->data[k + 1] = costab1q->data[k + 1];
      sintab->data[k + 1] = costab1q->data[(rt - k) - 1];
    }

    i = costab1q->size[1];
    for (k = i; k <= nd2; k++) {
      costab->data[k] = -costab1q->data[nd2 - k];
      sintab->data[k] = costab1q->data[k - rt];
    }

    sintabinv->size[0] = 1;
    sintabinv->size[1] = 0;
  }

  emxFree_real_T(&costab1q);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    nd2 = x->size[1];
    b_x = *x;
    c_x[0] = nd2;
    b_x.size = &c_x[0];
    b_x.numDimensions = 1;
    c_FFTImplementationCallback_r2b(&b_x, x->size[1], costab, sintab, yCol);
    if (yCol->size[0] > 1) {
      nt_im = 1.0 / (double)yCol->size[0];
      nd2 = yCol->size[0];
      for (i = 0; i < nd2; i++) {
        yCol->data[i].re *= nt_im;
        yCol->data[i].im *= nt_im;
      }
    }
  } else {
    emxInit_creal_T(&wwc, 1);
    nd2 = (x->size[1] + x->size[1]) - 1;
    i = wwc->size[0];
    wwc->size[0] = nd2;
    emxEnsureCapacity_creal_T(wwc, i);
    idx = x->size[1];
    rt = 0;
    wwc->data[x->size[1] - 1].re = 1.0;
    wwc->data[x->size[1] - 1].im = 0.0;
    nInt2 = x->size[1] << 1;
    i = x->size[1];
    for (k = 0; k <= i - 2; k++) {
      b_y = ((k + 1) << 1) - 1;
      if (nInt2 - rt <= b_y) {
        rt += b_y - nInt2;
      } else {
        rt += b_y;
      }

      nt_im = 3.1415926535897931 * (double)rt / (double)len;
      if (nt_im == 0.0) {
        nt_re = 1.0;
        nt_im = 0.0;
      } else {
        nt_re = cos(nt_im);
        nt_im = sin(nt_im);
      }

      wwc->data[idx - 2].re = nt_re;
      wwc->data[idx - 2].im = -nt_im;
      idx--;
    }

    idx = 0;
    i = nd2 - 1;
    for (k = i; k >= len; k--) {
      wwc->data[k] = wwc->data[idx];
      idx++;
    }

    i = yCol->size[0];
    yCol->size[0] = x->size[1];
    emxEnsureCapacity_creal_T(yCol, i);
    nd2 = x->size[1];
    rt = 0;
    for (k = 0; k < nd2; k++) {
      nInt2 = (len + k) - 1;
      nt_re = wwc->data[nInt2].re;
      nt_im = wwc->data[nInt2].im;
      yCol->data[k].re = nt_re * x->data[rt].re + nt_im * x->data[rt].im;
      yCol->data[k].im = nt_re * x->data[rt].im - nt_im * x->data[rt].re;
      rt++;
    }

    i = x->size[1] + 1;
    for (k = i; k <= len; k++) {
      yCol->data[k - 1].re = 0.0;
      yCol->data[k - 1].im = 0.0;
    }

    emxInit_creal_T(&fv, 1);
    emxInit_creal_T(&b_fv, 1);
    c_FFTImplementationCallback_r2b(yCol, N2blue, costab, sintab, fv);
    c_FFTImplementationCallback_r2b(wwc, N2blue, costab, sintab, b_fv);
    i = b_fv->size[0];
    b_fv->size[0] = fv->size[0];
    emxEnsureCapacity_creal_T(b_fv, i);
    nd2 = fv->size[0];
    for (i = 0; i < nd2; i++) {
      nt_im = fv->data[i].re * b_fv->data[i].im + fv->data[i].im * b_fv->data[i]
        .re;
      b_fv->data[i].re = fv->data[i].re * b_fv->data[i].re - fv->data[i].im *
        b_fv->data[i].im;
      b_fv->data[i].im = nt_im;
    }

    c_FFTImplementationCallback_r2b(b_fv, N2blue, costab, sintabinv, fv);
    emxFree_creal_T(&b_fv);
    if (fv->size[0] > 1) {
      nt_im = 1.0 / (double)fv->size[0];
      nd2 = fv->size[0];
      for (i = 0; i < nd2; i++) {
        fv->data[i].re *= nt_im;
        fv->data[i].im *= nt_im;
      }
    }

    idx = 0;
    nt_re = x->size[1];
    i = x->size[1];
    nd2 = wwc->size[0];
    for (k = i; k <= nd2; k++) {
      yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re + wwc->
        data[k - 1].im * fv->data[k - 1].im;
      yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im - wwc->
        data[k - 1].im * fv->data[k - 1].re;
      if (yCol->data[idx].im == 0.0) {
        re = yCol->data[idx].re / nt_re;
        nt_im = 0.0;
      } else if (yCol->data[idx].re == 0.0) {
        re = 0.0;
        nt_im = yCol->data[idx].im / nt_re;
      } else {
        re = yCol->data[idx].re / nt_re;
        nt_im = yCol->data[idx].im / nt_re;
      }

      yCol->data[idx].re = re;
      yCol->data[idx].im = nt_im;
      idx++;
    }

    emxFree_creal_T(&fv);
    emxFree_creal_T(&wwc);
  }

  emxFree_real_T(&sintabinv);
  emxFree_real_T(&sintab);
  emxFree_real_T(&costab);
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = x->size[1];
  emxEnsureCapacity_creal_T(y, i);
  nd2 = x->size[1];
  for (i = 0; i < nd2; i++) {
    y->data[i] = yCol->data[i];
  }

  emxFree_creal_T(&yCol);
}

static void imodwtrec(const emxArray_real_T *Vin, const emxArray_real_T *Win,
                      const emxArray_creal_T *G, const emxArray_creal_T *H, int
                      J, emxArray_real_T *Vout)
{
  emxArray_creal_T *Vhat;
  emxArray_creal_T *What;
  int N;
  int upfactor;
  int k;
  int idx;
  double G_im;
  emxInit_creal_T(&Vhat, 2);
  emxInit_creal_T(&What, 2);
  N = Vin->size[1];
  upfactor = 1 << (J - 1);
  b_fft(Vin, Vhat);
  b_fft(Win, What);
  for (k = 0; k < N; k++) {
    idx = upfactor * k;
    idx -= div_s32(idx, Vin->size[1]) * Vin->size[1];
    G_im = G->data[idx].re * Vhat->data[k].im + -G->data[idx].im * Vhat->data[k]
      .re;
    Vhat->data[k].re = (G->data[idx].re * Vhat->data[k].re - -G->data[idx].im *
                        Vhat->data[k].im) + (H->data[idx].re * What->data[k].re
      - -H->data[idx].im * What->data[k].im);
    Vhat->data[k].im = G_im + (H->data[idx].re * What->data[k].im + -H->data[idx]
      .im * What->data[k].re);
  }

  ifft(Vhat, What);
  N = Vout->size[0] * Vout->size[1];
  Vout->size[0] = 1;
  Vout->size[1] = What->size[1];
  emxEnsureCapacity_real_T(Vout, N);
  idx = What->size[0] * What->size[1];
  emxFree_creal_T(&Vhat);
  for (N = 0; N < idx; N++) {
    Vout->data[N] = What->data[N].re;
  }

  emxFree_creal_T(&What);
}

static void j_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv)
{
  emxArray_creal_T *ytmp;
  int hnRows;
  int ix;
  emxArray_real_T *unusedU0;
  emxArray_int32_T *wrapIndex;
  emxArray_real_T *costable;
  emxArray_real_T *sintable;
  emxArray_real_T *hsintab;
  emxArray_real_T *hcostabinv;
  emxArray_real_T *hsintabinv;
  emxArray_creal_T *reconVar1;
  emxArray_creal_T *reconVar2;
  int idx;
  int i;
  int xidx;
  int temp_re_tmp;
  double twid_re;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double temp_re;
  double temp_im;
  double twid_im;
  emxArray_creal_T *fv;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 8) {
    ix = ytmp->size[0];
    ytmp->size[0] = hnRows;
    emxEnsureCapacity_creal_T(ytmp, ix);
    for (ix = 0; ix < hnRows; ix++) {
      ytmp->data[ix].re = 0.0;
      ytmp->data[ix].im = 0.0;
    }
  }

  emxInit_real_T(&unusedU0, 2);
  emxInit_int32_T(&wrapIndex, 2);
  emxInit_real_T(&costable, 2);
  emxInit_real_T(&sintable, 2);
  emxInit_real_T(&hsintab, 2);
  emxInit_real_T(&hcostabinv, 2);
  emxInit_real_T(&hsintabinv, 2);
  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  d_FFTImplementationCallback_gen(nRows << 1, costable, sintable, unusedU0);
  e_FFTImplementationCallback_get(costab, sintab, costabinv, sintabinv, unusedU0,
    hsintab, hcostabinv, hsintabinv);
  ix = reconVar1->size[0];
  reconVar1->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar1, ix);
  ix = reconVar2->size[0];
  reconVar2->size[0] = hnRows;
  emxEnsureCapacity_creal_T(reconVar2, ix);
  idx = 0;
  ix = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = hnRows;
  emxEnsureCapacity_int32_T(wrapIndex, ix);
  for (i = 0; i < hnRows; i++) {
    reconVar1->data[i].re = sintable->data[idx] + 1.0;
    reconVar1->data[i].im = -costable->data[idx];
    reconVar2->data[i].re = 1.0 - sintable->data[idx];
    reconVar2->data[i].im = costable->data[idx];
    idx += 2;
    if (i + 1 != 1) {
      wrapIndex->data[i] = (hnRows - i) + 1;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxFree_real_T(&sintable);
  emxFree_real_T(&costable);
  xidx = 1;
  if (8 < nRows) {
    idx = 8;
  } else {
    idx = nRows;
  }

  ix = (int)((double)idx / 2.0);
  for (idx = 0; idx < ix; idx++) {
    temp_re_tmp = (hnRows + idx) - 1;
    twid_re = dv1[xidx - 1];
    ytmp->data[idx].re = wwc->data[temp_re_tmp].re * twid_re + wwc->
      data[temp_re_tmp].im * dv1[xidx];
    ytmp->data[idx].im = wwc->data[temp_re_tmp].re * dv1[xidx] - wwc->
      data[temp_re_tmp].im * twid_re;
    xidx += 2;
  }

  ix++;
  if (ix <= hnRows) {
    for (i = ix; i <= hnRows; i++) {
      ytmp->data[i - 1].re = 0.0;
      ytmp->data[i - 1].im = 0.0;
    }
  }

  emxInit_creal_T(&fy, 1);
  loop_ub_tmp = (int)((double)nfft / 2.0);
  ix = fy->size[0];
  fy->size[0] = loop_ub_tmp;
  emxEnsureCapacity_creal_T(fy, ix);
  if (loop_ub_tmp > ytmp->size[0]) {
    ix = fy->size[0];
    fy->size[0] = loop_ub_tmp;
    emxEnsureCapacity_creal_T(fy, ix);
    for (ix = 0; ix < loop_ub_tmp; ix++) {
      fy->data[ix].re = 0.0;
      fy->data[ix].im = 0.0;
    }
  }

  xidx = ytmp->size[0];
  if (xidx >= loop_ub_tmp) {
    xidx = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= xidx - 2; i++) {
    fy->data[idx] = ytmp->data[ix];
    idx = loop_ub_tmp;
    tst = true;
    while (tst) {
      idx >>= 1;
      ju ^= idx;
      tst = ((ju & idx) == 0);
    }

    idx = ju;
    ix++;
  }

  fy->data[idx] = ytmp->data[ix];
  if (loop_ub_tmp > 1) {
    for (i = 0; i <= iheight; i += 2) {
      temp_re = fy->data[i + 1].re;
      temp_im = fy->data[i + 1].im;
      twid_re = fy->data[i].re;
      twid_im = fy->data[i].im;
      fy->data[i + 1].re = fy->data[i].re - fy->data[i + 1].re;
      fy->data[i + 1].im = fy->data[i].im - fy->data[i + 1].im;
      twid_re += temp_re;
      twid_im += temp_im;
      fy->data[i].re = twid_re;
      fy->data[i].im = twid_im;
    }
  }

  idx = 2;
  xidx = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += xidx) {
      temp_re_tmp = i + idx;
      temp_re = fy->data[temp_re_tmp].re;
      temp_im = fy->data[temp_re_tmp].im;
      fy->data[temp_re_tmp].re = fy->data[i].re - fy->data[temp_re_tmp].re;
      fy->data[temp_re_tmp].im = fy->data[i].im - fy->data[temp_re_tmp].im;
      fy->data[i].re += temp_re;
      fy->data[i].im += temp_im;
    }

    ix = 1;
    for (ju = k; ju < nRowsD2; ju += k) {
      twid_re = unusedU0->data[ju];
      twid_im = hsintab->data[ju];
      i = ix;
      ihi = ix + iheight;
      while (i < ihi) {
        temp_re_tmp = i + idx;
        temp_re = twid_re * fy->data[temp_re_tmp].re - twid_im * fy->
          data[temp_re_tmp].im;
        temp_im = twid_re * fy->data[temp_re_tmp].im + twid_im * fy->
          data[temp_re_tmp].re;
        fy->data[temp_re_tmp].re = fy->data[i].re - temp_re;
        fy->data[temp_re_tmp].im = fy->data[i].im - temp_im;
        fy->data[i].re += temp_re;
        fy->data[i].im += temp_im;
        i += xidx;
      }

      ix++;
    }

    k /= 2;
    idx = xidx;
    xidx += xidx;
    iheight -= idx;
  }

  emxInit_creal_T(&fv, 1);
  c_FFTImplementationCallback_r2b(wwc, loop_ub_tmp, unusedU0, hsintab, fv);
  idx = fy->size[0];
  emxFree_real_T(&hsintab);
  emxFree_real_T(&unusedU0);
  for (ix = 0; ix < idx; ix++) {
    twid_im = fy->data[ix].re * fv->data[ix].im + fy->data[ix].im * fv->data[ix]
      .re;
    fy->data[ix].re = fy->data[ix].re * fv->data[ix].re - fy->data[ix].im *
      fv->data[ix].im;
    fy->data[ix].im = twid_im;
  }

  c_FFTImplementationCallback_r2b(fy, loop_ub_tmp, hcostabinv, hsintabinv, fv);
  emxFree_creal_T(&fy);
  emxFree_real_T(&hsintabinv);
  emxFree_real_T(&hcostabinv);
  if (fv->size[0] > 1) {
    twid_re = 1.0 / (double)fv->size[0];
    idx = fv->size[0];
    for (ix = 0; ix < idx; ix++) {
      fv->data[ix].re *= twid_re;
      fv->data[ix].im *= twid_re;
    }
  }

  idx = 0;
  ix = wwc->size[0];
  for (k = hnRows; k <= ix; k++) {
    ytmp->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re + wwc->data[k
      - 1].im * fv->data[k - 1].im;
    ytmp->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im - wwc->data[k
      - 1].im * fv->data[k - 1].re;
    idx++;
  }

  emxFree_creal_T(&fv);
  for (i = 0; i < hnRows; i++) {
    ix = wrapIndex->data[i];
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    y->data[i].re = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].re -
      ytmp->data[i].im * reconVar1->data[i].im) + (twid_re * reconVar2->data[i].
      re - twid_im * reconVar2->data[i].im));
    y->data[i].im = 0.5 * ((ytmp->data[i].re * reconVar1->data[i].im +
      ytmp->data[i].im * reconVar1->data[i].re) + (twid_re * reconVar2->data[i].
      im + twid_im * reconVar2->data[i].re));
    twid_re = ytmp->data[ix - 1].re;
    twid_im = -ytmp->data[ix - 1].im;
    ix = hnRows + i;
    y->data[ix].re = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].re -
      ytmp->data[i].im * reconVar2->data[i].im) + (twid_re * reconVar1->data[i].
      re - twid_im * reconVar1->data[i].im));
    y->data[ix].im = 0.5 * ((ytmp->data[i].re * reconVar2->data[i].im +
      ytmp->data[i].im * reconVar2->data[i].re) + (twid_re * reconVar1->data[i].
      im + twid_im * reconVar1->data[i].re));
  }

  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_int32_T(&wrapIndex);
  emxFree_creal_T(&ytmp);
}

static void k_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int nChan, const emxArray_real_T *costab, const
  emxArray_real_T *sintab)
{
  emxArray_real_T *hcostab;
  emxArray_real_T *hsintab;
  int iDelta2;
  int i;
  emxArray_creal_T *reconVar1;
  int ix;
  emxArray_creal_T *reconVar2;
  emxArray_int32_T *wrapIndex;
  emxArray_int32_T *bitrevIndex;
  emxArray_int32_T *r;
  boolean_T nxeven;
  int u0;
  int chan;
  double temp_re;
  double temp_im;
  int k;
  int iheight;
  int istart;
  int temp_re_tmp;
  int j;
  double twid_re;
  double twid_im;
  int ihi;
  emxInit_real_T(&hcostab, 2);
  emxInit_real_T(&hsintab, 2);
  iDelta2 = hcostab->size[0] * hcostab->size[1];
  hcostab->size[0] = 1;
  hcostab->size[1] = 16384;
  emxEnsureCapacity_real_T(hcostab, iDelta2);
  iDelta2 = hsintab->size[0] * hsintab->size[1];
  hsintab->size[0] = 1;
  hsintab->size[1] = 16384;
  emxEnsureCapacity_real_T(hsintab, iDelta2);
  for (i = 0; i < 16384; i++) {
    ix = ((i + 1) << 1) - 2;
    hcostab->data[i] = costab->data[ix];
    hsintab->data[i] = sintab->data[ix];
  }

  emxInit_creal_T(&reconVar1, 1);
  emxInit_creal_T(&reconVar2, 1);
  emxInit_int32_T(&wrapIndex, 2);
  iDelta2 = reconVar1->size[0];
  reconVar1->size[0] = 32768;
  emxEnsureCapacity_creal_T(reconVar1, iDelta2);
  iDelta2 = reconVar2->size[0];
  reconVar2->size[0] = 32768;
  emxEnsureCapacity_creal_T(reconVar2, iDelta2);
  iDelta2 = wrapIndex->size[0] * wrapIndex->size[1];
  wrapIndex->size[0] = 1;
  wrapIndex->size[1] = 32768;
  emxEnsureCapacity_int32_T(wrapIndex, iDelta2);
  for (i = 0; i < 32768; i++) {
    reconVar1->data[i].re = sintab->data[i] + 1.0;
    reconVar1->data[i].im = -costab->data[i];
    reconVar2->data[i].re = 1.0 - sintab->data[i];
    reconVar2->data[i].im = costab->data[i];
    if (i + 1 != 1) {
      wrapIndex->data[i] = 32769 - i;
    } else {
      wrapIndex->data[0] = 1;
    }
  }

  emxInit_int32_T(&bitrevIndex, 1);
  emxInit_int32_T(&r, 1);
  d_FFTImplementationCallback_get(32767, 32768, r);
  iDelta2 = bitrevIndex->size[0];
  bitrevIndex->size[0] = r->size[0];
  emxEnsureCapacity_int32_T(bitrevIndex, iDelta2);
  ix = r->size[0];
  for (iDelta2 = 0; iDelta2 < ix; iDelta2++) {
    bitrevIndex->data[iDelta2] = r->data[iDelta2];
  }

  emxFree_int32_T(&r);
  if ((x->size[0] & 1) == 0) {
    nxeven = true;
    u0 = x->size[0];
  } else if (x->size[0] >= 65536) {
    nxeven = true;
    u0 = 65536;
  } else {
    nxeven = false;
    u0 = x->size[0] - 1;
  }

  if (u0 >= 65536) {
    u0 = 65536;
  }

  for (chan = 0; chan < nChan; chan++) {
    ix = 0;
    iDelta2 = (int)((double)u0 / 2.0);
    for (i = 0; i < iDelta2; i++) {
      y->data[bitrevIndex->data[i] - 1].re = x->data[ix];
      y->data[bitrevIndex->data[i] - 1].im = x->data[ix + 1];
      ix += 2;
    }

    if (!nxeven) {
      iDelta2 = bitrevIndex->data[iDelta2] - 1;
      y->data[iDelta2].re = x->data[ix];
      y->data[iDelta2].im = 0.0;
    }

    for (i = 0; i <= 32766; i += 2) {
      temp_re = y->data[i + 1].re;
      temp_im = y->data[i + 1].im;
      y->data[i + 1].re = y->data[i].re - y->data[i + 1].re;
      y->data[i + 1].im = y->data[i].im - y->data[i + 1].im;
      y->data[i].re += temp_re;
      y->data[i].im += temp_im;
    }

    ix = 2;
    iDelta2 = 4;
    k = 8192;
    iheight = 32765;
    while (k > 0) {
      for (i = 0; i < iheight; i += iDelta2) {
        temp_re_tmp = i + ix;
        temp_re = y->data[temp_re_tmp].re;
        temp_im = y->data[temp_re_tmp].im;
        y->data[temp_re_tmp].re = y->data[i].re - temp_re;
        y->data[temp_re_tmp].im = y->data[i].im - temp_im;
        y->data[i].re += temp_re;
        y->data[i].im += temp_im;
      }

      istart = 1;
      for (j = k; j < 16384; j += k) {
        twid_re = hcostab->data[j];
        twid_im = hsintab->data[j];
        i = istart;
        ihi = istart + iheight;
        while (i < ihi) {
          temp_re_tmp = i + ix;
          temp_re = twid_re * y->data[temp_re_tmp].re - twid_im * y->
            data[temp_re_tmp].im;
          temp_im = twid_re * y->data[temp_re_tmp].im + twid_im * y->
            data[temp_re_tmp].re;
          y->data[temp_re_tmp].re = y->data[i].re - temp_re;
          y->data[temp_re_tmp].im = y->data[i].im - temp_im;
          y->data[i].re += temp_re;
          y->data[i].im += temp_im;
          i += iDelta2;
        }

        istart++;
      }

      k /= 2;
      ix = iDelta2;
      iDelta2 += iDelta2;
      iheight -= ix;
    }

    g_FFTImplementationCallback_get(y, reconVar1, reconVar2, wrapIndex);
  }

  emxFree_int32_T(&wrapIndex);
  emxFree_creal_T(&reconVar2);
  emxFree_creal_T(&reconVar1);
  emxFree_real_T(&hsintab);
  emxFree_real_T(&hcostab);
  emxFree_int32_T(&bitrevIndex);
}

static void localComputeSpectra(const emxArray_real_T *Sxx, const
  emxArray_real_T *x, const emxArray_real_T *xStart, const emxArray_real_T *xEnd,
  const emxArray_real_T *win, const char options_range[8], double k,
  emxArray_real_T *Pxx, emxArray_real_T *w)
{
  emxArray_real_T *Sxx1;
  int i;
  emxArray_real_T *Sxxk;
  emxArray_real_T *b_w;
  emxArray_real_T *b_x;
  int ii;
  double d;
  double d1;
  int loop_ub;
  int i1;
  int i2;
  emxArray_real_T c_w;
  int iv[2];
  int b_loop_ub;
  int i3;
  emxInit_real_T(&Sxx1, 2);
  Sxx1->size[0] = 0;
  Sxx1->size[1] = 0;
  i = (int)k;
  emxInit_real_T(&Sxxk, 2);
  emxInit_real_T(&b_w, 1);
  emxInit_real_T(&b_x, 2);
  for (ii = 0; ii < i; ii++) {
    d = xStart->data[ii];
    d1 = xEnd->data[ii];
    if (d > d1) {
      i1 = 0;
      i2 = 0;
    } else {
      i1 = (int)d - 1;
      i2 = (int)d1;
    }

    loop_ub = x->size[1];
    b_loop_ub = i2 - i1;
    i2 = b_x->size[0] * b_x->size[1];
    b_x->size[0] = b_loop_ub;
    b_x->size[1] = x->size[1];
    emxEnsureCapacity_real_T(b_x, i2);
    for (i2 = 0; i2 < loop_ub; i2++) {
      for (i3 = 0; i3 < b_loop_ub; i3++) {
        b_x->data[i3 + b_x->size[0] * i2] = x->data[(i1 + i3) + x->size[0] * i2];
      }
    }

    computeperiodogram(b_x, win, Sxxk, b_w);
    if (ii + 1U == 1U) {
      i1 = Sxx1->size[0] * Sxx1->size[1];
      Sxx1->size[0] = 65536;
      Sxx1->size[1] = Sxx->size[1];
      emxEnsureCapacity_real_T(Sxx1, i1);
      loop_ub = Sxx->size[0] * Sxx->size[1];
      for (i1 = 0; i1 < loop_ub; i1++) {
        Sxx1->data[i1] = Sxx->data[i1] + Sxxk->data[i1];
      }
    } else {
      loop_ub = Sxx1->size[0] * Sxx1->size[1];
      for (i1 = 0; i1 < loop_ub; i1++) {
        Sxx1->data[i1] += Sxxk->data[i1];
      }
    }
  }

  emxFree_real_T(&b_x);
  emxFree_real_T(&Sxxk);
  loop_ub = Sxx1->size[0] * Sxx1->size[1];
  for (i = 0; i < loop_ub; i++) {
    Sxx1->data[i] /= k;
  }

  psdfreqvec(b_w);
  c_w = *b_w;
  iv[0] = 65536;
  iv[1] = 1;
  c_w.size = &iv[0];
  c_w.numDimensions = 2;
  computepsd(Sxx1, &c_w, options_range, Pxx, w);
  emxFree_real_T(&b_w);
  emxFree_real_T(&Sxx1);
}

static void modwtmra(const emxArray_real_T *w, emxArray_real_T *mra)
{
  int ncw;
  int cfslength;
  int ncopies;
  emxArray_real_T *nullinput;
  int i;
  emxArray_real_T *ww;
  int loop_ub;
  int i1;
  int k;
  emxArray_real_T *sintabinv;
  int offset;
  emxArray_real_T *v;
  emxArray_real_T *b_w;
  boolean_T useRadix2;
  int N2blue;
  int nRows;
  emxArray_creal_T *yCol;
  double ww_data[10];
  emxArray_creal_T *wwc;
  int nInt2m1;
  int idx;
  emxArray_creal_T *H;
  double nt_im;
  double nt_re;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  emxArray_creal_T b_yCol;
  int iv[2];
  int iv1[2];
  ncw = w->size[1];
  cfslength = w->size[1];
  if (w->size[1] < 8) {
    ncopies = 8 - w->size[1];
    cfslength = (9 - w->size[1]) * w->size[1];
  } else {
    ncopies = 0;
  }

  emxInit_real_T(&nullinput, 2);
  i = nullinput->size[0] * nullinput->size[1];
  nullinput->size[0] = 1;
  nullinput->size[1] = cfslength;
  emxEnsureCapacity_real_T(nullinput, i);
  for (i = 0; i < cfslength; i++) {
    nullinput->data[i] = 0.0;
  }

  emxInit_real_T(&ww, 2);
  i = ww->size[0] * ww->size[1];
  ww->size[0] = 10;
  ww->size[1] = cfslength;
  emxEnsureCapacity_real_T(ww, i);
  loop_ub = w->size[1];
  for (i = 0; i < loop_ub; i++) {
    for (i1 = 0; i1 < 10; i1++) {
      offset = i1 + 10 * i;
      ww->data[offset] = w->data[offset];
    }
  }

  if (ncopies > 0) {
    for (k = 0; k < ncopies; k++) {
      offset = (k + 1) * ncw;
      for (loop_ub = 0; loop_ub < ncw; loop_ub++) {
        i = offset + loop_ub;
        for (i1 = 0; i1 < 10; i1++) {
          ww_data[i1] = ww->data[i1 + 10 * loop_ub];
        }

        for (i1 = 0; i1 < 10; i1++) {
          ww->data[i1 + 10 * i] = ww_data[i1];
        }
      }
    }
  }

  emxInit_real_T(&sintabinv, 2);
  emxInit_real_T(&v, 2);
  emxInit_real_T(&b_w, 2);
  useRadix2 = ((cfslength & (cfslength - 1)) == 0);
  c_FFTImplementationCallback_get(cfslength, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, b_w, v, sintabinv);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = cfslength;
    emxEnsureCapacity_creal_T(yCol, i);
    if (cfslength > 8) {
      i = yCol->size[0];
      yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(yCol, i);
      for (i = 0; i < cfslength; i++) {
        yCol->data[i].re = 0.0;
        yCol->data[i].im = 0.0;
      }
    }

    g_FFTImplementationCallback_doH(yCol, cfslength, b_w, v);
  } else {
    i = cfslength & 1;
    emxInit_creal_T(&wwc, 1);
    if (i == 0) {
      nRows = cfslength / 2;
      nInt2m1 = (nRows + nRows) - 1;
      i1 = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, i1);
      idx = nRows;
      offset = 0;
      wwc->data[nRows - 1].re = 1.0;
      wwc->data[nRows - 1].im = 0.0;
      ncopies = nRows << 1;
      for (k = 0; k <= nRows - 2; k++) {
        loop_ub = ((k + 1) << 1) - 1;
        if (ncopies - offset <= loop_ub) {
          offset += loop_ub - ncopies;
        } else {
          offset += loop_ub;
        }

        nt_im = -3.1415926535897931 * (double)offset / (double)nRows;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      i1 = nInt2m1 - 1;
      for (k = i1; k >= nRows; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    } else {
      nInt2m1 = (cfslength + cfslength) - 1;
      i1 = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, i1);
      idx = cfslength;
      offset = 0;
      wwc->data[cfslength - 1].re = 1.0;
      wwc->data[cfslength - 1].im = 0.0;
      ncopies = cfslength << 1;
      for (k = 0; k <= cfslength - 2; k++) {
        loop_ub = ((k + 1) << 1) - 1;
        if (ncopies - offset <= loop_ub) {
          offset += loop_ub - ncopies;
        } else {
          offset += loop_ub;
        }

        nt_im = -3.1415926535897931 * (double)offset / (double)cfslength;
        if (nt_im == 0.0) {
          nt_re = 1.0;
          nt_im = 0.0;
        } else {
          nt_re = cos(nt_im);
          nt_im = sin(nt_im);
        }

        wwc->data[idx - 2].re = nt_re;
        wwc->data[idx - 2].im = -nt_im;
        idx--;
      }

      idx = 0;
      i1 = nInt2m1 - 1;
      for (k = i1; k >= cfslength; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    }

    i1 = yCol->size[0];
    yCol->size[0] = cfslength;
    emxEnsureCapacity_creal_T(yCol, i1);
    if (cfslength > 8) {
      i1 = yCol->size[0];
      yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(yCol, i1);
      for (i1 = 0; i1 < cfslength; i1++) {
        yCol->data[i1].re = 0.0;
        yCol->data[i1].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      h_FFTImplementationCallback_doH(yCol, cfslength, N2blue, wwc, b_w, v, b_w,
        sintabinv);
    } else {
      if (cfslength < 8) {
        offset = cfslength - 1;
      } else {
        offset = 7;
      }

      ncopies = 0;
      for (k = 0; k <= offset; k++) {
        loop_ub = (cfslength + k) - 1;
        yCol->data[k].re = wwc->data[loop_ub].re * dv[ncopies];
        yCol->data[k].im = wwc->data[loop_ub].im * -dv[ncopies];
        ncopies++;
      }

      i = offset + 2;
      for (k = i; k <= cfslength; k++) {
        yCol->data[k - 1].re = 0.0;
        yCol->data[k - 1].im = 0.0;
      }

      emxInit_creal_T(&fv, 1);
      emxInit_creal_T(&b_fv, 1);
      c_FFTImplementationCallback_r2b(yCol, N2blue, b_w, v, fv);
      c_FFTImplementationCallback_r2b(wwc, N2blue, b_w, v, b_fv);
      i = b_fv->size[0];
      b_fv->size[0] = fv->size[0];
      emxEnsureCapacity_creal_T(b_fv, i);
      loop_ub = fv->size[0];
      for (i = 0; i < loop_ub; i++) {
        nt_re = fv->data[i].re * b_fv->data[i].im + fv->data[i].im * b_fv->
          data[i].re;
        b_fv->data[i].re = fv->data[i].re * b_fv->data[i].re - fv->data[i].im *
          b_fv->data[i].im;
        b_fv->data[i].im = nt_re;
      }

      c_FFTImplementationCallback_r2b(b_fv, N2blue, b_w, sintabinv, fv);
      emxFree_creal_T(&b_fv);
      if (fv->size[0] > 1) {
        nt_re = 1.0 / (double)fv->size[0];
        loop_ub = fv->size[0];
        for (i = 0; i < loop_ub; i++) {
          fv->data[i].re *= nt_re;
          fv->data[i].im *= nt_re;
        }
      }

      idx = 0;
      i = wwc->size[0];
      for (k = cfslength; k <= i; k++) {
        yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re +
          wwc->data[k - 1].im * fv->data[k - 1].im;
        yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im -
          wwc->data[k - 1].im * fv->data[k - 1].re;
        idx++;
      }

      emxFree_creal_T(&fv);
    }

    emxFree_creal_T(&wwc);
  }

  emxInit_creal_T(&H, 2);
  c_fft(cfslength, H);
  i = mra->size[0] * mra->size[1];
  mra->size[0] = 10;
  mra->size[1] = w->size[1];
  emxEnsureCapacity_real_T(mra, i);
  loop_ub = 10 * w->size[1];
  for (i = 0; i < loop_ub; i++) {
    mra->data[i] = 0.0;
  }

  loop_ub = ww->size[1];
  for (ncopies = 8; ncopies >= 0; ncopies--) {
    i = b_w->size[0] * b_w->size[1];
    b_w->size[0] = 1;
    b_w->size[1] = loop_ub;
    emxEnsureCapacity_real_T(b_w, i);
    for (i = 0; i < loop_ub; i++) {
      b_w->data[i] = ww->data[ncopies + 10 * i];
    }

    i = v->size[0] * v->size[1];
    v->size[0] = 1;
    v->size[1] = cfslength;
    emxEnsureCapacity_real_T(v, i);
    for (i = 0; i < cfslength; i++) {
      v->data[i] = 0.0;
    }

    for (nRows = ncopies + 1; nRows >= 1; nRows--) {
      i = sintabinv->size[0] * sintabinv->size[1];
      sintabinv->size[0] = 1;
      sintabinv->size[1] = v->size[1];
      emxEnsureCapacity_real_T(sintabinv, i);
      offset = v->size[0] * v->size[1] - 1;
      for (i = 0; i <= offset; i++) {
        sintabinv->data[i] = v->data[i];
      }

      b_yCol = *yCol;
      iv1[0] = 1;
      iv1[1] = cfslength;
      b_yCol.size = &iv1[0];
      b_yCol.numDimensions = 2;
      imodwtrec(sintabinv, b_w, &b_yCol, H, nRows, v);
      i = b_w->size[0] * b_w->size[1];
      b_w->size[0] = 1;
      b_w->size[1] = cfslength;
      emxEnsureCapacity_real_T(b_w, i);
      for (i = 0; i < cfslength; i++) {
        b_w->data[i] = 0.0;
      }
    }

    for (i = 0; i < ncw; i++) {
      mra->data[ncopies + 10 * i] = v->data[i];
    }
  }

  emxFree_real_T(&b_w);
  loop_ub = ww->size[1];
  i = v->size[0] * v->size[1];
  v->size[0] = 1;
  v->size[1] = ww->size[1];
  emxEnsureCapacity_real_T(v, i);
  for (i = 0; i < loop_ub; i++) {
    v->data[i] = ww->data[10 * i + 9];
  }

  emxFree_real_T(&ww);
  for (ncopies = 8; ncopies >= 0; ncopies--) {
    i = sintabinv->size[0] * sintabinv->size[1];
    sintabinv->size[0] = 1;
    sintabinv->size[1] = v->size[1];
    emxEnsureCapacity_real_T(sintabinv, i);
    loop_ub = v->size[0] * v->size[1] - 1;
    for (i = 0; i <= loop_ub; i++) {
      sintabinv->data[i] = v->data[i];
    }

    b_yCol = *yCol;
    iv[0] = 1;
    iv[1] = cfslength;
    b_yCol.size = &iv[0];
    b_yCol.numDimensions = 2;
    imodwtrec(sintabinv, nullinput, &b_yCol, H, ncopies + 1, v);
  }

  emxFree_real_T(&sintabinv);
  emxFree_creal_T(&yCol);
  emxFree_creal_T(&H);
  emxFree_real_T(&nullinput);
  loop_ub = w->size[1];
  for (i = 0; i < loop_ub; i++) {
    mra->data[10 * i + 9] = v->data[i];
  }

  emxFree_real_T(&v);
}

static void psdfreqvec(emxArray_real_T *w)
{
  emxArray_real_T *w1;
  int i;
  emxInit_real_T(&w1, 2);
  i = w1->size[0] * w1->size[1];
  w1->size[0] = 1;
  w1->size[1] = 65536;
  emxEnsureCapacity_real_T(w1, i);
  for (i = 0; i < 65536; i++) {
    w1->data[i] = 0.000762939453125 * (double)i;
  }

  w1->data[32768] = 25.0;
  w1->data[65535] = 49.999237060546875;
  i = w->size[0];
  w->size[0] = 65536;
  emxEnsureCapacity_real_T(w, i);
  for (i = 0; i < 65536; i++) {
    w->data[i] = w1->data[i];
  }

  emxFree_real_T(&w1);
}

static void pwelch(const emxArray_real_T *x, emxArray_real_T *varargout_1,
                   emxArray_real_T *varargout_2)
{
  emxArray_real_T *x1;
  int i;
  int loop_ub;
  int L;
  int noverlap;
  emxArray_real_T *y;
  emxArray_real_T *b_y;
  int nx;
  int k;
  int i1;
  int eint;
  double k1;
  double b;
  emxArray_real_T *xStart;
  emxArray_real_T *xEnd;
  emxArray_real_T *r;
  char method_Value_data[4];
  static const char options_range[8] = { 'o', 'n', 'e', 's', 'i', 'd', 'e', 'd'
  };

  boolean_T b_bool;
  int exitg1;
  static const char cv[4] = { 'p', 'l', 'u', 's' };

  emxArray_real_T *avgPxx;
  emxInit_real_T(&x1, 2);
  if ((x->size[0] == 1) || (x->size[1] == 1)) {
    i = x1->size[0] * x1->size[1];
    x1->size[0] = x->size[0] * x->size[1];
    x1->size[1] = 1;
    emxEnsureCapacity_real_T(x1, i);
    loop_ub = x->size[0] * x->size[1];
    for (i = 0; i < loop_ub; i++) {
      x1->data[i] = x->data[i];
    }
  } else {
    i = x1->size[0] * x1->size[1];
    x1->size[0] = x->size[0];
    x1->size[1] = x->size[1];
    emxEnsureCapacity_real_T(x1, i);
    loop_ub = x->size[0] * x->size[1];
    for (i = 0; i < loop_ub; i++) {
      x1->data[i] = x->data[i];
    }
  }

  L = (int)trunc((double)x1->size[0] / 4.5);
  noverlap = (int)trunc(0.5 * (double)L);
  emxInit_real_T(&y, 1);
  emxInit_real_T(&b_y, 1);
  if (fmod(L, 2.0) == 0.0) {
    loop_ub = (int)floor((double)L / 2.0 - 1.0);
    i = y->size[0];
    y->size[0] = loop_ub + 1;
    emxEnsureCapacity_real_T(y, i);
    for (i = 0; i <= loop_ub; i++) {
      y->data[i] = 6.2831853071795862 * ((double)i / ((double)L - 1.0));
    }

    nx = y->size[0];
    for (k = 0; k < nx; k++) {
      y->data[k] = cos(y->data[k]);
    }

    loop_ub = y->size[0];
    for (i = 0; i < loop_ub; i++) {
      y->data[i] = 0.54 - 0.46 * y->data[i];
    }

    i = b_y->size[0];
    b_y->size[0] = (y->size[0] + div_s32_floor(1 - y->size[0], -1)) + 1;
    emxEnsureCapacity_real_T(b_y, i);
    loop_ub = y->size[0];
    for (i = 0; i < loop_ub; i++) {
      b_y->data[i] = y->data[i];
    }

    loop_ub = div_s32_floor(1 - y->size[0], -1);
    for (i = 0; i <= loop_ub; i++) {
      b_y->data[i + y->size[0]] = y->data[(y->size[0] - i) - 1];
    }

    i = y->size[0];
    y->size[0] = b_y->size[0];
    emxEnsureCapacity_real_T(y, i);
    loop_ub = b_y->size[0];
    for (i = 0; i < loop_ub; i++) {
      y->data[i] = b_y->data[i];
    }
  } else {
    loop_ub = (int)floor(((double)L + 1.0) / 2.0 - 1.0);
    i = y->size[0];
    y->size[0] = loop_ub + 1;
    emxEnsureCapacity_real_T(y, i);
    for (i = 0; i <= loop_ub; i++) {
      y->data[i] = 6.2831853071795862 * ((double)i / ((double)L - 1.0));
    }

    nx = y->size[0];
    for (k = 0; k < nx; k++) {
      y->data[k] = cos(y->data[k]);
    }

    loop_ub = y->size[0];
    for (i = 0; i < loop_ub; i++) {
      y->data[i] = 0.54 - 0.46 * y->data[i];
    }

    if (1 > y->size[0] - 1) {
      i = 0;
      nx = 1;
      i1 = -1;
    } else {
      i = y->size[0] - 2;
      nx = -1;
      i1 = 0;
    }

    loop_ub = div_s32_floor(i1 - i, nx);
    i1 = b_y->size[0];
    b_y->size[0] = (y->size[0] + loop_ub) + 1;
    emxEnsureCapacity_real_T(b_y, i1);
    k = y->size[0];
    for (i1 = 0; i1 < k; i1++) {
      b_y->data[i1] = y->data[i1];
    }

    for (i1 = 0; i1 <= loop_ub; i1++) {
      b_y->data[i1 + y->size[0]] = y->data[i + nx * i1];
    }

    i = y->size[0];
    y->size[0] = b_y->size[0];
    emxEnsureCapacity_real_T(y, i);
    loop_ub = b_y->size[0];
    for (i = 0; i < loop_ub; i++) {
      y->data[i] = b_y->data[i];
    }
  }

  emxFree_real_T(&b_y);
  frexp(L, &eint);
  nx = L - noverlap;
  k1 = trunc((double)(x1->size[0] - noverlap) / (double)nx);
  b = k1 * (double)nx;
  emxInit_real_T(&xStart, 2);
  if ((nx == 0) || ((1.0 < b) && (nx < 0)) || ((b < 1.0) && (nx > 0))) {
    xStart->size[0] = 1;
    xStart->size[1] = 0;
  } else {
    i = xStart->size[0] * xStart->size[1];
    xStart->size[0] = 1;
    loop_ub = (int)floor((b - 1.0) / (double)nx);
    xStart->size[1] = loop_ub + 1;
    emxEnsureCapacity_real_T(xStart, i);
    for (i = 0; i <= loop_ub; i++) {
      xStart->data[i] = (double)nx * (double)i + 1.0;
    }
  }

  emxInit_real_T(&xEnd, 2);
  i = xEnd->size[0] * xEnd->size[1];
  xEnd->size[0] = 1;
  xEnd->size[1] = xStart->size[1];
  emxEnsureCapacity_real_T(xEnd, i);
  loop_ub = xStart->size[0] * xStart->size[1];
  for (i = 0; i < loop_ub; i++) {
    xEnd->data[i] = (xStart->data[i] + (double)L) - 1.0;
  }

  emxInit_real_T(&r, 2);
  method_Value_data[0] = 'p';
  method_Value_data[1] = 'l';
  method_Value_data[2] = 'u';
  method_Value_data[3] = 's';
  i = r->size[0] * r->size[1];
  r->size[0] = 65536;
  r->size[1] = x1->size[1];
  emxEnsureCapacity_real_T(r, i);
  loop_ub = x1->size[1] << 16;
  for (i = 0; i < loop_ub; i++) {
    r->data[i] = 0.0;
  }

  localComputeSpectra(r, x1, xStart, xEnd, y, options_range, k1, varargout_1,
                      varargout_2);
  b_bool = false;
  nx = 0;
  do {
    exitg1 = 0;
    if (nx < 4) {
      if (method_Value_data[nx] != cv[nx]) {
        exitg1 = 1;
      } else {
        nx++;
      }
    } else {
      b_bool = true;
      exitg1 = 1;
    }
  } while (exitg1 == 0);

  if (!b_bool) {
    i = r->size[0] * r->size[1];
    r->size[0] = 65536;
    r->size[1] = x1->size[1];
    emxEnsureCapacity_real_T(r, i);
    loop_ub = x1->size[1] << 16;
    for (i = 0; i < loop_ub; i++) {
      r->data[i] = 0.0;
    }

    emxInit_real_T(&avgPxx, 2);
    localComputeSpectra(r, x1, xStart, xEnd, y, options_range, k1, avgPxx,
                        varargout_2);
    emxFree_real_T(&avgPxx);
  }

  emxFree_real_T(&r);
  emxFree_real_T(&y);
  emxFree_real_T(&x1);
  emxFree_real_T(&xEnd);
  emxFree_real_T(&xStart);
}

static double rt_hypotd(double u0, double u1)
{
  double y;
  double a;
  double b;
  a = fabs(u0);
  b = fabs(u1);
  if (a < b) {
    a /= b;
    y = b * sqrt(a * a + 1.0);
  } else if (a > b) {
    b /= a;
    y = a * sqrt(b * b + 1.0);
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

void CardioRespAnalysis(const emxArray_real_T *rawBreathing, const
  emxArray_real_T *rawHeart, emxArray_real_T *breathingFilter, emxArray_real_T
  *heartFilter, double rates[2])
{
  int datalength;
  int offset;
  int ncopies;
  int Nrep;
  emxArray_real_T *xx;
  int i;
  int loop_ub;
  int k;
  emxArray_real_T *wt;
  emxArray_creal_T *G;
  emxArray_creal_T *H;
  emxArray_creal_T *Vhat;
  emxArray_creal_T *Z;
  emxArray_creal_T *What;
  emxArray_real_T *mra;
  double delta1;
  double varargin_1[49];
  double ex;
  double HR_data[1];
  int HR_size[1];
  datalength = rawBreathing->size[0];
  frexp(rawBreathing->size[0], &offset);
  frexp(rawBreathing->size[0], &ncopies);
  if (rawBreathing->size[0] < 8) {
    ncopies = 8 - rawBreathing->size[0];
    Nrep = (9 - rawBreathing->size[0]) * rawBreathing->size[0];
  } else {
    ncopies = 0;
    Nrep = rawBreathing->size[0];
  }

  emxInit_real_T(&xx, 2);
  i = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = Nrep;
  emxEnsureCapacity_real_T(xx, i);
  loop_ub = rawBreathing->size[0];
  for (i = 0; i < loop_ub; i++) {
    xx->data[i] = rawBreathing->data[i];
  }

  if (ncopies > 0) {
    for (k = 0; k < ncopies; k++) {
      offset = (k + 1) * rawBreathing->size[0];
      for (loop_ub = 0; loop_ub < datalength; loop_ub++) {
        xx->data[offset + loop_ub] = xx->data[loop_ub];
      }
    }
  }

  emxInit_real_T(&wt, 2);
  i = wt->size[0] * wt->size[1];
  wt->size[0] = 10;
  wt->size[1] = rawBreathing->size[0];
  emxEnsureCapacity_real_T(wt, i);
  loop_ub = 10 * rawBreathing->size[0];
  for (i = 0; i < loop_ub; i++) {
    wt->data[i] = 0.0;
  }

  emxInit_creal_T(&G, 2);
  emxInit_creal_T(&H, 2);
  emxInit_creal_T(&Vhat, 2);
  fft(dv, Nrep, G);
  fft(dv1, Nrep, H);
  b_fft(xx, Vhat);
  loop_ub = rawBreathing->size[0];
  emxInit_creal_T(&Z, 2);
  emxInit_creal_T(&What, 2);
  for (datalength = 0; datalength < 9; datalength++) {
    offset = Vhat->size[1];
    Nrep = 1 << datalength;
    i = Z->size[0] * Z->size[1];
    Z->size[0] = Vhat->size[0];
    Z->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(Z, i);
    i = What->size[0] * What->size[1];
    What->size[0] = Vhat->size[0];
    What->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(What, i);
    for (k = 0; k < offset; k++) {
      ncopies = Nrep * k;
      ncopies -= div_s32(ncopies, offset) * Vhat->size[1];
      Z->data[k].re = G->data[ncopies].re * Vhat->data[k].re - G->data[ncopies].
        im * Vhat->data[k].im;
      Z->data[k].im = G->data[ncopies].re * Vhat->data[k].im + G->data[ncopies].
        im * Vhat->data[k].re;
      What->data[k].re = H->data[ncopies].re * Vhat->data[k].re - H->
        data[ncopies].im * Vhat->data[k].im;
      What->data[k].im = H->data[ncopies].re * Vhat->data[k].im + H->
        data[ncopies].im * Vhat->data[k].re;
    }

    i = Vhat->size[0] * Vhat->size[1];
    Vhat->size[0] = 1;
    Vhat->size[1] = Z->size[1];
    emxEnsureCapacity_creal_T(Vhat, i);
    offset = Z->size[0] * Z->size[1];
    for (i = 0; i < offset; i++) {
      Vhat->data[i] = Z->data[i];
    }

    ifft(What, Z);
    i = xx->size[0] * xx->size[1];
    xx->size[0] = 1;
    xx->size[1] = Z->size[1];
    emxEnsureCapacity_real_T(xx, i);
    offset = Z->size[0] * Z->size[1];
    for (i = 0; i < offset; i++) {
      xx->data[i] = Z->data[i].re;
    }

    for (i = 0; i < loop_ub; i++) {
      wt->data[datalength + 10 * i] = xx->data[i];
    }
  }

  emxFree_creal_T(&What);
  emxFree_creal_T(&H);
  emxFree_creal_T(&G);
  ifft(Vhat, Z);
  i = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = Z->size[1];
  emxEnsureCapacity_real_T(xx, i);
  loop_ub = Z->size[0] * Z->size[1];
  emxFree_creal_T(&Vhat);
  for (i = 0; i < loop_ub; i++) {
    xx->data[i] = Z->data[i].re;
  }

  loop_ub = rawBreathing->size[0];
  for (i = 0; i < loop_ub; i++) {
    wt->data[10 * i + 9] = xx->data[i];
  }

  emxInit_real_T(&mra, 2);
  modwtmra(wt, mra);
  i = mra->size[1] - 1;
  offset = breathingFilter->size[0] * breathingFilter->size[1];
  breathingFilter->size[0] = 1;
  breathingFilter->size[1] = mra->size[1];
  emxEnsureCapacity_real_T(breathingFilter, offset);
  emxFree_real_T(&wt);
  for (Nrep = 0; Nrep <= i; Nrep++) {
    ncopies = Nrep * 3;
    offset = ncopies + 1;
    delta1 = mra->data[(ncopies % 3 + 10 * (ncopies / 3)) + 6] + mra->data
      [(offset % 3 + 10 * (offset / 3)) + 6];
    offset = ncopies + 2;
    delta1 += mra->data[(offset % 3 + 10 * (offset / 3)) + 6];
    breathingFilter->data[Nrep] = delta1;
  }

  emxFree_real_T(&mra);
  i = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = breathingFilter->size[1];
  emxEnsureCapacity_real_T(xx, i);
  xx->data[breathingFilter->size[1] - 1] = 50.0;
  xx->data[0] = 0.0;
  if (xx->size[1] >= 3) {
    delta1 = 50.0 / ((double)xx->size[1] - 1.0);
    i = xx->size[1];
    for (k = 0; k <= i - 3; k++) {
      xx->data[k + 1] = ((double)k + 1.0) * delta1;
    }
  }

  b_fft(breathingFilter, Z);
  for (k = 0; k < 49; k++) {
    varargin_1[k] = rt_hypotd(Z->data[k].re, Z->data[k].im);
  }

  emxFree_creal_T(&Z);
  ex = varargin_1[0];
  offset = -1;
  for (k = 0; k < 48; k++) {
    delta1 = varargin_1[k + 1];
    if (ex < delta1) {
      ex = delta1;
      offset = k;
    }
  }

  rates[0] = xx->data[offset + 1] * 60.0;
  heartRateFilt(rawHeart, heartFilter, HR_data, HR_size);
  emxFree_real_T(&xx);
  if (0 <= HR_size[0] - 1) {
    rates[1] = HR_data[0];
  }
}

void CardioRespAnalysis_initialize(void)
{
}

void CardioRespAnalysis_terminate(void)
{
}
