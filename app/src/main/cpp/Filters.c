#include "Filters.h"
#include "Filters_emxutil.h"
#include <math.h>

static const double dv[8] = { 0.022785172947974931, -0.0089123507208401943,
  -0.070158812089422817, 0.21061726710176826, 0.56832912170437477,
  0.35186953432761287, -0.020955482562526946, -0.053574450708941054 };

static const double dv1[8] = { -0.053574450708941054, 0.020955482562526946,
  0.35186953432761287, -0.56832912170437477, 0.21061726710176826,
  0.070158812089422817, -0.0089123507208401943, -0.022785172947974931 };

static void b_fft(const emxArray_real_T *x, emxArray_creal_T *y);
static void b_modwtmra(const emxArray_real_T *w, emxArray_real_T *mra);
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
static void d_FFTImplementationCallback_doH(const double x[8], emxArray_creal_T *
  y, int nRows, int nfft, const emxArray_creal_T *wwc, const emxArray_real_T
  *costab, const emxArray_real_T *sintab, const emxArray_real_T *costabinv,
  const emxArray_real_T *sintabinv);
static void d_FFTImplementationCallback_gen(int nRows, emxArray_real_T *costab,
  emxArray_real_T *sintab, emxArray_real_T *sintabinv);
static void d_FFTImplementationCallback_get(int nRowsM1, int nfftLen,
  emxArray_int32_T *bitrevIndex);
static void d_fft(const double x[4], double varargin_1, emxArray_creal_T *y);
static void e_FFTImplementationCallback_doH(const emxArray_real_T *x,
  emxArray_creal_T *y, int unsigned_nRows, const emxArray_real_T *costab, const
  emxArray_real_T *sintab);
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
static void g_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab);
static void h_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv);
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
static void k_FFTImplementationCallback_doH(const double x[4], emxArray_creal_T *
  y, int unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T
  *sintab);
static void l_FFTImplementationCallback_doH(const double x[4], emxArray_creal_T *
  y, int nRows, int nfft, const emxArray_creal_T *wwc, const emxArray_real_T
  *costab, const emxArray_real_T *sintab, const emxArray_real_T *costabinv,
  const emxArray_real_T *sintabinv);
static void m_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab);
static void modwtmra(const emxArray_real_T *w, emxArray_real_T *mra);
static void n_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv);
static void o_FFTImplementationCallback_doH(emxArray_creal_T *y, int
  unsigned_nRows, const emxArray_real_T *costab, const emxArray_real_T *sintab);
static void p_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
  nfft, const emxArray_creal_T *wwc, const emxArray_real_T *costab, const
  emxArray_real_T *sintab, const emxArray_real_T *costabinv, const
  emxArray_real_T *sintabinv);
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

static void b_modwtmra(const emxArray_real_T *w, emxArray_real_T *mra)
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
  int useRadix2_tmp;
  boolean_T useRadix2;
  int N2blue;
  int nRows;
  emxArray_creal_T *yCol;
  double ww_data[10];
  emxArray_creal_T *wwc;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  int nInt2m1;
  int idx;
  emxArray_creal_T *b_yCol;
  double nt_im;
  double nt_re;
  emxArray_creal_T c_yCol;
  int iv[2];
  int iv1[2];
  emxArray_creal_T d_yCol;
  int iv2[2];
  int iv3[2];
  ncw = w->size[1];
  cfslength = w->size[1];
  if (w->size[1] < 4) {
    ncopies = 4 - w->size[1];
    cfslength = (5 - w->size[1]) * w->size[1];
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
  useRadix2_tmp = cfslength & (cfslength - 1);
  useRadix2 = (useRadix2_tmp == 0);
  c_FFTImplementationCallback_get(cfslength, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, b_w, v, sintabinv);
  emxInit_creal_T(&yCol, 1);
  emxInit_creal_T(&wwc, 1);
  emxInit_creal_T(&fv, 1);
  emxInit_creal_T(&b_fv, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = cfslength;
    emxEnsureCapacity_creal_T(yCol, i);
    if (cfslength > 4) {
      i = yCol->size[0];
      yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(yCol, i);
      for (i = 0; i < cfslength; i++) {
        yCol->data[i].re = 0.0;
        yCol->data[i].im = 0.0;
      }
    }

    m_FFTImplementationCallback_doH(yCol, cfslength, b_w, v);
  } else {
    i = cfslength & 1;
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
    if (cfslength > 4) {
      i1 = yCol->size[0];
      yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(yCol, i1);
      for (i1 = 0; i1 < cfslength; i1++) {
        yCol->data[i1].re = 0.0;
        yCol->data[i1].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      n_FFTImplementationCallback_doH(yCol, cfslength, N2blue, wwc, b_w, v, b_w,
        sintabinv);
    } else {
      yCol->data[0].re = wwc->data[cfslength - 1].re * 0.4623966089481239;
      yCol->data[0].im = wwc->data[cfslength - 1].im * -0.4623966089481239;
      yCol->data[1].re = wwc->data[cfslength].re * 0.53264408776809669;
      yCol->data[1].im = wwc->data[cfslength].im * -0.53264408776809669;
      yCol->data[2].re = wwc->data[cfslength + 1].re * 0.037603393287944008;
      yCol->data[2].im = wwc->data[cfslength + 1].im * -0.037603393287944008;
      yCol->data[3].re = wwc->data[cfslength + 2].re * -0.032644090004164718;
      yCol->data[3].im = wwc->data[cfslength + 2].im * 0.032644090004164718;
      for (k = 5; k <= cfslength; k++) {
        yCol->data[k - 1].re = 0.0;
        yCol->data[k - 1].im = 0.0;
      }

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
    }
  }

  useRadix2 = (useRadix2_tmp == 0);
  c_FFTImplementationCallback_get(cfslength, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, b_w, v, sintabinv);
  emxInit_creal_T(&b_yCol, 1);
  if (useRadix2) {
    i = b_yCol->size[0];
    b_yCol->size[0] = cfslength;
    emxEnsureCapacity_creal_T(b_yCol, i);
    if (cfslength > 4) {
      i = b_yCol->size[0];
      b_yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(b_yCol, i);
      for (i = 0; i < cfslength; i++) {
        b_yCol->data[i].re = 0.0;
        b_yCol->data[i].im = 0.0;
      }
    }

    o_FFTImplementationCallback_doH(b_yCol, cfslength, b_w, v);
  } else {
    i = cfslength & 1;
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

    i1 = b_yCol->size[0];
    b_yCol->size[0] = cfslength;
    emxEnsureCapacity_creal_T(b_yCol, i1);
    if (cfslength > 4) {
      i1 = b_yCol->size[0];
      b_yCol->size[0] = cfslength;
      emxEnsureCapacity_creal_T(b_yCol, i1);
      for (i1 = 0; i1 < cfslength; i1++) {
        b_yCol->data[i1].re = 0.0;
        b_yCol->data[i1].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      p_FFTImplementationCallback_doH(b_yCol, cfslength, N2blue, wwc, b_w, v,
        b_w, sintabinv);
    } else {
      b_yCol->data[0].re = wwc->data[cfslength - 1].re * -0.032644090004164718;
      b_yCol->data[0].im = wwc->data[cfslength - 1].im * 0.032644090004164718;
      b_yCol->data[1].re = wwc->data[cfslength].re * -0.037603393287944008;
      b_yCol->data[1].im = wwc->data[cfslength].im * 0.037603393287944008;
      b_yCol->data[2].re = wwc->data[cfslength + 1].re * 0.53264408776809669;
      b_yCol->data[2].im = wwc->data[cfslength + 1].im * -0.53264408776809669;
      b_yCol->data[3].re = wwc->data[cfslength + 2].re * -0.4623966089481239;
      b_yCol->data[3].im = wwc->data[cfslength + 2].im * 0.4623966089481239;
      for (k = 5; k <= cfslength; k++) {
        b_yCol->data[k - 1].re = 0.0;
        b_yCol->data[k - 1].im = 0.0;
      }

      c_FFTImplementationCallback_r2b(b_yCol, N2blue, b_w, v, fv);
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
        b_yCol->data[idx].re = wwc->data[k - 1].re * fv->data[k - 1].re +
          wwc->data[k - 1].im * fv->data[k - 1].im;
        b_yCol->data[idx].im = wwc->data[k - 1].re * fv->data[k - 1].im -
          wwc->data[k - 1].im * fv->data[k - 1].re;
        idx++;
      }
    }
  }

  emxFree_creal_T(&b_fv);
  emxFree_creal_T(&fv);
  emxFree_creal_T(&wwc);
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

      c_yCol = *yCol;
      iv1[0] = 1;
      iv1[1] = cfslength;
      c_yCol.size = &iv1[0];
      c_yCol.numDimensions = 2;
      d_yCol = *b_yCol;
      iv3[0] = 1;
      iv3[1] = cfslength;
      d_yCol.size = &iv3[0];
      d_yCol.numDimensions = 2;
      imodwtrec(sintabinv, b_w, &c_yCol, &d_yCol, nRows, v);
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

    c_yCol = *yCol;
    iv[0] = 1;
    iv[1] = cfslength;
    c_yCol.size = &iv[0];
    c_yCol.numDimensions = 2;
    d_yCol = *b_yCol;
    iv2[0] = 1;
    iv2[1] = cfslength;
    d_yCol.size = &iv2[0];
    d_yCol.numDimensions = 2;
    imodwtrec(sintabinv, nullinput, &c_yCol, &d_yCol, ncopies + 1, v);
  }

  emxFree_creal_T(&b_yCol);
  emxFree_real_T(&sintabinv);
  emxFree_creal_T(&yCol);
  emxFree_real_T(&nullinput);
  loop_ub = w->size[1];
  for (i = 0; i < loop_ub; i++) {
    mra->data[10 * i + 9] = v->data[i];
  }

  emxFree_real_T(&v);
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

static void d_fft(const double x[4], double varargin_1, emxArray_creal_T *y)
{
  emxArray_real_T *costab;
  emxArray_real_T *sintab;
  emxArray_real_T *sintabinv;
  int len_tmp;
  int useRadix2_tmp;
  boolean_T useRadix2;
  int N2blue;
  int nRows;
  emxArray_creal_T *yCol;
  int i;
  emxArray_creal_T *wwc;
  int nInt2m1;
  int rt;
  int idx;
  int nInt2;
  int k;
  int b_y;
  double nt_im;
  double nt_re;
  emxArray_creal_T *fv;
  emxArray_creal_T *b_fv;
  emxInit_real_T(&costab, 2);
  emxInit_real_T(&sintab, 2);
  emxInit_real_T(&sintabinv, 2);
  len_tmp = (int)varargin_1;
  useRadix2_tmp = len_tmp - 1;
  useRadix2 = ((len_tmp & useRadix2_tmp) == 0);
  c_FFTImplementationCallback_get((int)varargin_1, useRadix2, &N2blue, &nRows);
  c_FFTImplementationCallback_gen(nRows, useRadix2, costab, sintab, sintabinv);
  emxInit_creal_T(&yCol, 1);
  if (useRadix2) {
    i = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, i);
    if (len_tmp > 4) {
      i = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, i);
      for (i = 0; i < len_tmp; i++) {
        yCol->data[i].re = 0.0;
        yCol->data[i].im = 0.0;
      }
    }

    k_FFTImplementationCallback_doH(x, yCol, (int)varargin_1, costab, sintab);
  } else {
    i = len_tmp & 1;
    emxInit_creal_T(&wwc, 1);
    if (i == 0) {
      nRows = len_tmp / 2;
      nInt2m1 = (nRows + nRows) - 1;
      rt = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, rt);
      idx = nRows;
      rt = 0;
      wwc->data[nRows - 1].re = 1.0;
      wwc->data[nRows - 1].im = 0.0;
      nInt2 = nRows << 1;
      for (k = 0; k <= nRows - 2; k++) {
        b_y = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= b_y) {
          rt += b_y - nInt2;
        } else {
          rt += b_y;
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
      rt = nInt2m1 - 1;
      for (k = rt; k >= nRows; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    } else {
      nInt2m1 = (len_tmp + len_tmp) - 1;
      rt = wwc->size[0];
      wwc->size[0] = nInt2m1;
      emxEnsureCapacity_creal_T(wwc, rt);
      idx = len_tmp;
      rt = 0;
      wwc->data[useRadix2_tmp].re = 1.0;
      wwc->data[useRadix2_tmp].im = 0.0;
      nInt2 = len_tmp << 1;
      for (k = 0; k <= len_tmp - 2; k++) {
        b_y = ((k + 1) << 1) - 1;
        if (nInt2 - rt <= b_y) {
          rt += b_y - nInt2;
        } else {
          rt += b_y;
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
      rt = nInt2m1 - 1;
      for (k = rt; k >= len_tmp; k--) {
        wwc->data[k] = wwc->data[idx];
        idx++;
      }
    }

    rt = yCol->size[0];
    yCol->size[0] = len_tmp;
    emxEnsureCapacity_creal_T(yCol, rt);
    if (len_tmp > 4) {
      rt = yCol->size[0];
      yCol->size[0] = len_tmp;
      emxEnsureCapacity_creal_T(yCol, rt);
      for (rt = 0; rt < len_tmp; rt++) {
        yCol->data[rt].re = 0.0;
        yCol->data[rt].im = 0.0;
      }
    }

    if ((N2blue != 1) && (i == 0)) {
      l_FFTImplementationCallback_doH(x, yCol, (int)varargin_1, N2blue, wwc,
        costab, sintab, costab, sintabinv);
    } else {
      yCol->data[0].re = wwc->data[useRadix2_tmp].re * x[0];
      yCol->data[0].im = wwc->data[useRadix2_tmp].im * -x[0];
      yCol->data[1].re = wwc->data[len_tmp].re * x[1];
      yCol->data[1].im = wwc->data[len_tmp].im * -x[1];
      rt = len_tmp + 1;
      yCol->data[2].re = wwc->data[rt].re * x[2];
      yCol->data[2].im = wwc->data[rt].im * -x[2];
      rt = len_tmp + 2;
      yCol->data[3].re = wwc->data[rt].re * x[3];
      yCol->data[3].im = wwc->data[rt].im * -x[3];
      for (k = 5; k <= len_tmp; k++) {
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
      rt = fv->size[0];
      for (i = 0; i < rt; i++) {
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
        rt = fv->size[0];
        for (i = 0; i < rt; i++) {
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

  temp1_re = y->data[iterVar].re;
  temp1_im = y->data[iterVar].im;
  y_im = y->data[iterVar].re * reconVar1->data[iterVar].im + y->data[iterVar].im
    * reconVar1->data[iterVar].re;
  y_re = y->data[iterVar].re;
  b_y_im = -y->data[iterVar].im;
  y->data[iterVar].re = 0.5 * ((y->data[iterVar].re * reconVar1->data[iterVar].
    re - y->data[iterVar].im * reconVar1->data[iterVar].im) + (y_re *
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
    idx -= idx / Vin->size[1] * Vin->size[1];
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

static void k_FFTImplementationCallback_doH(const double x[4], emxArray_creal_T *
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
  y->data[bitrevIndex->data[0] - 1].re = x[0];
  y->data[bitrevIndex->data[0] - 1].im = x[1];
  y->data[bitrevIndex->data[1] - 1].re = x[2];
  y->data[bitrevIndex->data[1] - 1].im = x[3];
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

static void l_FFTImplementationCallback_doH(const double x[4], emxArray_creal_T *
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
  double temp_re;
  double temp_im;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iDelta2;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double twid_re;
  double twid_im;
  emxArray_creal_T *fv;
  int temp_re_tmp;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 4) {
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
  temp_re = wwc->data[hnRows - 1].re;
  temp_im = wwc->data[hnRows - 1].im;
  ytmp->data[0].re = temp_re * x[0] + temp_im * x[1];
  ytmp->data[0].im = temp_re * x[1] - temp_im * x[0];
  ytmp->data[1].re = wwc->data[hnRows].re * x[2] + wwc->data[hnRows].im * x[3];
  ytmp->data[1].im = wwc->data[hnRows].re * x[3] - wwc->data[hnRows].im * x[2];
  if (3 <= hnRows) {
    for (i = 3; i <= hnRows; i++) {
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

  iDelta2 = ytmp->size[0];
  if (iDelta2 >= loop_ub_tmp) {
    iDelta2 = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= iDelta2 - 2; i++) {
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
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += iDelta2) {
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
        i += iDelta2;
      }

      ix++;
    }

    k /= 2;
    idx = iDelta2;
    iDelta2 += iDelta2;
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

static void m_FFTImplementationCallback_doH(emxArray_creal_T *y, int
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
  y->data[bitrevIndex->data[0] - 1].re = 0.4623966089481239;
  y->data[bitrevIndex->data[0] - 1].im = 0.53264408776809669;
  y->data[bitrevIndex->data[1] - 1].re = 0.037603393287944008;
  y->data[bitrevIndex->data[1] - 1].im = -0.032644090004164718;
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

static void n_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
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
  double temp_re;
  double temp_im;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iDelta2;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double twid_re;
  double twid_im;
  emxArray_creal_T *fv;
  int temp_re_tmp;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 4) {
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
  temp_re = wwc->data[hnRows - 1].re;
  temp_im = wwc->data[hnRows - 1].im;
  ytmp->data[0].re = temp_re * 0.4623966089481239 + temp_im *
    0.53264408776809669;
  ytmp->data[0].im = temp_re * 0.53264408776809669 - temp_im *
    0.4623966089481239;
  ytmp->data[1].re = wwc->data[hnRows].re * 0.037603393287944008 + wwc->
    data[hnRows].im * -0.032644090004164718;
  ytmp->data[1].im = wwc->data[hnRows].re * -0.032644090004164718 - wwc->
    data[hnRows].im * 0.037603393287944008;
  if (3 <= hnRows) {
    for (i = 3; i <= hnRows; i++) {
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

  iDelta2 = ytmp->size[0];
  if (iDelta2 >= loop_ub_tmp) {
    iDelta2 = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= iDelta2 - 2; i++) {
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
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += iDelta2) {
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
        i += iDelta2;
      }

      ix++;
    }

    k /= 2;
    idx = iDelta2;
    iDelta2 += iDelta2;
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

static void o_FFTImplementationCallback_doH(emxArray_creal_T *y, int
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
  y->data[bitrevIndex->data[0] - 1].re = -0.032644090004164718;
  y->data[bitrevIndex->data[0] - 1].im = -0.037603393287944008;
  y->data[bitrevIndex->data[1] - 1].re = 0.53264408776809669;
  y->data[bitrevIndex->data[1] - 1].im = -0.4623966089481239;
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

static void p_FFTImplementationCallback_doH(emxArray_creal_T *y, int nRows, int
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
  double temp_re;
  double temp_im;
  emxArray_creal_T *fy;
  int loop_ub_tmp;
  int iDelta2;
  int iheight;
  int nRowsD2;
  int k;
  int ju;
  boolean_T tst;
  double twid_re;
  double twid_im;
  emxArray_creal_T *fv;
  int temp_re_tmp;
  int ihi;
  emxInit_creal_T(&ytmp, 1);
  hnRows = nRows / 2;
  ix = ytmp->size[0];
  ytmp->size[0] = hnRows;
  emxEnsureCapacity_creal_T(ytmp, ix);
  if (hnRows > 4) {
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
  temp_re = wwc->data[hnRows - 1].re;
  temp_im = wwc->data[hnRows - 1].im;
  ytmp->data[0].re = temp_re * -0.032644090004164718 + temp_im *
    -0.037603393287944008;
  ytmp->data[0].im = temp_re * -0.037603393287944008 - temp_im *
    -0.032644090004164718;
  ytmp->data[1].re = wwc->data[hnRows].re * 0.53264408776809669 + wwc->
    data[hnRows].im * -0.4623966089481239;
  ytmp->data[1].im = wwc->data[hnRows].re * -0.4623966089481239 - wwc->
    data[hnRows].im * 0.53264408776809669;
  if (3 <= hnRows) {
    for (i = 3; i <= hnRows; i++) {
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

  iDelta2 = ytmp->size[0];
  if (iDelta2 >= loop_ub_tmp) {
    iDelta2 = loop_ub_tmp;
  }

  iheight = loop_ub_tmp - 2;
  nRowsD2 = loop_ub_tmp / 2;
  k = nRowsD2 / 2;
  ix = 0;
  idx = 0;
  ju = 0;
  for (i = 0; i <= iDelta2 - 2; i++) {
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
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += iDelta2) {
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
        i += iDelta2;
      }

      ix++;
    }

    k /= 2;
    idx = iDelta2;
    iDelta2 += iDelta2;
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

void Filters(const emxArray_real_T *data, emxArray_real_T *BR_Filter,
             emxArray_real_T *HR_Filter)
{
  int datalength;
  int ncopies;
  int offset;
  int Nrep;
  emxArray_real_T *xx;
  int k;
  int loop_ub;
  emxArray_real_T *wt;
  emxArray_creal_T *G;
  emxArray_creal_T *H;
  emxArray_creal_T *Vhat;
  emxArray_creal_T *b_Vhat;
  emxArray_creal_T *What;
  emxArray_real_T *mra;
  int eint;
  int b_eint;
  double d;
  static const double Lo[4] = { 0.4623966089481239, 0.53264408776809669,
    0.037603393287944008, -0.032644090004164718 };

  static const double Hi[4] = { -0.032644090004164718, -0.037603393287944008,
    0.53264408776809669, -0.4623966089481239 };

  datalength = data->size[0];
  frexp(data->size[0], &ncopies);
  frexp(data->size[0], &offset);
  if (data->size[0] < 8) {
    ncopies = 8 - data->size[0];
    Nrep = (9 - data->size[0]) * data->size[0];
  } else {
    ncopies = 0;
    Nrep = data->size[0];
  }

  emxInit_real_T(&xx, 2);
  k = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = Nrep;
  emxEnsureCapacity_real_T(xx, k);
  loop_ub = data->size[0];
  for (k = 0; k < loop_ub; k++) {
    xx->data[k] = data->data[k];
  }

  if (ncopies > 0) {
    for (k = 0; k < ncopies; k++) {
      offset = (k + 1) * data->size[0];
      for (loop_ub = 0; loop_ub < datalength; loop_ub++) {
        xx->data[offset + loop_ub] = xx->data[loop_ub];
      }
    }
  }

  emxInit_real_T(&wt, 2);
  k = wt->size[0] * wt->size[1];
  wt->size[0] = 10;
  wt->size[1] = data->size[0];
  emxEnsureCapacity_real_T(wt, k);
  loop_ub = 10 * data->size[0];
  for (k = 0; k < loop_ub; k++) {
    wt->data[k] = 0.0;
  }

  emxInit_creal_T(&G, 2);
  emxInit_creal_T(&H, 2);
  emxInit_creal_T(&Vhat, 2);
  fft(dv, Nrep, G);
  fft(dv1, Nrep, H);
  b_fft(xx, Vhat);
  loop_ub = data->size[0];
  emxInit_creal_T(&b_Vhat, 2);
  emxInit_creal_T(&What, 2);
  for (datalength = 0; datalength < 9; datalength++) {
    offset = Vhat->size[1];
    Nrep = 1 << datalength;
    k = b_Vhat->size[0] * b_Vhat->size[1];
    b_Vhat->size[0] = Vhat->size[0];
    b_Vhat->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(b_Vhat, k);
    k = What->size[0] * What->size[1];
    What->size[0] = Vhat->size[0];
    What->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(What, k);
    for (k = 0; k < offset; k++) {
      ncopies = Nrep * k;
      ncopies -= ncopies / offset * Vhat->size[1];
      b_Vhat->data[k].re = G->data[ncopies].re * Vhat->data[k].re - G->
        data[ncopies].im * Vhat->data[k].im;
      b_Vhat->data[k].im = G->data[ncopies].re * Vhat->data[k].im + G->
        data[ncopies].im * Vhat->data[k].re;
      What->data[k].re = H->data[ncopies].re * Vhat->data[k].re - H->
        data[ncopies].im * Vhat->data[k].im;
      What->data[k].im = H->data[ncopies].re * Vhat->data[k].im + H->
        data[ncopies].im * Vhat->data[k].re;
    }

    k = Vhat->size[0] * Vhat->size[1];
    Vhat->size[0] = 1;
    Vhat->size[1] = b_Vhat->size[1];
    emxEnsureCapacity_creal_T(Vhat, k);
    ncopies = b_Vhat->size[0] * b_Vhat->size[1];
    for (k = 0; k < ncopies; k++) {
      Vhat->data[k] = b_Vhat->data[k];
    }

    ifft(What, b_Vhat);
    k = xx->size[0] * xx->size[1];
    xx->size[0] = 1;
    xx->size[1] = b_Vhat->size[1];
    emxEnsureCapacity_real_T(xx, k);
    ncopies = b_Vhat->size[0] * b_Vhat->size[1];
    for (k = 0; k < ncopies; k++) {
      xx->data[k] = b_Vhat->data[k].re;
    }

    for (k = 0; k < loop_ub; k++) {
      wt->data[datalength + 10 * k] = xx->data[k];
    }
  }

  ifft(Vhat, b_Vhat);
  k = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = b_Vhat->size[1];
  emxEnsureCapacity_real_T(xx, k);
  loop_ub = b_Vhat->size[0] * b_Vhat->size[1];
  for (k = 0; k < loop_ub; k++) {
    xx->data[k] = b_Vhat->data[k].re;
  }

  loop_ub = data->size[0];
  for (k = 0; k < loop_ub; k++) {
    wt->data[10 * k + 9] = xx->data[k];
  }

  emxInit_real_T(&mra, 2);
  modwtmra(wt, mra);
  k = mra->size[1] - 1;
  Nrep = BR_Filter->size[0] * BR_Filter->size[1];
  BR_Filter->size[0] = 1;
  BR_Filter->size[1] = mra->size[1];
  emxEnsureCapacity_real_T(BR_Filter, Nrep);
  for (offset = 0; offset <= k; offset++) {
    ncopies = offset * 3;
    Nrep = ncopies + 1;
    d = mra->data[(ncopies % 3 + 10 * (ncopies / 3)) + 6] + mra->data[(Nrep % 3
      + 10 * (Nrep / 3)) + 6];
    Nrep = ncopies + 2;
    d += mra->data[(Nrep % 3 + 10 * (Nrep / 3)) + 6];
    BR_Filter->data[offset] = d;
  }

  datalength = data->size[0];
  frexp(data->size[0], &eint);
  frexp(data->size[0], &b_eint);
  if (data->size[0] < 4) {
    ncopies = 4 - data->size[0];
    Nrep = (5 - data->size[0]) * data->size[0];
  } else {
    ncopies = 0;
    Nrep = data->size[0];
  }

  k = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = Nrep;
  emxEnsureCapacity_real_T(xx, k);
  loop_ub = data->size[0];
  for (k = 0; k < loop_ub; k++) {
    xx->data[k] = data->data[k];
  }

  if (ncopies > 0) {
    for (k = 0; k < ncopies; k++) {
      offset = (k + 1) * data->size[0];
      for (loop_ub = 0; loop_ub < datalength; loop_ub++) {
        xx->data[offset + loop_ub] = xx->data[loop_ub];
      }
    }
  }

  k = wt->size[0] * wt->size[1];
  wt->size[0] = 10;
  wt->size[1] = data->size[0];
  emxEnsureCapacity_real_T(wt, k);
  loop_ub = 10 * data->size[0];
  for (k = 0; k < loop_ub; k++) {
    wt->data[k] = 0.0;
  }

  d_fft(Lo, Nrep, G);
  d_fft(Hi, Nrep, H);
  b_fft(xx, Vhat);
  loop_ub = data->size[0];
  for (datalength = 0; datalength < 9; datalength++) {
    offset = Vhat->size[1];
    Nrep = 1 << datalength;
    k = b_Vhat->size[0] * b_Vhat->size[1];
    b_Vhat->size[0] = Vhat->size[0];
    b_Vhat->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(b_Vhat, k);
    k = What->size[0] * What->size[1];
    What->size[0] = Vhat->size[0];
    What->size[1] = Vhat->size[1];
    emxEnsureCapacity_creal_T(What, k);
    for (k = 0; k < offset; k++) {
      ncopies = Nrep * k;
      ncopies -= ncopies / offset * Vhat->size[1];
      b_Vhat->data[k].re = G->data[ncopies].re * Vhat->data[k].re - G->
        data[ncopies].im * Vhat->data[k].im;
      b_Vhat->data[k].im = G->data[ncopies].re * Vhat->data[k].im + G->
        data[ncopies].im * Vhat->data[k].re;
      What->data[k].re = H->data[ncopies].re * Vhat->data[k].re - H->
        data[ncopies].im * Vhat->data[k].im;
      What->data[k].im = H->data[ncopies].re * Vhat->data[k].im + H->
        data[ncopies].im * Vhat->data[k].re;
    }

    k = Vhat->size[0] * Vhat->size[1];
    Vhat->size[0] = 1;
    Vhat->size[1] = b_Vhat->size[1];
    emxEnsureCapacity_creal_T(Vhat, k);
    ncopies = b_Vhat->size[0] * b_Vhat->size[1];
    for (k = 0; k < ncopies; k++) {
      Vhat->data[k] = b_Vhat->data[k];
    }

    ifft(What, b_Vhat);
    k = xx->size[0] * xx->size[1];
    xx->size[0] = 1;
    xx->size[1] = b_Vhat->size[1];
    emxEnsureCapacity_real_T(xx, k);
    ncopies = b_Vhat->size[0] * b_Vhat->size[1];
    for (k = 0; k < ncopies; k++) {
      xx->data[k] = b_Vhat->data[k].re;
    }

    for (k = 0; k < loop_ub; k++) {
      wt->data[datalength + 10 * k] = xx->data[k];
    }
  }

  emxFree_creal_T(&What);
  emxFree_creal_T(&H);
  emxFree_creal_T(&G);
  ifft(Vhat, b_Vhat);
  k = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = b_Vhat->size[1];
  emxEnsureCapacity_real_T(xx, k);
  loop_ub = b_Vhat->size[0] * b_Vhat->size[1];
  emxFree_creal_T(&Vhat);
  for (k = 0; k < loop_ub; k++) {
    xx->data[k] = b_Vhat->data[k].re;
  }

  emxFree_creal_T(&b_Vhat);
  loop_ub = data->size[0];
  for (k = 0; k < loop_ub; k++) {
    wt->data[10 * k + 9] = xx->data[k];
  }

  b_modwtmra(wt, mra);
  k = mra->size[1] - 1;
  loop_ub = mra->size[1];
  Nrep = xx->size[0] * xx->size[1];
  xx->size[0] = 1;
  xx->size[1] = mra->size[1];
  emxEnsureCapacity_real_T(xx, Nrep);
  emxFree_real_T(&wt);
  for (Nrep = 0; Nrep < loop_ub; Nrep++) {
    xx->data[Nrep] = mra->data[10 * Nrep + 4];
  }

  Nrep = HR_Filter->size[0] * HR_Filter->size[1];
  HR_Filter->size[0] = 1;
  HR_Filter->size[1] = xx->size[1];
  emxEnsureCapacity_real_T(HR_Filter, Nrep);
  emxFree_real_T(&xx);
  for (offset = 0; offset <= k; offset++) {
    HR_Filter->data[offset] = mra->data[10 * offset + 4];
  }

  emxFree_real_T(&mra);
}

void Filters_initialize(void)
{
}

void Filters_terminate(void)
{
}
