#ifndef CARDIORESPANALYSIS_EMXAPI_H
#define CARDIORESPANALYSIS_EMXAPI_H
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "CardioRespAnalysis_types.h"

extern emxArray_real_T *emxCreateND_real_T(int numDimensions, const int *size);
extern emxArray_real_T *emxCreateWrapperND_real_T(double *data, int
  numDimensions, const int *size);
extern emxArray_real_T *emxCreateWrapper_real_T(double *data, int rows, int cols);
extern emxArray_real_T *emxCreate_real_T(int rows, int cols);
extern void emxDestroyArray_real_T(emxArray_real_T *emxArray);
extern void emxInitArray_real_T(emxArray_real_T **pEmxArray, int numDimensions);

#endif
