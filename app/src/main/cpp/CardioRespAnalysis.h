#ifndef CARDIORESPANALYSIS_H
#define CARDIORESPANALYSIS_H
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "CardioRespAnalysis_types.h"

extern void CardioRespAnalysis(const emxArray_real_T *rawBreathing, const
  emxArray_real_T *rawHeart, emxArray_real_T *breathingFilter, emxArray_real_T
  *heartFilter, double rates[2]);
extern void CardioRespAnalysis_initialize(void);
extern void CardioRespAnalysis_terminate(void);

#endif
