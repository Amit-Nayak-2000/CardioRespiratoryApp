#ifndef FILTERS_H
#define FILTERS_H
#include <stddef.h>
#include <stdlib.h>
#include "rtwtypes.h"
#include "Filters_types.h"

extern void Filters(const emxArray_real_T *data, emxArray_real_T *BR_Filter,
                    emxArray_real_T *HR_Filter);
extern void Filters_initialize(void);
extern void Filters_terminate(void);

#endif
