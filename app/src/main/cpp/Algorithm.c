#import <jni.h>
#include <stdlib.h>
#include <stddef.h>
#include "Filters.h"
#include "Filters_emxAPI.h"
#include "Filters_emxutil.h"
#include "Filters_types.h"
#include "rtwtypes.h"
#include "BreathingMaxFFT.h"
#include "BreathingMaxFFT_types.h"

JNIEXPORT jdoubleArray JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_FilterBrHr(JNIEnv* env, jobject this, jdoubleArray dataARR, jboolean isBreathing) {
    jsize size = (*env)->GetArrayLength(env, dataARR);
    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);

    emxArray_real_T *dataArray = NULL;
    dataArray = emxCreateWrapper_real_T(body, size, 1);

    emxArray_real_T *breathingArray;
    emxArray_real_T *heartBeatArray;

    emxInitArray_real_T(&breathingArray, 2);
    emxInitArray_real_T(&heartBeatArray, 2);

    Filters_initialize();

    Filters(dataArray, breathingArray, heartBeatArray);

    jdoubleArray result;
    result = (*env)->NewDoubleArray(env, size);

    if(isBreathing == true){
        (*env)->SetDoubleArrayRegion(env, result, 0, size, breathingArray->data);
    }
    else{
        (*env)->SetDoubleArrayRegion(env, result, 0, size, heartBeatArray->data);
    }

    Filters_terminate();
    (*env)->ReleaseDoubleArrayElements(env, dataARR, body, 0);

    return result;

}

JNIEXPORT jdouble JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_BreathingRateFFT(JNIEnv* env, jobject this, jdoubleArray dataARR, jint samplingFreq) {
    jsize size = (*env)->GetArrayLength(env, dataARR);
    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);

    emxArray_real_T *dataArray = NULL;
    dataArray = emxCreateWrapper_real_T(body, size, 1);

    jdouble result;

    BreathingMaxFFT_initialize();

    result = BreathingMaxFFT(dataArray, samplingFreq);

    BreathingMaxFFT_terminate();

    return result;
}

