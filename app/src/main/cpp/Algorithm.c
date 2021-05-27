#import <jni.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <stdbool.h>
#include "CardioRespAnalysis.h"
#include "CardioRespAnalysis_types.h"
#include "CardioRespAnalysis_emxutil.h"
#include "CardioRespAnalysis_emxAPI.h"
#include "rtwtypes.h"
#include "tmwtypes.h"

JNIEXPORT void JNICALL
Java_com_example_cardiorespiratoryfilter_MainActivity_FilterBrHr(JNIEnv *env, jobject javaThis, jobject obj, jdoubleArray breathingSensor, jdoubleArray heartSensor) {
    jclass objectclass = (*env)->GetObjectClass(env, obj);

    jfieldID breathingField = (*env)->GetFieldID(env, objectclass, "breathingFilter", "[D");
    jfieldID heartField = (*env)->GetFieldID(env, objectclass, "heartFilter", "[D");
    jfieldID ratesField = (*env)->GetFieldID(env, objectclass, "rates", "[D");

    jdoubleArray breathingResult = (*env)->GetObjectField(env, obj, breathingField);
    jdoubleArray heartResult = (*env)->GetObjectField(env, obj, heartField);
    jdoubleArray ratesResult = (*env)->GetObjectField(env, obj, ratesField);

    jsize size = (*env)->GetArrayLength(env, breathingSensor);
    jdouble *breathingBody = (*env)->GetDoubleArrayElements(env, breathingSensor, 0);
    jdouble *heartBody = (*env)->GetDoubleArrayElements(env, heartSensor, 0);

    double rates[2];

    emxArray_real_T *breathingData = NULL;
    emxArray_real_T *heartData = NULL;
    breathingData = emxCreateWrapper_real_T(breathingBody, size, 1);
    heartData = emxCreateWrapper_real_T(heartBody, size, 1);

    emxArray_real_T *filteredBreathing;
    emxArray_real_T *filteredHeart;
    emxInitArray_real_T(&filteredBreathing, 2);
    emxInitArray_real_T(&filteredHeart, 2);

    CardioRespAnalysis_initialize();
    CardioRespAnalysis(breathingData, heartData, filteredBreathing, filteredHeart, rates);

    (*env)->SetDoubleArrayRegion(env, breathingResult, 0, size, filteredBreathing->data);
    (*env)->SetDoubleArrayRegion(env, heartResult, 0, size, filteredHeart->data);
    (*env)->SetDoubleArrayRegion(env, ratesResult, 0, 2, (const jdouble*) rates);

    CardioRespAnalysis_terminate();

    (*env)->ReleaseDoubleArrayElements(env, breathingSensor, breathingBody, 0);
    (*env)->ReleaseDoubleArrayElements(env, heartSensor, heartBody, 0);


}