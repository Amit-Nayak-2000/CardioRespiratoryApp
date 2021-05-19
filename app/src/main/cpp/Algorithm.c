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

//JNIEXPORT jdoubleArray JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_FilterBrHr(JNIEnv* env, jobject this, jdoubleArray dataARR, jboolean isBreathing) {
//    jsize size = (*env)->GetArrayLength(env, dataARR);
//    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);
//
//    double rates[2];
//
//    emxArray_real_T *dataArray = NULL;
//    dataArray = emxCreateWrapper_real_T(body, size, 1);
//
//    emxArray_real_T *breathingArray;
//    emxArray_real_T *heartBeatArray;
//
//    emxInitArray_real_T(&breathingArray, 2);
//    emxInitArray_real_T(&heartBeatArray, 2);
//
//    CardioRespAnalysis_initialize();
//
////    Filters(dataArray, breathingArray, heartBeatArray);
//
//    CardioRespAnalysis(dataArray, dataArray, breathingArray, heartBeatArray, rates);
//
//    jdoubleArray result;
//    result = (*env)->NewDoubleArray(env, size);
//
//    if(isBreathing == true){
//        (*env)->SetDoubleArrayRegion(env, result, 0, size, breathingArray->data);
//    }
//    else{
//        (*env)->SetDoubleArrayRegion(env, result, 0, size, heartBeatArray->data);
//    }
//
//    CardioRespAnalysis_terminate();
//    (*env)->ReleaseDoubleArrayElements(env, dataARR, body, 0);
//
//    return result;
//
//}
//
//JNIEXPORT jdouble JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_FinalMetric(JNIEnv* env, jobject this, jdoubleArray dataARR, jboolean isBreathing) {
//    jsize size = (*env)->GetArrayLength(env, dataARR);
//    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);
//
//    double rates[2];
//
//    emxArray_real_T *dataArray = NULL;
//    dataArray = emxCreateWrapper_real_T(body, size, 1);
//
//    emxArray_real_T *breathingArray;
//    emxArray_real_T *heartBeatArray;
//
//    emxInitArray_real_T(&breathingArray, 2);
//    emxInitArray_real_T(&heartBeatArray, 2);
//
//    CardioRespAnalysis_initialize();
//
////    Filters(dataArray, breathingArray, heartBeatArray);
//
//    CardioRespAnalysis(dataArray, dataArray, breathingArray, heartBeatArray, rates);
//
//    jdouble result;
//
//
//    if(isBreathing == 1){
//        result = rates[0];
//    }
//    else{
//        result = rates[1];
//    }
//
//    CardioRespAnalysis_terminate();
//
//    return result;
//
//}

//JNIEXPORT jdoubleArray JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_FilterBrHr(JNIEnv* env, jobject this, jdoubleArray dataARR, jboolean isBreathing) {
//    jsize size = (*env)->GetArrayLength(env, dataARR);
//    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);
//
//    emxArray_real_T *dataArray = NULL;
//    dataArray = emxCreateWrapper_real_T(body, size, 1);
//
//    emxArray_real_T *breathingArray;
//    emxArray_real_T *heartBeatArray;
//
//    emxInitArray_real_T(&breathingArray, 2);
//    emxInitArray_real_T(&heartBeatArray, 2);
//
//    Filters_initialize();
//
//    Filters(dataArray, breathingArray, heartBeatArray);
//
//    jdoubleArray result;
//    result = (*env)->NewDoubleArray(env, size);
//
//    if(isBreathing == true){
//        (*env)->SetDoubleArrayRegion(env, result, 0, size, breathingArray->data);
//    }
//    else{
//        (*env)->SetDoubleArrayRegion(env, result, 0, size, heartBeatArray->data);
//    }
//
//    Filters_terminate();
//    (*env)->ReleaseDoubleArrayElements(env, dataARR, body, 0);
//
//    return result;
//
//}


//
//JNIEXPORT jdouble JNICALL Java_com_example_cardiorespiratoryfilter_MainActivity_BreathingRateFFT(JNIEnv* env, jobject this, jdoubleArray dataARR, jint samplingFreq) {
//    jsize size = (*env)->GetArrayLength(env, dataARR);
//    jdouble *body = (*env)->GetDoubleArrayElements(env, dataARR, 0);
//
//    emxArray_real_T *dataArray = NULL;
//    dataArray = emxCreateWrapper_real_T(body, size, 1);
//
//    jdouble result;
//
//    BreathingMaxFFT_initialize();
//
//    result = BreathingMaxFFT(dataArray, samplingFreq);
//
//    BreathingMaxFFT_terminate();
//
//    return result;
//}




