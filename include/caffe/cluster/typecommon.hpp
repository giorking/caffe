//NovuNetThread class header file
//This file is to define member varialbes and functions of NovuNetThread class
//Author: Xin Chen
//February 9, 2016
//NovuMind Inc
//version 1.0
//
#ifndef _TYPECOMMON_H_
#define _TYOE_COMMON_H_

#ifndef UINT
#define UINT
typedef unsigned int uint;
#endif

#ifndef UCHAR
#define UCHAR
typedef unsigned char uchar;
#endif

#ifndef CUDACHECKERROR
#define CUDACHECKERROR
#define cudaCheckError() {                                          \
        cudaError_t e=cudaGetLastError();                                 \
        if(e!=cudaSuccess) {                                              \
                printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
                exit(EXIT_FAILURE);                                           \
        }                                                                 \
}
#endif

#endif

