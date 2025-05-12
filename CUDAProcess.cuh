#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <device_atomic_functions.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <opencv2/opencv.hpp>


struct AffineMat
{
	float v0, v1, v2;
	float v3, v4, v5;
};

#define BLOCK_SIZE 16
enum class ColorMode { RGB, GRAY };

void bgr2rgbDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, cudaStream_t stream);

void normDevice(const int& batchSize, uint8_t* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight,
	float mean0, float mean1, float mean2, float std0, float std1, float std2, cudaStream_t stream);

void hwc2chwDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, cudaStream_t stream);

void normDevice(const int& batchSize, float* dst, uint8_t* src, int srcWidth, int srcHeight,
	float mean0, float mean1, float mean2, float std0, float std1, float std2, cudaStream_t stream);

void normDevice(const int& batchSize, float* dst, float* src, int srcWidth, int srcHeight,
	float mean0, float mean1, float mean2, float std0, float std1, float std2, cudaStream_t stream);

void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, float paddingValue, AffineMat matrix);

void resizeDevice(const int& batchSize, unsigned char* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, float paddingValue, AffineMat matrix);

void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, ColorMode mode, AffineMat matrix);

void cutImageDevice(uint8_t* inputImage, uint8_t* dst, int width, int height, int subWidth, int subHeight, cudaStream_t stream);

void decodeDevice(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight, int num_class, float conf_thresh);

void transposeDevice(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight);

void nmsDeviceV1(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea);

void nmsDeviceV2(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea, int* idx, float* conf, float iou_thresh);

void decodeInsSegDevice(int batch_size, float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight, int num_class, float conf_thresh);
