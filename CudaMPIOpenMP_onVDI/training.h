#ifndef __functions_H
#define __functions_H
#pragma warning(disable: 4996)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <cuda.h>
#include<string.h>
#define INPUT_FILE_PATH "C:\\Users\\cudauser\\Downloads\\CudaMPIOpenMP_onVDI\\input.txt"
#define OUTPUT_FILE_PATH "C:\\Users\\cudauser\\Downloads\\CudaMPIOpenMP_onVDI\\output.txt"
#define INIT_WEIGHT_VALUE 0
#define MAX_SIZE_COORDINATES 20
typedef struct
{
	int label;
	float coordinates[MAX_SIZE_COORDINATES];

}Point;

typedef struct {
	bool thereIsAlpaha;
	float foundAlpha,q;
	float *weights;
	
}MyMinAlpaha;

typedef struct {
	bool successAllocate;
	Point retPoint;
	int numOfMissPoint;
}cudaVals;

void printWeights(float *weights, int dimSize);
void updateWeights(float *weights, int dimSize, Point lastPointChecked, int lastPointSign, float alpha);
int calculatePointSum(Point point, float *weights, int dimSize);
void calculateAlpahasArray(float* alphas, int numOfAlpahas, float minAlpaha);
cudaError_t checkAllPointsLabel(int numOfPoints, int dimSize, Point *points, float *weights, cudaVals* myCudaVals);
cudaError_t numberOfMissPoints(int numOfPoints, int dimSize, Point *points, float *weights, cudaVals* myCudaVals);
void initWeights(float * weights, int dimSize);
void findMinAlpaha(Point *allPoints, int pointsSize, MyMinAlpaha *minAlpaha, float * myAlphas, int alphasPerProccess, float qc, int dimSize,int limit);
bool train(MyMinAlpaha *minAlpaha, int pointsSize, int dimSize, Point* allPoints, int limit, float currentAlpaha, cudaVals* myCudaVals);
void freeMem(Point* allPoints, float *allAlphas, float *myAlphas);


#endif