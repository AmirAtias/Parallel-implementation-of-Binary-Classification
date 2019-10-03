
#include "training.h"
#ifndef __CUDACC__ 

#define __CUDACC__

#endif

#include <device_functions.h>
#include <cuda_runtime_api.h>

#define BLOCKS_PER_BATCH 1000
#define MAX_SIZE_OF_THREADS  1000

// free allocated memory
cudaError_t finalize(cudaError_t cudaStatus, Point *d_points, float *d_resultArray, float *resultArray, float* d_weights, cudaVals* myCudaVals)
{
	cudaFree(d_points);
	cudaFree(d_resultArray);
	cudaFree(d_weights);
	free(resultArray);
	myCudaVals->successAllocate = false;
	return cudaStatus;
}
cudaError_t finalizeForMissPoints(cudaError_t cudaStatus, Point* d_points, int * d_MissPoints, int * missPoints, float* d_weights, cudaVals* myCudaVals) {
	cudaFree(d_points);
	cudaFree(d_MissPoints);
	cudaFree(d_weights);
	free(missPoints);
	myCudaVals->successAllocate = false;
	return cudaStatus;
}


__global__ void sumArray(Point * d_points, float * d_weights, float *d_resultArray) {

	extern __shared__ float s_point[];
	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	s_point[tid] = d_points[bid].coordinates[tid] * d_weights[tid];

	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			s_point[tid] += s_point[tid + s];
		}
		__syncthreads();

	}

	if (tid == 0) {
		s_point[0] += d_weights[blockDim.x];
		d_resultArray[blockIdx.x] = s_point[0];

	}
}


cudaError_t checkAllPointsLabel(int numOfPoints, int dimSize, Point *points, float *weights, cudaVals* myCudaVals) {
	int batchSize = numOfPoints / BLOCKS_PER_BATCH;
	int lastIter = numOfPoints%BLOCKS_PER_BATCH;
	Point retPoint, *d_points = 0;
	int findWrongPoint = 0;
	if (lastIter != 0)
		batchSize++;
	int counter = 0, i = 0;
	int blocksPerIter;
	float *d_weights = 0, *d_resultArray = 0, *resultArray = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&d_weights, (dimSize + 1) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("malloc d_weights failed!\n");
		return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
	}
	cudaStatus = cudaMemcpy(d_weights, weights, (dimSize + 1) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy for d_weights failed!\n");
		return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
	}
	while (counter < batchSize && !findWrongPoint) {
		if (counter == batchSize - 1 && lastIter != 0) //last iteration
			blocksPerIter = lastIter;
		else
			blocksPerIter = BLOCKS_PER_BATCH;
		
		resultArray = (float *)malloc(blocksPerIter * sizeof(float));
		cudaStatus = cudaMalloc(&d_resultArray, blocksPerIter * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			printf(" cudaMalloc device result array failed!\n");
			return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
		}


		cudaStatus = cudaMalloc((void**)&d_points, blocksPerIter * sizeof(Point));
		if (cudaStatus != cudaSuccess) {
			printf("malloc d_points failed!\n");
			return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
		}
		cudaStatus = cudaMemcpy(d_points, &points[counter*BLOCKS_PER_BATCH], blocksPerIter * sizeof(Point), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			printf("memcopy d_points failed!\n");
			return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
		}
		sumArray << <blocksPerIter, dimSize, dimSize * sizeof(float) >> > (d_points, d_weights, d_resultArray);//need to check dimsize-1
		cudaStatus = cudaMemcpy(resultArray, d_resultArray, blocksPerIter * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("memcpy resultarray failed!\n");
			return finalize(cudaStatus, d_points, d_resultArray, resultArray, d_weights, myCudaVals);
		}
		for (i = 0; i < blocksPerIter; i++) {
			if (points[counter*BLOCKS_PER_BATCH + i].label == 1) {
				if (!(resultArray[i] > 0)) { //Miss
					retPoint = points[counter*BLOCKS_PER_BATCH + i];
					findWrongPoint = 1;
					break;
				}
			}
			else { //label==-1
				if (!(resultArray[i] < 0)) { //Miss
					retPoint = points[counter*BLOCKS_PER_BATCH + i];
					findWrongPoint = 1;
					break;
				}
			}
		}

		cudaFree(d_points);
		cudaFree(d_resultArray);
		free(resultArray);
		counter++;
	}
	cudaFree(d_weights);
	if (findWrongPoint) {
		memcpy(&(myCudaVals->retPoint), &retPoint, sizeof(Point));
		return cudaStatus;

	}
	else {
		memcpy(&(myCudaVals->retPoint), &(points[numOfPoints - 1]), sizeof(Point));
		return cudaStatus;
	}
}
__global__ void calculateSumOfCoordinatesAllPoints(Point* d_points, int dimSize, int* d_MissPoints, float * d_weights, int startLocation) {
	int i, thread_index = threadIdx.x;
	int d_startLocation;
	if ((startLocation) == 0) {//case remainderEqualToZero || first function call
		int block_index = blockIdx.x;
		d_startLocation = block_index * blockDim.x;
	}
	else
		d_startLocation = startLocation;
	int index = thread_index + (d_startLocation); //(*d_startLocation);
	float sumCoordinates = 0;
	for (i = 0; i < dimSize; i++)
		sumCoordinates += d_points[index].coordinates[i] * d_weights[i];
	sumCoordinates += d_weights[i];//bias
	if (d_points[index].label == 1) {
		if ((sumCoordinates > 0))
			d_MissPoints[index] = 0;
		else
			d_MissPoints[index] = 1;
	}
	else { //label==-1
		if ((sumCoordinates < 0))
			d_MissPoints[index] = 0;
		else
			d_MissPoints[index] = 1;
	}
}
__global__ void calculateSumOfMissPoints(int *d_MissPoints, int sizeOfthreads)
{
	int index = threadIdx.x *sizeOfthreads, i;
	for (i = index + 1; i < index + sizeOfthreads; i++)
		if (d_MissPoints[i] != 0)
			d_MissPoints[index]++;
}

cudaError_t numberOfMissPoints(int numOfPoints, int dimSize, Point *points, float *weights, cudaVals* myCudaVals) {
	float *d_weights = 0;
	cudaError_t cudaStatus;
	Point *d_points = 0;
	int* d_MissPoints = 0, *missPoints = 0;
	int numberofBlocks, numberOfThreads, remainderEqualToZero, startLocation = 0;
	if (numOfPoints / MAX_SIZE_OF_THREADS > 0) {// case number of point bigger then max Num of threads
		numberofBlocks = numOfPoints / MAX_SIZE_OF_THREADS;
		numberOfThreads = MAX_SIZE_OF_THREADS;
	}
	else {// number of points < max number of threads
		numberofBlocks = 1;
		numberOfThreads = numOfPoints;
	}
	if (numOfPoints%MAX_SIZE_OF_THREADS != 0 && numOfPoints / MAX_SIZE_OF_THREADS > 0)// check if there is remainder
		remainderEqualToZero = 0;
	else
		remainderEqualToZero = 1;

	cudaStatus = cudaMalloc(&d_weights, (dimSize + 1) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc for d_weights failed!\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}
	cudaStatus = cudaMemcpy(d_weights, weights, (dimSize + 1) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cuda memcpy for d_weights failed\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}

	missPoints = (int *)malloc(numOfPoints * sizeof(int));
	cudaStatus = cudaMalloc(&d_MissPoints, numOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cuda malloc for d_MissPoints failed\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}

	cudaStatus = cudaMalloc((void**)&d_points, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		printf("malloc d_points failed\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}
	cudaStatus = cudaMemcpy(d_points, points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("memcpy d_points failed\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}

	calculateSumOfCoordinatesAllPoints << <numberofBlocks, numberOfThreads >> > (d_points, dimSize, d_MissPoints, d_weights, startLocation);

	if (!remainderEqualToZero) {
		numberOfThreads = numOfPoints%MAX_SIZE_OF_THREADS;
		startLocation = numOfPoints / MAX_SIZE_OF_THREADS;
		startLocation *= MAX_SIZE_OF_THREADS;// startLocation contain the start position of remainder
		calculateSumOfCoordinatesAllPoints << <1, numberOfThreads >> > (d_points, dimSize, d_MissPoints, d_weights, startLocation);
	}
	if (numberofBlocks > 0)
		calculateSumOfMissPoints << <1, numberofBlocks >> > (d_MissPoints, numberOfThreads);
	cudaStatus = cudaMemcpy(missPoints, d_MissPoints, numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("memcpy d_MissPoints failed\n");
		return finalizeForMissPoints(cudaStatus, d_points, d_MissPoints, missPoints, d_weights, myCudaVals);
	}
	int retVal = 0, i;
	for (i = 0; i < numOfPoints; i += numberOfThreads) {
		retVal += missPoints[i];
	}
	if (!remainderEqualToZero) { //case there is remainder -need to calculate separately
		for (i = startLocation; i < numOfPoints; i++) {// startLocation contain the start position of remainder
			retVal += missPoints[i];
		}
	}
	free(missPoints);
	cudaFree(d_weights);
	cudaFree(d_points);
	cudaFree(d_MissPoints);
	myCudaVals->numOfMissPoint = retVal;
	return cudaStatus;
}




