
#include "training.h"

void calculateAlpahasArray(float* alphas, int numOfAlphas, float minAlpaha) {
	float a = minAlpaha;
	for (int i = 0; i < numOfAlphas; i++) {
		alphas[i] = a;
		a += minAlpaha;
	}
}

void updateWeights(float *weights, int dimSize, Point point, int lastPointSign, float alpha) {
	float updateWeightVal = alpha * lastPointSign;
	int i;
#pragma omp parallel for 
	for (i = 0; i < dimSize; i++) {
		weights[i] += updateWeightVal * point.coordinates[i];
	}
	weights[dimSize] += updateWeightVal;

}
void initWeights(float * weights, int dimSize) {
	int i;
	for (i = 0; i < dimSize; i++)
		weights[i] = INIT_WEIGHT_VALUE;
}



int calculatePointSum(Point point, float *weights, int dimSize) {
	float  functionSum = 0;
#pragma omp parallel reduction (+: functionSum)
	for (int i = 0; i < dimSize; i++) {
		functionSum += weights[i] * point.coordinates[i];
	}
	functionSum += weights[dimSize];
	if (functionSum > 0)
		return 1;
	else if (functionSum < 0)
		return -1;
	else
		return 0;

}

bool train(MyMinAlpaha *minAlpaha, int pointsSize, int dimSize, Point* allPoints, int limit, float currentAlpaha, cudaVals* myCudaVals) {
	bool isAllClassified = false;
	Point lastPointChecked;
	int signForCheck, i = 0;
	while (isAllClassified == false && i < limit) {
		checkAllPointsLabel(pointsSize, dimSize, allPoints, minAlpaha->weights, myCudaVals); //cuda
		if (myCudaVals->successAllocate == false)// allocate gpu dosen't success
			break;
		lastPointChecked = myCudaVals->retPoint;
		signForCheck = calculatePointSum(lastPointChecked, minAlpaha->weights, dimSize); // OpenMP
		if (signForCheck == lastPointChecked.label) {
			isAllClassified = true;
			break;
		}
		else {
			if (signForCheck == 0)//first iteration
				signForCheck = lastPointChecked.label * -1;
			signForCheck *= -1;
			updateWeights(minAlpaha->weights, dimSize, lastPointChecked, signForCheck, currentAlpaha);// OpenMP
		}

		i++;
	}
	return isAllClassified;
}


void findMinAlpaha(Point *allPoints, int pointsSize, MyMinAlpaha *minAlpaha, float * myAlphas, int alphasPerProccess, float qc, int dimSize, int limit) {
	int numOfMissPoints, i = 0;
	float q;
	bool isAllClassified;
	cudaVals myCudaVals;
	myCudaVals.successAllocate = true;
	while (i < alphasPerProccess) {
		initWeights(minAlpaha->weights, dimSize + 1);
		isAllClassified = train(minAlpaha, pointsSize, dimSize, allPoints, limit, myAlphas[i], &myCudaVals);
		if (myCudaVals.successAllocate == false)//return in case allocate doesnt success
			break;
		if (isAllClassified == false) {
			numberOfMissPoints(pointsSize, dimSize, allPoints, minAlpaha->weights, &myCudaVals);
			if (myCudaVals.successAllocate == false)//return in case allocate doesnt success
				break;
			numOfMissPoints = myCudaVals.numOfMissPoint;
		}
		else numOfMissPoints = 0;
		q = numOfMissPoints / (float)pointsSize; // divide return float
		if (q < qc) {
			minAlpaha->foundAlpha = myAlphas[i];
			minAlpaha->thereIsAlpaha = true;
			minAlpaha->q = q;
			break;
		}
		i++;
	}
}



void printWeights(float *weights, int dimSize) {
	for (int i = 0; i < dimSize + 1; i++) {
		printf("%f\n", weights[i]);
	}
}

void freeMem(Point* allPoints, float *allAlphas, float *myAlphas) {
	free(allPoints);
	free(allAlphas);
	free(myAlphas);
}


