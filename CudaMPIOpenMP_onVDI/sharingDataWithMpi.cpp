#include"sharingDataWithMpi.h"

int readIntFromFile(FILE* f) {
	int valFromFile;
	fscanf(f, "%d", &valFromFile);
	return 	valFromFile;

}
float readFloatFromFile(FILE* f) {
	float valFromFile;
	fscanf(f, "%f", &valFromFile);
	return 	valFromFile;

}
void readInitialValuesFromFile(FILE* f, int*pointsSize, int*dimSize, float*minAlpha, float*maxAlpha, int*limit, float *qc) {
	*pointsSize = readIntFromFile(f);
	*dimSize = readIntFromFile(f);
	*minAlpha = readFloatFromFile(f);
	*maxAlpha = readFloatFromFile(f);
	*limit = readIntFromFile(f);
	*qc = readFloatFromFile(f);
}

void broadcastInitialValues(int*pointsSize, int*dimSize, int *alphasPerProccess, int*limit, float *qc) {
	MPI_Bcast(pointsSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(dimSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(alphasPerProccess, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(qc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void readAllPoints(int numOfpoints, int dimSize, FILE* f, Point *points) {
	int i, j;
	for (i = 0; i < numOfpoints; i++) {
		for (j = 0; j < dimSize; j++) {
			fscanf(f, "%f", &points[i].coordinates[j]);
		}
		fscanf(f, "%d", &points[i].label);
	}

}

void sendLeftoverAlphas(int numprocs, float * allAlphas, int startPosOfLeftoverAlphas, int leftoverAlphasAmount) {
	for (int i = 0; i < numprocs - 1; i++) {
		if (leftoverAlphasAmount > 0) {
			MPI_Send(&allAlphas[startPosOfLeftoverAlphas + i], 1, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD);
			leftoverAlphasAmount--;
		}
		else //no more leftover alphas - send 0
			MPI_Send(&leftoverAlphasAmount, 1, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD);
	}
}


void sendPointsArray(int pointsSize, Point *allPoints, int numprocs) {
	for (int i = 0; i < pointsSize; i++) {
		for (int j = 0; j < numprocs - 1; j++) {
			MPI_Send(&(allPoints[i].label), 1, MPI_INT, j + 1, 0, MPI_COMM_WORLD);
			MPI_Send(allPoints[i].coordinates, MAX_SIZE_COORDINATES, MPI_FLOAT, j + 1, 0, MPI_COMM_WORLD);
		}
	}
}
void recevieAllPoints(Point* allPoints, int pointsSize, MPI_Status* status) {
	for (int i = 0; i < pointsSize; i++) {
		MPI_Recv(&(allPoints[i].label), 1, MPI_INT, 0, 0, MPI_COMM_WORLD, status);
		MPI_Recv(allPoints[i].coordinates, MAX_SIZE_COORDINATES, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, status);
	}
}
void sendMyMinAlpaha(MyMinAlpaha minAlpaha, int dimSize) {
	MPI_Send(&(minAlpaha.foundAlpha), 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	MPI_Send(&(minAlpaha.q), 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	MPI_Send(&(minAlpaha.thereIsAlpaha), 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD);
	MPI_Send(minAlpaha.weights, dimSize + 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
}
void receiveAllAlpahas(MyMinAlpaha* minAlpahaArr, int numOfProccess, MPI_Status* status, int dimSize) {
	int i;
	for (i = 1; i < numOfProccess; i++) {
		MPI_Recv(&(minAlpahaArr[i].foundAlpha), 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, status);
		MPI_Recv(&(minAlpahaArr[i].q), 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, status);
		MPI_Recv(&(minAlpahaArr[i].thereIsAlpaha), 1, MPI_C_BOOL, i, 0, MPI_COMM_WORLD, status);
		MPI_Recv(&(minAlpahaArr[i].weights), dimSize + 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, status);

	}
}

void outputMinAlpaha(char * file_path, MyMinAlpaha minAlpaha, int dimSize) {

	FILE * outputFile = fopen(file_path, "w");
	if (outputFile == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(outputFile, "Alpha minimum = %lf  q = %lf\n", minAlpaha.foundAlpha, minAlpaha.q);
	for (int i = 0; i <= dimSize; i++)
		fprintf(outputFile, "w%d) %.9f\n", i + 1, minAlpaha.weights[i]);	fclose(outputFile);
}



void outputNoAlpahaFound(char * file_path) {
	FILE * outputFile = fopen(file_path, "w");
	if (outputFile == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	fprintf(outputFile, "Alpha not FOUND!\n");
	fclose(outputFile);
}


void minimalAlpahasBetweenProcesses(MyMinAlpaha*minAlpahaArr, int dimSize, int numOfProcsess) {
	int indexOfMinAlpaha = -1, i;
	for (i = 0; i < numOfProcsess; i++) {
		if (minAlpahaArr[i].thereIsAlpaha == true) {
			if (indexOfMinAlpaha == -1)
				indexOfMinAlpaha = i;
			else {
				if (minAlpahaArr[i].foundAlpha < minAlpahaArr[indexOfMinAlpaha].foundAlpha)
					indexOfMinAlpaha = i;
			}
		}
	}
	if (indexOfMinAlpaha == -1)
		outputNoAlpahaFound(OUTPUT_FILE_PATH);
	else
		outputMinAlpaha(OUTPUT_FILE_PATH, minAlpahaArr[indexOfMinAlpaha], dimSize);
}

