
#include "sharingDataWithMpi.h"

int main(int argc, char *argv[])
{
	//init mpi
	int numOfProcesses, myid;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);
	MPI_Status status;
	int pointsSize = 0, alphasPerProccess = 0, limit = 0, dimSize = 0;
	float *allAlphas, *myAlphas, myLeftoverAlpha = 0, qc = 0;
	Point *allPoints;
	MyMinAlpaha minAlpaha;
	// first,process 0 read from  input file  and send data to all process

	if (myid == 0) {
		float minAlpha = 0, maxAlpha = 0;
		//change INPUT_FILE_PATH according to  input file location
		FILE* inputFile = fopen(INPUT_FILE_PATH, "r");
		readInitialValuesFromFile(inputFile, &pointsSize, &dimSize, &minAlpha, &maxAlpha, &limit, &qc);
		allPoints = (Point*)calloc(pointsSize, sizeof(Point));
		readAllPoints(pointsSize, dimSize, inputFile, allPoints);
		fclose(inputFile);
		//calculate alphas according to values from  file
		int numOfAlphas = int(((maxAlpha - minAlpha) / minAlpha) + 1);
		alphasPerProccess = numOfAlphas / numOfProcesses;
		int leftoverAlphasAmount = numOfAlphas % numOfProcesses;
		allAlphas = (float*)malloc(numOfAlphas * sizeof(float));
		calculateAlpahasArray(allAlphas, numOfAlphas, minAlpha);
		//  divide  leftover alphas for all process except process 0
		int startPosOfLeftoverAlphas = alphasPerProccess * numOfProcesses;
		sendLeftoverAlphas(numOfProcesses, allAlphas, startPosOfLeftoverAlphas, leftoverAlphasAmount);
		// sending inital values for all process
		broadcastInitialValues(&pointsSize, &dimSize, &alphasPerProccess, &limit, &qc);
		sendPointsArray(pointsSize, allPoints, numOfProcesses);

	}

	else {// myid !=0
		MPI_Recv(&myLeftoverAlpha, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		// get inital values from process 0
		broadcastInitialValues(&pointsSize, &dimSize, &alphasPerProccess, &limit, &qc);
		allPoints = (Point*)calloc(pointsSize, sizeof(Point));
		recevieAllPoints(allPoints, pointsSize, &status);
	}
	//all process

	//divide alpha array between processes, if process has leftover alpha allocate alphasPerProccess + 1
	if (myLeftoverAlpha != 0)
		myAlphas = (float*)calloc(alphasPerProccess + 1, sizeof(float));
	else
		myAlphas = (float*)calloc(alphasPerProccess, sizeof(float));

	//process 0 send batch of alphas to all 
	MPI_Scatter(allAlphas, alphasPerProccess, MPI_FLOAT, myAlphas, alphasPerProccess, MPI_FLOAT, 0, MPI_COMM_WORLD);
	// Assign leftover alpha
	if (myLeftoverAlpha != 0) {
		myAlphas[alphasPerProccess] = myLeftoverAlpha;
		alphasPerProccess++;
	}

	minAlpaha.weights = (float*)calloc(dimSize + 1, sizeof(float));
	minAlpaha.q = 0, minAlpaha.thereIsAlpaha = false, minAlpaha.foundAlpha = 0;
	findMinAlpaha(allPoints, pointsSize, &minAlpaha, myAlphas, alphasPerProccess, qc, dimSize, limit);
	if (myid != 0)  //each proccess send his min alpaha	
		sendMyMinAlpaha(minAlpaha, dimSize);
	else {// process 0
		MyMinAlpaha* minAlpahaArr = (MyMinAlpaha*)calloc(numOfProcesses, sizeof(MyMinAlpaha));
		minAlpahaArr[0] = minAlpaha;// min alpaha of process 0
		receiveAllAlpahas(minAlpahaArr, numOfProcesses, &status, dimSize);
		minimalAlpahasBetweenProcesses(minAlpahaArr, dimSize, numOfProcesses);
		free(minAlpahaArr);
	}
	MPI_Finalize();
	return 0; // end the program
}






