#ifndef __sharingDataWithMpi_H
#define __sharingDataWithMpi_H
#pragma warning(disable: 4996)
#include "training.h"
#include <mpi.h>

void readInitialValuesFromFile(FILE*, int*, int*, float*, float*, int*, float *);
void broadcastInitialValues(int*pointsSize, int*dimSize, int *alphasPerProccess, int*limit, float *qc);
int readIntFromFile(FILE* f);
float readFloatFromFile(FILE* f);
void readAllPoints(int numOfpoints, int dimSize, FILE* f, Point *points);
void sendLeftoverAlphas(int numprocs, float * allAlphas, int startPosOfLeftoverAlphas, int leftoverAlphasAmount);
void sendPointsArray(int pointsSize, Point *allPoints, int numprocs);
void recevieAllPoints(Point* allPoints, int pointsSize, MPI_Status* status);
void sendMyMinAlpaha(MyMinAlpaha,int);
void receiveAllAlpahas(MyMinAlpaha* minAlpahaArr,int numOfProccess, MPI_Status* status,int dimSize);
void outputNoAlpahaFound(char *);
void outputMinAlpaha(char *,MyMinAlpaha minAlpaha,int dimSize);
void minimalAlpahasBetweenProcesses(MyMinAlpaha*, int, int);
#endif