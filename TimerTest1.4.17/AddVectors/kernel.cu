#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <Windows.h>
#include <vector>
#include <omp.h>

#include "../HighPerformanceTimer/HighPerformanceTimer.h"

using namespace std;

typedef int arrayType_t;

//for add kernal -- max num of blocks, max threads per blocks
__global__ void addKernel(arrayType_t*c, arrayType_t *a, arrayType_t*b, unsigned int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		c[i] = a[i] + b[i];
	}
}
void addTwoVectorsCPU(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size);
//initializes CPU array
bool initalizeArray(arrayType_t**a, arrayType_t**b, arrayType_t**c, int size);
//initializes GPU array
bool initializeGPUArray(arrayType_t *c, arrayType_t*a, arrayType_t*b, int size);
//frees the arrays
void cleanUpCPUMem(arrayType_t* a, arrayType_t* b, arrayType_t* c);
//frees the arrays
void cleanUpGPUMem(arrayType_t* a, arrayType_t* b, arrayType_t* c);
//fills GPU arrays
bool fillTheArray(arrayType_t * a, int size, bool isItC);
//calcs the average time
void averageTime(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size);

double calculateTimings(vector<double>times);

int main(int argc, char * argv[]) {

	int size = 20;

	srand(GetTickCount());

	arrayType_t* a;
	arrayType_t* b;
	arrayType_t* c;
	arrayType_t* dev_a = nullptr;
	arrayType_t* dev_b = nullptr;
	arrayType_t* dev_c = nullptr;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "Did not set device." << endl;
	}

	int blocksNeeded = size/1024 +1;
	int maxNumThreads = 1024;

	HighPrecisionTime timeForCPUVectorAdd;
	HighPrecisionTime timeForGPUVectorAdd;
	vector < double> timesOfGPU;
	vector<double> timesOfCPU;
	
	if (argc > 1) {
		size = stoi(argv[1]);
	}
	cout << argc << endl;
	cout << size << endl;

	//setup
	//initializes on CPU side
	for (int i = 0; i < size; i++) {
		if (initalizeArray(&a, &b, &c, size))
		{
			cleanUpCPUMem(a, b, c);
			// fill a and b with random number
			fillTheArray(a, size, false);
			fillTheArray(b, size, false);
			fillTheArray(c, size, true);
		
			//Actual timing
			timeForCPUVectorAdd.TimeSinceLastCall();
			addTwoVectorsCPU(a, b, c, size);
			timesOfCPU.push_back(timeForCPUVectorAdd.TimeSinceLastCall());

		}
	
		//initialize the memory on the gpu side
		if (initializeGPUArray(dev_c, dev_a, dev_b, size))
		{
			cout << "I initialized the gpu arrays." << endl;
			timeForGPUVectorAdd.TimeSinceLastCall();
			//copy to GPU
			cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(arrayType_t), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << "cudaMemcpy of A failed!" << endl;
			}
			cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(arrayType_t), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				cout << "cudaMemcpy B failed!" << endl;
			}

			//add
			addKernel << < blocksNeeded, maxNumThreads >> > (dev_c, dev_a, dev_b, size);

			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				cout << "cudaDeviceSynchronize returned error code %d after launching addKernel!" << endl;
			}

			//copy back c to the GPU
			cudaStatus = cudaMemcpy(c, dev_c, size*(sizeof(arrayType_t)), cudaMemcpyDeviceToHost);
			//deallocate the gpu memory

			timesOfGPU.push_back(timeForGPUVectorAdd.TimeSinceLastCall());
			cleanUpGPUMem(dev_c, dev_a, dev_b);
		}
		else 
		{
			cout << "initialize did not work" << endl;
		}
	}
		
	cout << "The Total CPU time for " << size << "runs " << calculateTimings(timesOfCPU) << endl;
	cout << "The Total GPU time for " << size << "runs " << calculateTimings(timesOfGPU) << endl;


	system("pause");
	return 0;
}

bool initalizeArray(arrayType_t** a, arrayType_t**b, arrayType_t** c, int size)
{
	bool temp = true;

	*a = (arrayType_t*)(malloc(size*(sizeof(arrayType_t))));
	*b = (arrayType_t*)(malloc(size*(sizeof(arrayType_t))));
	*c = (arrayType_t*)(malloc(size*(sizeof(arrayType_t))));

	//error check
	if (*a == NULL || *b == NULL || *c == NULL)
	{
		temp = false;
		cout << "We have encountered an issue during Allocation.";
	}
	else {
		cout << "It's all good in the cpu." << endl;
	}

	return temp;
}

bool initializeGPUArray(arrayType_t * c, arrayType_t * a, arrayType_t * b, int size)
{
	cudaError_t cudaStatus;
	bool temp = true;
	cudaStatus = cudaMalloc((void**)&c, size * sizeof(arrayType_t));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}

	cudaStatus = cudaMalloc((void**)&a, size * sizeof(arrayType_t));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;

	}

	cudaStatus = cudaMalloc((void**)&b, size * sizeof(arrayType_t));
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}
	return temp;
}

void cleanUpCPUMem(arrayType_t* a, arrayType_t* b, arrayType_t* c)
{
	a = b = c = nullptr;
	cout << "Memory has been deallocated." << endl;
}

void cleanUpGPUMem(arrayType_t * a, arrayType_t * b, arrayType_t * c)
{
	cudaFree(c);
	cudaFree(a);
	cudaFree(b);

}

bool fillTheArray(arrayType_t * a, int size, bool isItC)
{
	bool temp = true;
	int tempRandomNumber;
	#pragma omp parallel for
	if (!isItC) {
		for (int i = 0; i < size; i++) 
		{
		tempRandomNumber = (rand() % 100);
		a[i] = tempRandomNumber;
		//cout << tempRandomNumber << endl;
		}
	}
	else {
		for (int i = 0; i < size; i++)
		{
			a[i] = 0;
		}
	}
	return temp;
}


void averageTime(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size)
{
	HighPrecisionTime timetheVectorFill;
	vector < double> times;

	for (int i = 0; i < size; i++) {
		//time filling the arrays 100 times
		//Print average

		//so start timer in function
		//fill the arrays
		//end timer
		//push number into vector
		//add all the numbers and divide by size
		
		timetheVectorFill.TimeSinceLastCall();

		addTwoVectorsCPU(a, b, c, size);
		times.push_back(timetheVectorFill.TimeSinceLastCall());
	}
	double temp;
	for (int j = 0; j < times.size(); j++) {
		temp = temp + times[j];
		cout << " Current time" << times[j] << endl;
	} 
	temp = temp / times.size();
	cout << "The average time is: " << temp << endl;
}

double calculateTimings(vector<double> times)
{
	double temp;
	for (int j = 0; j < times.size(); j++) {
		temp = temp + times[j];
	//	cout << " Current time" << times[j] << endl;
	}
	return temp;
}

void addTwoVectorsCPU(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size)
{
	//cout << "[" << endl;
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
		//cout << c[i] << ",";
	}
	//cout << "]" << endl;
	//cout << "arrays added." << endl;
}