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

bool initalizeArray(arrayType_t**a, arrayType_t**b, arrayType_t**c, int size);
void cleanUpMem(arrayType_t* a, arrayType_t* b, arrayType_t* c);
bool fillTheArray(arrayType_t * a, int size);
void averageTime(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size);
void addTwoVectors(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size);

int main(int argc, char * argv[]) {

	srand(GetTickCount());

	int size = 20;
	if (argc > 1) {
		size = stoi(argv[1]);
	}
	cout << argc << endl;
	cout << size << endl;

	arrayType_t* a;
	arrayType_t* b;
	arrayType_t* c;

	if (initalizeArray(&a, &b, &c, size)) {
		cleanUpMem(a, b, c);
		// fill a and b with random numbers

		averageTime(a, b, c, size);
	}

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
		cout << "It's all good." << endl;
	}

	return temp;
}

void cleanUpMem(arrayType_t* a, arrayType_t* b, arrayType_t* c)
{
	a = b = c = nullptr;
	cout << "Memory has been deallocated." << endl;
}

bool fillTheArray(arrayType_t * a, int size)
{
	bool temp = true;
	int tempRandomNumber;
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		tempRandomNumber = (rand() % 100);
		a[i] = tempRandomNumber;
		//cout << tempRandomNumber << endl;
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
		
		fillTheArray(a, size);
		fillTheArray(b, size);

		//fill c with 0;
		for (int i = 0; i < size; i++) {
			c[i] = 0;
		}
		
		timetheVectorFill.TimeSinceLastCall();

		addTwoVectors(a, b, c, size);
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

void addTwoVectors(arrayType_t * a, arrayType_t * b, arrayType_t * c, int size)
{
	//cout << "[" << endl;
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
		//cout << c[i] << ",";
	}
	//cout << "]" << endl;
	//cout << "arrays added." << endl;
}