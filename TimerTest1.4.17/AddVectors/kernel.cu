
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <Windows.h>

using namespace std;

typedef int arrayType_t;

bool initalizeArray(arrayType_t**a, arrayType_t**b, arrayType_t**c, int size);
void cleanUpMem(arrayType_t* a, arrayType_t* b, arrayType_t* c);
bool fillTheArray(arrayType_t ** a, int size);


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

	cleanUpMem(a, b, c);


	// fill a and b with random numbers
	fillTheArray(&a, size);
	fillTheArray(&b, size);

	//fill c with 0;
	for (int i = 0; i < size; i++) {
		c[i] = 0;
	}

	if (initalizeArray(&a, &b, &c, size) == true) {
		cout << "YAY" << endl;
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

bool fillTheArray(arrayType_t ** a, int size)
{

	bool temp = true;
	int tempRandomNumber;
	for (int i = 0; i < size; i++) {
		tempRandomNumber = (rand() % 100);
		(*a)[i] = tempRandomNumber;
	}
	return temp;
}
