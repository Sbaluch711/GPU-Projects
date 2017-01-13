
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

//kernal takes in 
__global__ void thresholdKernel(unsigned char* data, int size) {

	int threshold = 128;
	int j = blockIdx.x *blockDim.x + threadIdx.x;

	if(j < size){
		if (data[j] > threshold) {
			data[j] = 255;
		}
		else {
			data[j] = 0;
		}
	}
}

void threshold(int threshold, int width, int height, unsigned char* data);
bool initializeImageGPU(unsigned char* data, int width, int height, Mat image);

//time just the execution of the kernal

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	if (!image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}

	cvtColor(image, image, COLOR_RGB2GRAY);
	cout << "Hi";

	//threshold(128, image.rows, image.cols, image.data);

	if (initializeImageGPU(image.data, image.rows, image.cols, image)) {
		cout << "We worked with the GPU" << endl;
	}
	else {
		cout << "It failed." << endl;
	}

	namedWindow("Display Window", WINDOW_NORMAL);
	imshow("Display Window", image);

	waitKey(0);
	return 0;
}

void threshold(int threshold, int width, int height, unsigned char * data)
{
	for (int i = 0; i < height *width; i++) {
		if (data[i] > threshold) {
			data[i] = 255;
		}
		else{
			data[i] = 0;
		}
	}

}

bool initializeImageGPU(unsigned char * data, int width, int height, Mat image)
{
	bool temp;
	Mat* ImageOriginal = nullptr;
	Mat* ImageModified = nullptr;
	int size = width*height;

	cudaError_t cudaTest;

	cudaTest = cudaSetDevice(0);
	if (cudaTest != cudaSuccess) {
		cout << "Error with device" << endl;
	}
	else {
		cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageOriginal, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}
	else {
		cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageModified, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc2 failed!" << endl;
		temp = false;
	}
	else {
		cout << "suscsess" << endl;
	}

	cudaTest = cudaDeviceSynchronize();
	if(cudaTest != cudaSuccess) {
		cout << "cudaSync failed!" << endl;
		temp = false;
	}
	else {
		cout << "suscsess" << endl;
	}

	cudaTest = cudaMemcpy(ImageOriginal, image.data, size, cudaMemcpyHostToDevice);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy failed!" << endl;
		temp = false;
	}
	else {
		cout << "suscsess" << endl;
	}
	
	ImageModified = ImageOriginal;
	
	int blocksNeeded = size / 1024 + 1;

	thresholdKernel<<<blocksNeeded,1024>>>(ImageModified->data, ImageModified->rows *ImageOriginal->cols);

	cudaTest = cudaMemcpy(image.data, ImageModified->data, size, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
		temp = false;
	}

	return temp;
}

/*CudaThreshold
cudaMalloc the original image space
cudaMalloc the modified image space
cudaMemcpy image.data
kernel original writing modified
cudamemcpy modified back to host and display;*/