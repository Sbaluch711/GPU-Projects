
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "HPT.h"


using namespace cv;
using namespace std;

int trackbarSize;
Mat image;
unsigned char* ImageModified;

//kernal takes in two arrays and size
__global__ void thresholdKernel(unsigned char* data, unsigned char* data2, int size, int thresholdSlider) {


	int j = (blockIdx.x *blockDim.x) + threadIdx.x;

	if(j < size){
		if (data[j] > thresholdSlider) {
			data2[j] = 255;
		}
		else {
			data2[j] = 0;
		}
	}
}
//threshold change in cpu
void threshold(int threshold, int width, int height, unsigned char* data);
//threshold change in gpu
bool initializeImageGPU(int width, int height, Mat image);
//creates trackbar for image
void on_trackbar(Mat Image, unsigned char* data2, int size, int threshold_slider);

int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}


	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	if (!image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}

	cvtColor(image, image, COLOR_RGB2GRAY);


	threshold(128, image.rows, image.cols, image.data);


	if (initializeImageGPU(image.rows, image.cols, image)) {
		cout << "We worked with the GPU" << endl;
	}
	else {
		cout << "It failed." << endl;
	}

	namedWindow("Display Window", WINDOW_NORMAL);
	//createTrackbar("Threshold", "Display Window", &threshold_slider, THRESHOLD_SLIDER_MAX, on_tracker(int, void *, Image, unsigned char* data2, size,threshold_slider));
	imshow("Display Window", image);

	waitKey(0);
	return 0;
}

void threshold(int threshold, int width, int height, unsigned char * data)
{
	HighPrecisionTime timeTheModification;
	double currentTime;
	timeTheModification.TimeSinceLastCall();
	for (int i = 0; i < height *width; i++) {
		if (data[i] > threshold) {
			data[i] = 255;
		}
		else{
			data[i] = 0;
		}
	}
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "CPU: " << currentTime << endl;
}

bool initializeImageGPU(int width, int height, Mat image)
{
	HighPrecisionTime timeTheModification;
	double currentTime;

	bool temp = true;
	unsigned char* ImageOriginal = nullptr;
	ImageModified = nullptr;
	int size = width*height * sizeof(char);
	trackbarSize = size;

	cudaError_t cudaTest;

	cudaTest = cudaSetDevice(0);
	if (cudaTest != cudaSuccess) {
		cout << "Error with device" << endl;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageOriginal, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageModified, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc2 failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaDeviceSynchronize();
	if(cudaTest != cudaSuccess) {
		cout << "cudaSync failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMemcpy(ImageOriginal, image.data, size, cudaMemcpyHostToDevice);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}
	
	int blocksNeeded = (size+ 1023) / 1024;

	timeTheModification.TimeSinceLastCall();
	thresholdKernel<<<blocksNeeded,1024>>>(ImageOriginal, ImageModified, size,128);
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "GPU: " << currentTime << endl;
	
	cudaTest = cudaMemcpy(image.data, ImageModified, size, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
		temp = false;
	}

	int thresholdSlider = 50;
	namedWindow("Display Window", WINDOW_NORMAL);
	createTrackbar("Threshold", "Display Window", &thresholdSlider, 255, on_trackbar);
	on_trackbar(thresholdSlider, 0);
	waitKey(0);

	return temp;
}

void on_trackbar(int thresholdSlider, void*)
{
	HighPrecisionTime T;
	double currentTime;
	int blocksNeeded = (trackbarSize + 1023) / 1024;
	cudaDeviceSynchronize();
	
	T.TimeSinceLastCall();
	thresholdKernel << < blocksNeeded, 1024 >> > (image.data, ImageModified, size, thresholdSlider);
	currentTime = T.TimeSinceLastCall();
	cout << "CurrentTime: " << currentTime << endl;

	if (cudaMemcpy(image.data, ImageModified, trackbarSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Error copying." << endl;
	}
}

/*CudaThreshold
cudaMalloc the original image space
cudaMalloc the modified image space
cudaMemcpy image.data
kernel original writing modified
cudamemcpy modified back to host and display;*/