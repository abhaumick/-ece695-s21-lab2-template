

#ifndef CPU_LIB_H
#define CPU_LIB_H

	#include <iostream>
	#include <cstdlib>
	#include <ctime>
	#include <random>
    #include <iomanip>
	#include <chrono>
	#include <cstring>
	#include <cstdarg>
	#include <fstream>
	#include <vector>
	#include <algorithm>
	
	// Uncomment this to suppress console output
	// #define DEBUG_PRINT_DISABLE

	extern void dbprintf(const char* fmt...);


	extern void vectorInit(float* v, int size);
	extern int verifyVector(float* a, float* b, float* c, float scale, int size);
	extern void printVector(float* v, int size);
	
	extern void saxpy_cpu(float* x, float* y, float scale, uint64_t size);

	extern int runCpuSaxpy(uint64_t vectorSize);

	extern int runCpuMCPi(uint64_t iterationCount, uint64_t sampleSize);


	////    Lab 2    ////

	typedef struct ImageDim_t
	{
		uint32_t height;
		uint32_t width;
		uint32_t channels;
		uint32_t pixelSize;
	} ImageDim;

	extern std::ostream& operator<< (std::ostream &o,ImageDim imgDim);
	
	/**
	 * @brief 
	 * 
	 * @param bytesFilePath 
	 * @param imgDim 
	 * @param imgData 
	 * @return int 
	 */
	extern int loadBytesImage(std::string bytesFilePath, ImageDim &imgDim, uint8_t ** imgData);

	extern int writeBytesImage(std::string outPath, ImageDim &imgDim, uint8_t * outData);

	typedef struct MedianFilterArgs_t {
		uint32_t filterH;
		uint32_t filterW;
	} MedianFilterArgs;

	extern int medianFilter_cpu(uint8_t inPixels, ImageDim imgDim, 
		uint8_t outPixels, MedianFilterArgs args);

	extern int runCpuMedianFilter (std::string imgPath, std::string outPath, MedianFilterArgs args);

	enum class PoolOp{MaxPool, AvgPool, MinPool};
#endif
