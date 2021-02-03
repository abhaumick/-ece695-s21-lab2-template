
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, int scale, int size) {
	//	TODO: replaced with full 3D thread block computations
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < size) {
		y[threadId] = scale * x[threadId] + y[threadId];
	}
}

int runGpuSaxpy(int vectorSize) {

	uint64_t vectorBytes = vectorSize * sizeof(float);
	
	printf("Hello GPU Saxpy!\n");
	std::srand(std::time(0));
	
	float * h_x, * h_y, * h_z;

	h_x = (float *) malloc(vectorSize * sizeof(float));
	h_y = (float *) malloc(vectorSize * sizeof(float));
	h_z = (float *) malloc(vectorSize * sizeof(float));

	if (h_x == NULL || h_y == NULL || h_z == NULL) {
		std::cerr << "Unable to malloc memory ... Exiting!\n";
		exit(0);
	}
	
	vectorInit(h_x, vectorSize);
	vectorInit(h_y, vectorSize);
	float scale = 2.0f;
	
	float * d_x, * d_y;

	cudaDeviceReset();
	cudaError_t cudaStatus;
	int deviceCount = 0;

	cudaStatus = cudaGetDeviceCount ( &deviceCount ); 
	if (deviceCount == 0) {
		std::cerr << "No CUDA Devices found!\n";
		return 0;
	}
	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaSetDevice failed!\n";
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&d_x, vectorSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMalloc failed!\n";
		return 0;
	}
	cudaStatus = cudaMalloc((void**)&d_y, vectorSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaMalloc failed!\n";
		return 0;
	}

	//printVector(h_x, vectorSize);
	//printVector(h_y, vectorSize);

	cudaMemcpy(d_x, h_x, vectorBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, vectorBytes, cudaMemcpyHostToDevice);

	int threadDim = 1024;
	int blockDim = ((vectorSize - 1) / threadDim) + 1;

	std::cout << "Launching <<< " << blockDim << ", " << threadDim << " >>> kernel\n";

	saxpy_gpu <<< blockDim, threadDim >>> (d_x, d_y, scale, vectorSize);
	cudaDeviceSynchronize();

	cudaMemcpy(h_z, d_y, vectorBytes, cudaMemcpyDeviceToHost);

	int errorCount = verifyVector(h_x, h_y, h_z, scale, vectorSize);

	std::cout << "Found " << errorCount << " / " << vectorSize << " errors in vector! \n";

	cudaFree(d_x);
	cudaFree(d_y);

	free(h_x);
	free(h_y);
	free(h_z);
	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	// Find unique thread id
	uint64_t threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// Setup RNG
	curandState_t rng;
	curand_init(clock64(), threadId, 0, &rng);

	// Init counters
	uint64_t hitCount = 0;
	float x = 0.0f, y = 0.0f;

	if (threadId < pSumSize) {
		// Generate points & compute probability
		for (uint64_t iter = 0; iter < sampleSize; ++iter) {
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			if ( int(x * x + y * y) == 0 ) {
				++ hitCount;
			}
		}

		//  Write out results to memory
		pSums[threadId] = hitCount;
	}
}

/**
* @brief Optional GPU kernel to reduce a set of partial sums into a smaller set
*			by summing a subset into a single value
* 
* @param pSums 
* @param totals 
* @param pSumSize 
* @param reduceSize 
* @return void 
*/
__global__ void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	uint64_t tSum = 0;
	uint64_t threadId = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t arrayOffset = threadId * reduceSize;

	for (uint64_t idx = 0; idx < reduceSize; ++ idx) {
		if (arrayOffset + idx < pSumSize)
			tSum += pSums[arrayOffset + idx];
	}

	totals[threadId] = tSum;
}

/**
 * @brief Entrypoint for GPU Monte-Carlo estimation of Pi
 * 
 * @param generateThreadCount 	uint64_t	total number of generate threads	
 * @param sampleSize 			uint64_t	sample of points evaluated by each thread
 * @param reduceThreadCount 	uint64_t	number of reduction threads
 * @param reduceSize 			uint64_t	number of pSums summed by each reduce thread
 * @return int 
 */
int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}


/**
 * @brief main body for Monte-Carlo Pi estimation
 * 
 * @param generateThreadCount 	uint64_t	total number of generate threads	
 * @param sampleSize 			uint64_t	sample of points evaluated by each thread
 * @param reduceThreadCount 	uint64_t	number of reduction threads
 * @param reduceSize 			uint64_t	number of pSums summed by each reduce thread
 * @return double 	approx value of pi
 */
double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	uint64_t * hTSums, * hPSums;
	uint64_t * dPSums, * dTSums;

	reduceThreadCount = std::ceil(generateThreadCount / reduceSize);
	
	hPSums = new uint64_t[generateThreadCount];
	hTSums = new uint64_t[reduceThreadCount];

	//	Get CUDA Device Details
	int deviceId;
	cudaDeviceProp deviceProp;
	cudaGetDevice(&deviceId);
	cudaGetDeviceProperties (&deviceProp, deviceId);

	cudaDeviceReset();
	cudaMalloc(&dPSums, generateThreadCount * sizeof(uint64_t));
	cudaMalloc(&dTSums, reduceThreadCount * sizeof(uint64_t));


	uint64_t blockDim = std::min(generateThreadCount, (uint64_t)deviceProp.maxThreadsPerBlock);
	uint64_t gridDim = ((generateThreadCount - 1) / deviceProp.maxThreadsPerBlock) + 1;
	
	#ifndef DEBUG_PRINT_DISABLE
		printf("Launching kernel <<< %d, %d >>> \n", gridDim, blockDim);
	#endif

	generatePoints <<<gridDim, blockDim>>> (dPSums, generateThreadCount, sampleSize);
	
	gpuErrchk( cudaPeekAtLastError() );

	blockDim = std::min(reduceThreadCount, (uint64_t)deviceProp.maxThreadsPerBlock);
	gridDim = ((reduceThreadCount - 1) / deviceProp.maxThreadsPerBlock) + 1;
	printf("Launching kernel <<< %d, %d >>> \n", gridDim, blockDim);
	reduceCounts <<<gridDim, blockDim>>> (dPSums, dTSums, generateThreadCount, reduceSize);

	cudaMemcpy(hPSums, dPSums, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(hTSums, dTSums, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	gpuErrchk( cudaPeekAtLastError() );

	cudaFree(dTSums);
	cudaFree(dPSums);
	
	uint64_t totalHitCount = 0;
	for (int idx = 0; idx < reduceThreadCount; ++idx) {
		//	Each count is #hits out of reduceSize * sampleSize
		totalHitCount += hTSums[idx];
	}
	std::cout << std::setprecision(10);
	//std::cout << "Total Hits = " << totalHitCount << " / " << (generateThreadCount * sampleSize) << " \n";

	double approxPi = ( ((double)totalHitCount / generateThreadCount) / sampleSize );
	// Adjust for quarter circle
	approxPi = approxPi * 4.0;
	return approxPi;
}
