
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <vector>
#include <list>

#include <chrono>

#define BLOCK_SIZE 1024
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define G2ID(i) (i + maxV)
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define PASSCOUNT 100
#define EPSILON 0.01

const int verbose = 1;


inline void PS(std::string state){
    static int count = 1;
    if(verbose) std::cout<< "\u001b[42m\u001b[37;1m\u001b[1m "<<count<<" \u001b[0m  "<< state << std::endl;
}

__global__ void vinit(int VectorSize, float *vector, float value)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize)
    {
        vector[i] = 1.0;
    }
}

__global__ void yaxb(int VectorSize, float *Vector, float *Output, int b)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize)
    {
        Output[i] = b * Vector[i];
    }
}

/// @brief Element-Wise multipication of two arrays
/// @param VectorSize size of the tensor
/// @param aVector in-vector-1
/// @param bVector in-vector-2
/// @param Output output-vecotr
/// @return
__global__ void yaxbx(int VectorSize, float *aVector, float *bVector, float *Output)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize)
    {

        Output[i] = bVector[i] * aVector[i];
        // printf("O%f A%f B%f\n",Output[i],aVector[i],bVector[i]);
    }
}

__global__ void yambx(int VectorSize, float *aMatrix, float *bVector, float *OutpuotMatrix)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize * VectorSize)
    {
        OutpuotMatrix[i] = aMatrix[i] * bVector[i / VectorSize];
        // printf("O: %f A: %d B: %d\n",OutpuotMatrix[i],i,i/VectorSize);
    }
}

__global__ void yambxt(int VectorSize, float *aMatrix, float *bVector, float *OutpuotMatrix)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize * VectorSize)
    {
        OutpuotMatrix[i] = aMatrix[i] * bVector[i % VectorSize];
        // printf("O: %f A: %d B: %d\n",OutpuotMatrix[i],i,i/VectorSize);
    }
}

// remove ones
__global__ void r1s(int VectorSize, float *aMatrix, float *OutpuotMatrix)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < VectorSize * VectorSize)
    {
        OutpuotMatrix[i] = ceilf(((aMatrix[i] - 1) / 2) - EPSILON);
    }
}

__global__ void initial_balance_pass(const int graphSize, const float ratio, float *random, float *side)
{

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < graphSize)
    {

        if (random[i] < ratio)
            side[i] = -1;
        else
            side[i] = 1;
    }
}

// kernel to move each element
__global__ void move(float *sideGPU, float *lock, int maxIndex, float *sideGPUH)
{

    sideGPU[1] = -1 - 0;
    lock[maxIndex] = -1000000;

    // printf("lock: %f\n",lock[maxIndex]);
}

cudaError_t FM(int graphSize, float *graph, float *side, float ratio)
{

    // Setup Cuda Libraries
    cublasHandle_t handle;
    curandGenerator_t curand;
    cublasCreate(&handle);

    cudaError_t cudaStatus;

    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, (unsigned long long)clock());


    PS("Initial balncing");
    // init random spliter
    float *initial_balancer_vector;
    cudaMalloc((void **)&initial_balancer_vector, sizeof(float) * graphSize);
    curandGenerateUniform(curand, initial_balancer_vector, graphSize * sizeof(float));

    cudaDeviceSynchronize();

    // allocate Lock Array
    float *lockArray;
    cudaMalloc((void **)&lockArray, sizeof(float) * graphSize);
    

    int blockSize = BLOCK_SIZE;
    int gridSize = std::ceil(static_cast<float>(graphSize) / blockSize);

    float *sideGPU;
    cudaMalloc((void **)&sideGPU, sizeof(float) * graphSize);
    // std::cout << gridSize << "\t" << blockSize << std::endl;
    float *sideGPUH;
    cudaMalloc((void **)&sideGPUH, sizeof(float) * graphSize);
    // std::cout << gridSize << "\t" << blockSize << std::endl;

    initial_balance_pass<<<gridSize, blockSize>>>(graphSize, ratio, initial_balancer_vector, sideGPU);

    float *hold = new float[graphSize];
    cudaDeviceSynchronize();

    // check initial ratio
    cudaMemcpy(hold, sideGPU, sizeof(float) * graphSize, cudaMemcpyDeviceToHost);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cudaStatus);
    };

    float ratioCheck = 0;
    for (int j = 0; j < graphSize; j++)
    {
        if (hold[j] == -1)
            ratioCheck++;
    }
    std::cout << "initial ratio: " << ratioCheck << std::endl;

    free(hold);

    int passCount = PASSCOUNT;
    // Alocate & Move Graph
    float *graphGPU;
    cudaMalloc((void **)&graphGPU, sizeof(float) * graphSize * graphSize);

    cudaMemcpy(graphGPU, graph, sizeof(float) * graphSize * graphSize, cudaMemcpyHostToDevice);

    hold = new float[graphSize * graphSize];

    float *gain, *gainGPU;
    gain = new float[graphSize];
    
    // alocate gain
    cudaMalloc((void **)&gainGPU, sizeof(float) * graphSize);
    cudaMemset(gainGPU, 0, sizeof(float) * graphSize);

    // alocate gainHold
    float *gainGPUH;
    cudaMalloc((void **)&gainGPUH, sizeof(float) * graphSize);

    float *graphGPUH;
    cudaMalloc((void **)&graphGPUH, sizeof(float) * graphSize * graphSize);
    float *graphGPUH2;
    cudaMalloc((void **)&graphGPUH2, sizeof(float) * graphSize * graphSize);
    float *lockHold = new float[graphSize];
    // alocate lockXGain
    float *lockGainGPU;
    cudaMalloc((void **)&lockGainGPU, sizeof(float) * graphSize);

    vinit<<<gridSize, blockSize>>>(graphSize, lockArray, 1);
    cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };

    cudaMemcpy(gain, lockArray, sizeof(float) * graphSize, cudaMemcpyDeviceToHost);

        for(int l=0;l<graphSize;l++) printf("lockHold-%d:\t%d\n",l,lockHold[l]);



    
    float cut;
    // NormalPass

    int *minCut, *minCutGPU;
    cudaMalloc((void **)&minCutGPU, sizeof(int));
    cudaMemset(minCutGPU, graphSize, sizeof(int));
    int maxGainIndex = 0;
    for (int i = 0; i < passCount; i++)
    {

        float alpha = -1.0f;
        float beta = 0.0f;
        // cublasSscal(handle,graphSize*graphSize,&alpha,graphGPUH,1);
        // cudaDeviceSynchronize();
        cublasSgemv(handle, CUBLAS_OP_N, graphSize, graphSize, &alpha, graphGPU, graphSize, sideGPU, 1, &beta, gainGPU, 1);

        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };

        yaxbx<<<gridSize, blockSize>>>(graphSize, gainGPU, sideGPU, gainGPUH);

        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };

        cublasScopy(handle, graphSize, gainGPUH, 1, gainGPU, 1);
        cudaDeviceSynchronize();

        // calculate cut size
        cublasScopy(handle, graphSize * graphSize, graphGPU, 1, graphGPUH, 1);
        cudaDeviceSynchronize();
        cublasScopy(handle, graphSize * graphSize, graphGPUH, 1, graphGPUH2, 1);
        cudaDeviceSynchronize();

        // set gridsize for 2D
        gridSize = ((graphSize * graphSize) / blockSize) + 1;
        // printf("G: %d,B: %d", gridSize, blockSize);
        yambx<<<gridSize, blockSize>>>(graphSize, graphGPU, sideGPU, graphGPUH);
        cudaDeviceSynchronize();
        yambxt<<<gridSize, blockSize>>>(graphSize, graphGPUH, sideGPU, graphGPUH2);
        cudaDeviceSynchronize();

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };

        r1s<<<gridSize, blockSize>>>(graphSize, graphGPUH2, graphGPUH);
        cudaDeviceSynchronize();

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };

        cublasSasum(handle, graphSize * graphSize, graphGPUH, 1, &cut);
        cudaDeviceSynchronize();
        cudaMemcpy(hold, graphGPUH, sizeof(float) * graphSize * graphSize, cudaMemcpyDeviceToHost);

        cut = cut;
        std::cout << "cut size: " << cut << std::endl;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            std::cout << cudaGetErrorString(cudaStatus);
        };
        blockSize = BLOCK_SIZE;
        gridSize = std::ceil(static_cast<float>(graphSize) / blockSize);
        yaxbx<<<gridSize, blockSize>>>(graphSize, lockArray, gainGPU, lockGainGPU);
        cudaDeviceSynchronize();

        // cudaMemcpy(gain, lockGainGPU, sizeof(float) * graphSize, cudaMemcpyDeviceToHost);

        // cudaMemcpy(hold, graphGPUH, sizeof(float) * graphSize * graphSize, cudaMemcpyDeviceToHost);
        //     for (int j = 0; j < graphSize * graphSize; j++)
        // {
        //     printf("%.4f\t", hold[j]);
        //     if (j % graphSize == 0)
        //     {
        //         printf("\n");
        //     }
        // }

        
        cublasIsamax(handle, graphSize, lockGainGPU, 1, &maxGainIndex);
        cudaDeviceSynchronize();
        

        // cudaMemcpy(gain, sideGPU, sizeof(float) * graphSize, cudaMemcpyDeviceToHost);
        // cudaMemcpy(lockHold, lockArray, sizeof(float) * graphSize, cudaMemcpyDeviceToHost);
    


        // gain[maxGainIndex] = -gain[maxGainIndex];
        // lockHold[maxGainIndex] = -7;
        // printf("maxGainIndex: %d\n",maxGainIndex);
        // for(int l=0;l<graphSize;l++) printf("lockHold-%d:\t%d\n",l,lockHold[l]);

        // cudaMemcpy(sideGPU, gain, sizeof(float) * graphSize, cudaMemcpyHostToDevice);
        // cudaMemcpy(lockArray, lockHold, sizeof(float) * graphSize, cudaMemcpyHostToDevice);

        // cublasScopy(handle, graphSize, sideGPU, 1, sideGPUH, 1);
        // cudaDeviceSynchronize();

        // move<<<1,1>>>(sideGPU,lockArray,maxGainIndex,sideGPUH);
        // cudaDeviceSynchronize();
        // cudaStatus = cudaGetLastError();
        // if (cudaStatus != cudaSuccess)
        // {
        //     std::cout << cudaGetErrorString(cudaStatus);
        // };

        // cublasScopy(handle, graphSize, sideGPUH, 1, sideGPU, 1);
        // cudaDeviceSynchronize();

        // // std::cout << "cutSize: " << cut << std::endl;
        // for (int j = 0; j < graphSize; j++)
        //     printf("elem%d: %f\n", j, gain[j]);
    }
    cudaFree(gainGPU);
    cudaFree(gainGPUH);
    cudaFree(lockGainGPU);
    cudaFree(lockArray);
    cudaFree(graphGPU);
    cudaFree(graphGPUH2);
    cudaFree(graphGPUH);
    cudaFree(sideGPU);

    // std::cout << gridSize << "\t" << blockSize << std::endl;
    /*for (int i = 0; i < graphSize;i++) {

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
        std::cout << cudaGetErrorString(cudaStatus);
        };


        cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
        std::cout << cudaGetErrorString(cudaStatus);
        };


    }





    cudaMemcpy(graph,GPU_graphHold_2, sizeof(float) * graphSize * graphSize, cudaMemcpyDeviceToHost);
    ;

    cudaFree(GPU_graphPtr);
    cudaFree(GPU_sidePtr);
    cudaFree(GPU_gain);*/

    return cudaGetLastError();
}

void ReadGraph(float *&graph, int &graphsize, float *&side)
{
    int _graphSize = 0;
    std::ifstream GraphFile("Graph.txt");
    std::ifstream SideFile("Side.txt");
    std::string lineHold;

    while (std::getline(SideFile, lineHold))
    {
        _graphSize++;
    }

    int iterator = 0;
    SideFile.clear();
    SideFile.seekg(0);

    side = new float[_graphSize];
    graph = new float[_graphSize * _graphSize];

    while (std::getline(SideFile, lineHold))
    {
        side[iterator] = (float)std::stoi(lineHold);
        iterator++;
    }

    iterator = 0;
    char del = ',';

    while (std::getline(GraphFile, lineHold, del))
    {
        graph[iterator] = (float)std::stoi(lineHold);
        iterator++;
    }

    graphsize = _graphSize;
    SideFile.close();
    GraphFile.close();
}

int main()
{

    // Read Input
    int graphSize;
    float *graph = nullptr;
    float *side = nullptr;

    const float ratio = 0.5; // Left to Total

    // Read Input Graph
    ReadGraph(graph, graphSize, side);
    std::cout << "graphSize" << graphSize << std::endl;

    auto startGPU = std::chrono::high_resolution_clock::now();
    cudaError_t cudaStatus = FM(graphSize, graph, side, ratio);
    auto stopGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU - startGPU);
    cudaDeviceSynchronize();
    // auto startCPU = std::chrono::high_resolution_clock::now();
    // int maxV = 0;
    // int holdV = 0;
    // for (int i = 0; i < graphSize; i++)
    // {
    //     for (int j = 0; j < graphSize; j++)
    //     {
    //         holdV += graph[i * graphSize + j];
    //     }
    //     maxV = max(maxV, holdV);
    //     holdV = 0;
    // }

    // std::vector<std::list<int>> gainBucket;
    // std::vector<int> sideCPU(graphSize);
    // // make the gainbucket
    // for (int i = -maxV; i <= maxV; i++)
    // {
    //     gainBucket.push_back(std::list<int>());
    // }
    // // initial balance pass
    // float cutter;

    // for (int i = 0; i < graphSize; i++)
    // {
    //     cutter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    //     if (cutter < ratio)
    //     {
    //         sideCPU[i] = -1;
    //     }
    //     else
    //     {
    //         sideCPU[i] = 1;
    //     }
    //     gainBucket[G2ID(0)].push_back((i));
    // }
    // std::vector<int> gainVector(graphSize);
    // int cutC = 0;
    // for (int i = 0; i < PASSCOUNT; i++)
    // {
    //     // calculate gain
    //     int gainH = 0;
    //     int gainB = 0;

    //     for (int j = 0; j < graphSize; j++)
    //     {
    //         for (int k = 0; k < graphSize; k++)
    //         {
    //             if (side[j] == side[k] && graph[j * graphSize + k] == 1)
    //             {
    //                 gainH++;
    //             }
    //             else if (side[j] != side[k] && graph[j * graphSize + k] == 1)
    //             {
    //                 gainH--;
    //             }
    //         }
    //         gainB = gainVector[j];
    //         gainVector[j] = gainH;
    //         gainBucket[G2ID(gainB)].remove(j);
    //         gainBucket[G2ID(gainH)].push_back(j);

    //         gainH = 0;
    //         // std::cout<<gainVector[j]<<std::endl;
    //     }
    //     for (int j = 0; j < graphSize; j++)
    //     {
    //         for (int k = 0; k < graphSize; k++)
    //         {
    //             if (graph[j * graphSize + k] == 1)
    //             {

    //                 if (sideCPU[j] != sideCPU[k])
    //                 {
    //                     cutC++;
    //                 }
    //             }
    //         }
    //     }
    //     cutC = cutC / 2;
    //     printf("Cpu cut size: %d\n", cutC);
    //     cutC = 0;
    // }

    // auto stopCPU = std::chrono::high_resolution_clock::now();
    // auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(stopCPU - startCPU);

    // std::cout << "GPU Time: " << durationGPU.count() << "\t"
    //           << "CPU Time: " << durationCPU.count() << std::endl;
    free(graph);
    free(side);
    return 0;
}
