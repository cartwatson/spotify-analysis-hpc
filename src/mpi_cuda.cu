#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <fstream>
#include <assert.h>

struct Song {
    float feature1, feature2, feature3;
    int cluster;

    Song(): feature1(0.0), feature2(0.0), feature3(0.0), cluster(-1) {}

    Song(float f1, float f2, float f3):
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster(-1)
    {}
};

struct Centroid {
    float feature1, feature2, feature3;
    int cluster_size;

    Centroid(): feature1(0.0), feature2(0.0), feature3(0.0), cluster_size(0) {}

    Centroid(float f1, float f2, float f3):
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster_size(0)
    {}
};
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}
__device__ double sq_distance(Song* s1, Centroid* c)
{
    return (s1->feature1 - c->feature1) * (s1->feature1 - c->feature1) +
        (s1->feature2 - c->feature2) * (s1->feature2 - c->feature2) +
        (s1->feature3 - c->feature3) * (s1->feature3 - c->feature3);
}

__global__ void assignSongToCluster(Song* songs, Centroid* centroids, int n, int k)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float minDist = 100000000;
    int cluster = -1;
    if (gid < n)
        for (int c = 0; c < k; ++c)
        {
            double dist = sq_distance(&songs[gid], &centroids[c]);
            if (dist < minDist)
            {
                minDist = dist;
                cluster = c;
            }
        }
    songs[gid].cluster = cluster;
}

__global__ void calculateNewCentroids(Song* songs, Centroid* centroids, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n)
    {
        int cluster = songs[gid].cluster; // Get the cluster of each song
        atomicAdd(&centroids[cluster].feature1, songs[gid].feature1);
        atomicAdd(&centroids[cluster].feature2, songs[gid].feature2);
        atomicAdd(&centroids[cluster].feature3, songs[gid].feature3);
        atomicAdd(&centroids[cluster].cluster_size, 1);
    }
}

// Wrapper function for assignSongToCluster kernel
extern "C" void callAssignSongToCluster(Song * songs, Centroid * centroids, int n, int k) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; 

    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    // Launch the kernel
    assignSongToCluster << <grid, block >> > (songs, centroids, n, k);

    // Check for errors and synchronize
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}

// Wrapper function for calculateNewCentroids kernel
extern "C" void callCalculateNewCentroids(Song * songs, Centroid * centroids, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerGrid);
    dim3 block(threadsPerBlock);

    // Launch the kernel
    calculateNewCentroids << <grid, block >> > (songs, centroids, n);

    // Check for errors and synchronize
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}

// Definitions of CUDA wrapper functions
extern "C" void allocateMemoryAndCopyToGPU(Song** deviceSongs, Centroid** deviceCentroids, const Song* hostSongs, const Centroid* hostCentroids, int numSongs, int numCentroids) {
    // Allocate memory for songs on the device
    checkCuda(cudaMalloc(deviceSongs, numSongs * sizeof(Song)));
    checkCuda(cudaMemcpy(*deviceSongs, hostSongs, numSongs * sizeof(Song), cudaMemcpyHostToDevice));

    // Allocate memory for centroids on the device
    checkCuda(cudaMalloc(deviceCentroids, numCentroids * sizeof(Centroid)));
    checkCuda(cudaMemcpy(*deviceCentroids, hostCentroids, numCentroids * sizeof(Centroid), cudaMemcpyHostToDevice));
}

extern "C" void freeGPUMemory(Song* deviceSongs, Centroid* deviceCentroids) {
    checkCuda(cudaFree(deviceSongs));
    checkCuda(cudaFree(deviceCentroids));
}

extern "C" void gpuErrorCheck() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        assert(error == cudaSuccess);
    }
    checkCuda(cudaDeviceSynchronize());
}

extern "C" void resetCentroids(Centroid* centroids_d, int k) {
    checkCuda(cudaMemset(centroids_d, 0, k * sizeof(Centroid)));
}

extern "C" void copyCentroidsToHost(Centroid* centroids, Centroid* centroids_d, int k) {
    checkCuda(cudaMemcpy(centroids, centroids_d, k * sizeof(Centroid), cudaMemcpyDeviceToHost));
}

extern "C" void copyCentroidsToDevice(Centroid* deviceCentroids, Centroid* hostCentroids, int k) {
    checkCuda(cudaMemcpy(deviceCentroids, hostCentroids, k * sizeof(Centroid), cudaMemcpyHostToDevice));
}

extern "C" void copySongsToHost(Song* hostSongs, Song* deviceSongs, int localN) {
    checkCuda(cudaMemcpy(hostSongs, deviceSongs, localN * sizeof(Song), cudaMemcpyDeviceToHost));
}
