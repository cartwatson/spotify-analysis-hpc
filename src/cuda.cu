#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <assert.h>

#include "util.cpp"

#define FEATURES 3
#define BLOCKSIZE 256
#define EPOCHS 2
#define K 5

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

/**
 * Calculates the distance between two points in 3D space (no need to get the square root, it's all relative)
*/
__device__ double sq_distance(float f1, float f2, float f3, float c1, float c2, float c3)
{
    return (f1 - c1) * (f1 - c1) +
           (f2 - c2) * (f2 - c2) +
           (f3 - c3) * (f3 - c3);
}

__global__ void epochIter(float* songs, int* clusterAssignments, float* centroids, int* clusterCounts, int n)
{
    __shared__ float s_centroids[K*FEATURES];
    __shared__ int s_clusterCounts[K];

    __shared__ float s_songs[BLOCKSIZE*FEATURES];
    __shared__ int s_clusterAssignments[BLOCKSIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Load centroids into shared memory
    int tid = threadIdx.x;
    if (tid < K)
    {
        s_centroids[tid*FEATURES] = centroids[tid*FEATURES];
        s_centroids[tid*FEATURES+1] = centroids[tid*FEATURES+1];
        s_centroids[tid*FEATURES+2] = centroids[tid*FEATURES+2];
        s_clusterCounts[tid] = 0;
    }

    // Load songs into shared memory
    s_songs[tid*FEATURES] = songs[gid*FEATURES];
    s_songs[tid*FEATURES+1] = songs[gid*FEATURES+1];
    s_songs[tid*FEATURES+2] = songs[gid*FEATURES+2];
    s_clusterAssignments[tid] = -1;

    __syncthreads();

    // Find closest centroid
    float* song = &s_songs[tid*FEATURES];
    float minDist = sq_distance(song[0], song[1], song[2], s_centroids[0], s_centroids[1], s_centroids[2]);
    int closestCentroid = 0;

    for (int i = 1; i < K; ++i)
    {
        float dist = sq_distance(song[0], song[1], song[2], s_centroids[i*FEATURES], s_centroids[i*FEATURES+1], s_centroids[i*FEATURES+2]);
        if (dist < minDist)
        {
            minDist = dist;
            closestCentroid = i;
        }
    }

    // Assign song to closest centroid
    s_clusterAssignments[tid] = closestCentroid;
    atomicAdd(&s_clusterCounts[closestCentroid], 1);

    __syncthreads();

    // Update centroids
    if (tid < K)
    {
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        for (int i = 0; i < BLOCKSIZE; i++)
        {
            if (s_clusterAssignments[i] == tid)
            {
                sum1 += s_songs[i*FEATURES];
                sum2 += s_songs[i*FEATURES+1];
                sum3 += s_songs[i*FEATURES+2];
            }
        }

        int count = s_clusterCounts[tid];
        if (count > 0)
        {
            s_centroids[tid*FEATURES] = sum1 / count;
            s_centroids[tid*FEATURES+1] = sum2 / count;
            s_centroids[tid*FEATURES+2] = sum3 / count;
        }
    }
}

void kMeansCUDA(float* songs_h, int n)
{
    int allSongsSize = n*FEATURES*sizeof(float); // Song list size in bytes
    int allCentroidsSize = K*FEATURES*sizeof(float); // Centroid list size in bytes

    // Initialize songs on device
    float* songs_d;
    checkCuda(cudaMalloc(&songs_d, allSongsSize));
    checkCuda(cudaMemcpy(songs_d, songs_h, allSongsSize, cudaMemcpyHostToDevice));

    std::mt19937 rng(123);
    float centroids[K*FEATURES];
    for (int i = 0; i < K; ++i)
    {
        int randIdx = rng() % n;
        memcpy(&centroids[i*FEATURES], &songs_h[randIdx*FEATURES], FEATURES*sizeof(float));
    }

    // Initialize centroids on device
    float* centroids_d;
    checkCuda(cudaMalloc(&centroids_d, allCentroidsSize));
    checkCuda(cudaMemcpy(centroids_d, centroids, allCentroidsSize, cudaMemcpyHostToDevice));

    int nBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 gridDim(nBlocks, 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    // initialize all cluster counts to 0 and cluster assignments to -1
    int* clusterAssignments = new int[n];
    for (int i = 0; i < n; ++i)
        clusterAssignments[i] = -1;

    int clusterCounts[K];
    for (int i = 0; i < K; ++i)
        clusterCounts[i] = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        epochIter<<<gridDim, blockDim>>>(songs_d, clusterAssignments, centroids_d, clusterCounts, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    checkCuda(cudaMemcpy(songs_h, songs_d, allSongsSize, cudaMemcpyDeviceToHost));
}


int main(int argc, char* argv[])
{
    int maxLines = 250000;
    if (argc > 1)
    {
        maxLines = std::stoi(argv[1]);
        if (maxLines < 0 || maxLines > MAX_LINES)
            maxLines = MAX_LINES;
        std::cout << "maxLines = " << maxLines << std::endl;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double*> allData = parseCSV(maxLines);
    std::vector<std::string> featureNames = {"danceability", "acousticness", "liveness"};

    float* songs = new float[allData.size()*FEATURES]; // +1 to leave room for cluster
    for (size_t i = 0; i < allData.size(); ++i)
    {
        songs[i*FEATURES] = allData[i][0];
        songs[i*FEATURES+1] = allData[i][6];
        songs[i*FEATURES+2] = allData[i][8];
    }

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;

    kMeansCUDA(songs, allData.size());

    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    std::cout << "Writing output to file..." << std::endl;
    std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";

    std::vector<double*> output;
    for (size_t i = 0; i < allData.size(); ++i)
    {
        double* row = new double[4];
        row[0] = songs[i*FEATURES];
        row[1] = songs[i*FEATURES+1];
        row[2] = songs[i*FEATURES+2];
        row[3] = songs[i*FEATURES+3];
        output.push_back(row);
    }

    writeCSV(output, "src/data/output.csv", header);

    return 0;
}