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
#define EPOCHS 1
#define K 5


/**
 * Calculates the distance between two points in 3D space (no need to get the square root, it's all relative)
*/
__device__ double sq_distance(float f1, float f2, float f3, float c1, float c2, float c3)
{
    return (f1 - c1) * (f1 - c1) +
           (f2 - c2) * (f2 - c2) +
           (f3 - c3) * (f3 - c3);
}

__global__ void epochIter(float* songs, float* centroids, int n)
{
    __shared__ float s_centroids[K*(FEATURES+1)];
    if (threadIdx.x < K*(FEATURES+1))
        s_centroids[threadIdx.x] = centroids[threadIdx.x];
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    
    float* song = &songs[gid*(FEATURES+1)]; // Get pointer to this thread's song
    float minDist = sq_distance(song[0], song[1], song[2], s_centroids[0], s_centroids[1], s_centroids[2]); // Initialize to the distance to first centroid
    int closestCluster = 0; // centroid 0 is closest by default at this point
    for (int i = 1; i < K; ++i)
    {
        float dist = sq_distance(
            song[0], song[1], song[2],
            s_centroids[i*(FEATURES+1)], s_centroids[i*(FEATURES+1)+1], s_centroids[i*(FEATURES+1)+2]
        );
        if (dist < minDist)
        {
            minDist = dist;
            closestCluster = i;
        }
    }
    song[3] = closestCluster;

    __syncthreads();

    //reset centroids
    if (threadIdx.x < K*(FEATURES+1))
        s_centroids[threadIdx.x] = 0;
    
    __syncthreads();

    //calculate new centroids
    atomicAdd(&s_centroids[closestCluster*(FEATURES+1)], song[0]);
    atomicAdd(&s_centroids[closestCluster*(FEATURES+1)+1], song[1]);
    atomicAdd(&s_centroids[closestCluster*(FEATURES+1)+2], song[2]);
    atomicAdd(&s_centroids[closestCluster*(FEATURES+1)+3], 1);

    __syncthreads();

    //update centroids
    if (threadIdx.x < K*(FEATURES+1))
    {
        centroids[threadIdx.x] = s_centroids[threadIdx.x] / s_centroids[threadIdx.x+3];
    }
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

void kMeansCUDA(float* songs_h, int n)
{
    int songSize = (FEATURES+1)*sizeof(float); // Song/centroid size in bytes
    int allSongsSize = n*songSize;
    float* songs_d;
    checkCuda(cudaMalloc(&songs_d, allSongsSize));
    checkCuda(cudaMemcpy(songs_d, songs_h, allSongsSize, cudaMemcpyHostToDevice));

    std::mt19937 rng(123);
    float centroids[K*(FEATURES+1)]; // +1 to leave room for cluster size
    for (int i = 0; i < K; ++i)
    {
        int randIdx = rng() % n;
        memcpy(&centroids[i*(FEATURES+1)], &songs_h[randIdx*(FEATURES+1)], songSize);
        centroids[i*(FEATURES+1)+3] = 0; // Initialize cluster size to 0
    }
    int allCentroidsSize = K*songSize;
    float* centroids_d;
    checkCuda(cudaMalloc(&centroids_d, allCentroidsSize));
    checkCuda(cudaMemcpy(centroids_d, centroids, allCentroidsSize, cudaMemcpyHostToDevice));

    int nBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 gridDim(nBlocks, 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        epochIter<<<gridDim, blockDim>>>(songs_d, centroids_d, n);
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

    float* songs = new float[allData.size()*(FEATURES+1)]; // +1 to leave room for cluster
    for (size_t i = 0; i < allData.size(); ++i)
    {
        songs[i*(FEATURES+1)] = allData[i][0];
        songs[i*(FEATURES+1)+1] = allData[i][6];
        songs[i*(FEATURES+1)+2] = allData[i][8];
        songs[i*(FEATURES+1)+3] = -1; // Initialize cluster to -1
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
        row[0] = songs[i*(FEATURES+1)];
        row[1] = songs[i*(FEATURES+1)+1];
        row[2] = songs[i*(FEATURES+1)+2];
        row[3] = songs[i*(FEATURES+1)+3];
        output.push_back(row);
    }

    writeCSV(output, "src/data/output.csv", header);

    return 0;
}