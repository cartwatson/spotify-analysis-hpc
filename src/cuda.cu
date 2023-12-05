#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <assert.h>

#include "util.cpp"

#define BLOCKSIZE 256
#define EPOCHS 100
#define K 5


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

    __host__ __device__ Centroid(): feature1(0.0), feature2(0.0), feature3(0.0), cluster_size(0) {}

    Centroid(float f1, float f2, float f3):
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster_size(0)
    {}
};

/**
 * Calculates the distance between two points in 3D space (no need to get the square root, it's all relative)
*/
__device__ double sq_distance(Song* s1, Centroid* c)
{
    return (s1->feature1 - c->feature1) * (s1->feature1 - c->feature1) +
           (s1->feature2 - c->feature2) * (s1->feature2 - c->feature2) +
           (s1->feature3 - c->feature3) * (s1->feature3 - c->feature3);
}

__global__ void epochIter(Song* songs, Centroid* centroids, int n)
{
    extern __shared__ Centroid shared_centroids[];
    if (threadIdx.x < K)
        shared_centroids[threadIdx.x] = centroids[threadIdx.x];
    __syncthreads();

    // Assign each song to the cluster with the closest centroid
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    double minDist = sq_distance(&songs[gid], &shared_centroids[0]);
    int closestClust = 0;
    for (int c = 0; c < K; ++c)
    {
        double newDist = sq_distance(&songs[gid], &shared_centroids[c]);
        if (newDist < minDist)
        {
            minDist = newDist;
            closestClust = c;
        }
    }
    songs[gid].cluster = closestClust;

    __syncthreads();

    // Update the centroids
    if (threadIdx.x < K)
        shared_centroids[threadIdx.x] = Centroid();
    __syncthreads();

    // Calculate the new centroids by averaging the songs in each cluster
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        int cluster = songs[i].cluster;
        atomicAdd(&shared_centroids[cluster].feature1, songs[i].feature1);
        atomicAdd(&shared_centroids[cluster].feature2, songs[i].feature2);
        atomicAdd(&shared_centroids[cluster].feature3, songs[i].feature3);
        atomicAdd(&shared_centroids[cluster].cluster_size, 1);
    }
    __syncthreads();

    if (threadIdx.x < K)
    {
        shared_centroids[threadIdx.x].feature1 /= shared_centroids[threadIdx.x].cluster_size;
        shared_centroids[threadIdx.x].feature2 /= shared_centroids[threadIdx.x].cluster_size;
        shared_centroids[threadIdx.x].feature3 /= shared_centroids[threadIdx.x].cluster_size;
    }
    __syncthreads();

    // Copy the new centroids back to global memory
    if (threadIdx.x < K)
        centroids[threadIdx.x] = shared_centroids[threadIdx.x];
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

void kMeansCUDA(Song* songs, int n)
{
    Song* songs_d;
    checkCuda(cudaMalloc(&songs_d, n * sizeof(Song)));
    checkCuda(cudaMemcpy(songs_d, songs, n * sizeof(Song), cudaMemcpyHostToDevice));

    #ifdef TESTING
    std::mt19937 rng(123);
    #else
    std::mt19937 rng(static_cast<unsigned>(std::time(0)));
    #endif
    Centroid centroids[K];
    Centroid* centroids_d;
    for (int i = 0; i < K; ++i)
    {
        int rand_idx = rng() % n;
        centroids[i] = Centroid(songs[rand_idx].feature1, songs[rand_idx].feature2, songs[rand_idx].feature3);
    }
    long centroids_size = K*sizeof(Centroid);
    checkCuda(cudaMalloc(&centroids_d, centroids_size));
    checkCuda(cudaMemcpy(centroids_d, centroids, centroids_size, cudaMemcpyHostToDevice));

    int nBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 gridDim(nBlocks, 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    for (int epoch = 0; epoch < EPOCHS; ++epoch)
    {
        epochIter<<<gridDim, blockDim, centroids_size>>>(songs_d, centroids_d, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    checkCuda(cudaMemcpy(songs, songs_d, n*sizeof(Song), cudaMemcpyDeviceToHost));
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
    
    std::vector<double*> data = parseCSV(maxLines);
    std::vector<std::string> featureNames = {"danceability", "acousticness", "liveness"};

    Song* songs = new Song[data.size()];
    for (size_t i = 0; i < data.size(); ++i)
        songs[i] = Song(data[i][0], data[i][6], data[i][8]);

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;

    kMeansCUDA(songs, data.size());

    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    std::cout << "Writing output to file..." << std::endl;
    std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";

    std::vector<double*> output;
    for (size_t i = 0; i < data.size(); ++i)
    {
        double* row = new double[4];
        row[0] = songs[i].feature1;
        row[1] = songs[i].feature2;
        row[2] = songs[i].feature3;
        row[3] = songs[i].cluster;
        output.push_back(row);
    }

    writeCSV(output, "src/data/output.csv", header);

    return 0;
}