#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>
#include <assert.h>

#include "util.cpp"

#define BLOCKSIZE 256


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

/**
 * Calculates the distance between two points in 3D space (no need to get the square root, it's all relative)
*/
__device__ double sq_distance(Song* s1, Centroid* c)
{
    return (s1->feature1 - c->feature1) * (s1->feature1 - c->feature1) +
           (s1->feature2 - c->feature2) * (s1->feature2 - c->feature2) +
           (s1->feature3 - c->feature3) * (s1->feature3 - c->feature3);
}

/**
 * Assigns each song to the cluster with the closest centroid
*/
__global__ void assignSongToCluster(Song* songs, Centroid* centroids, int n, int k)
{
    extern __shared__ Centroid shared_centroids[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < k)
        shared_centroids[threadIdx.x] = centroids[threadIdx.x];
    __syncthreads();

    if (gid < n)
    {
        double min_dist = sq_distance(&songs[gid], &shared_centroids[0]);
        int cluster = 0;
        for (int c = 1; c < k; ++c)
        {
            double dist = sq_distance(&songs[gid], &shared_centroids[c]);
            if (dist < min_dist)
            {
                min_dist = dist;
                cluster = c;
            }
        }
        songs[gid].cluster = cluster;
    }
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

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        assert(result == cudaSuccess);
    }
    return result;
}

void kMeansCUDA(Song* songs, int n, int epochs, int k)
{
    Song* songs_d;
    checkCuda(cudaMalloc(&songs_d, n * sizeof(Song)));
    checkCuda(cudaMemcpy(songs_d, songs, n * sizeof(Song), cudaMemcpyHostToDevice));

    #ifdef TESTING
    std::mt19937 rng(123);
    #else
    std::mt19937 rng(static_cast<unsigned>(std::time(0)));
    #endif
    Centroid* centroids = new Centroid[k];
    Centroid* centroids_d;
    for (int i = 0; i < k; ++i)
    {
        int rand_idx = rng() % n;
        centroids[i] = Centroid(songs[rand_idx].feature1, songs[rand_idx].feature2, songs[rand_idx].feature3);
    }
    checkCuda(cudaMalloc(&centroids_d, k*sizeof(Centroid)));
    checkCuda(cudaMemcpy(centroids_d, centroids, k*sizeof(Centroid), cudaMemcpyHostToDevice));

    int nBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 gridDim(nBlocks, 1, 1);
    dim3 blockDim(BLOCKSIZE, 1, 1);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        assignSongToCluster<<<gridDim, blockDim>>>(songs_d, centroids_d, n, k); // Use the centroids to assign each song to a cluster
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaMemset(centroids_d, 0, k*sizeof(Centroid)));

        calculateNewCentroids<<<gridDim, blockDim>>>(songs_d, centroids_d, n);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaMemcpy(centroids, centroids_d, k*sizeof(Centroid), cudaMemcpyDeviceToHost));
        for (int i = 0; i < k; ++i)
        {
            centroids[i].feature1 /= centroids[i].cluster_size;
            centroids[i].feature2 /= centroids[i].cluster_size;
            centroids[i].feature3 /= centroids[i].cluster_size;
        }
        checkCuda(cudaMemcpy(centroids_d, centroids, k*sizeof(Centroid), cudaMemcpyHostToDevice));
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

    kMeansCUDA(songs, data.size(), 100, 5);

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