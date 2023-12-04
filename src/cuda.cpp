#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

#include "util.cpp"  // Assuming this contains necessary utility functions
#include "cudaSong.cpp"  // Assuming this contains the definition of the Song struct

// nvcc -o kmeans cuda.cpp song.cpp util.cpp -lcuda -lcudart -arch=sm_86

__device__ double distanceGPU(const Song& a, const Song& b) {
    double diff1 = a.feature1 - b.feature1;
    double diff2 = a.feature2 - b.feature2;
    double diff3 = a.feature3 - b.feature3;

    return sqrt(diff1 * diff1 + diff2 * diff2 + diff3 * diff3);
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void computeDistances(Song* songs, Song* centroids, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Song& song = songs[idx];
        song.minDist = std::numeric_limits<double>::max();

        for (int i = 0; i < k; ++i) {
            double dist = distanceGPU(song, centroids[i]);
            if (dist < song.minDist) {
                song.minDist = dist;
                song.cluster = i;
            }
        }
    }
}

__global__ void updateCentroids(Song* songs, int n, int k, double* sumDance, double* sumAcoustic, double* sumLive, int* nSongs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Song& song = songs[idx];
        atomicAddDouble(&sumDance[song.cluster], song.feature1);
        atomicAddDouble(&sumAcoustic[song.cluster], song.feature2);
        atomicAddDouble(&sumLive[song.cluster], song.feature3);
        atomicAdd(&nSongs[song.cluster], 1);
    }
}

void kMeansGPU(std::vector<Song>& songs, int epochs, int k) {
    int n = songs.size();

    // Allocate memory on GPU
    Song* d_songs, * d_centroids;
    double* d_sumDance, * d_sumAcoustic, * d_sumLive;
    int* d_nSongs;
    cudaMalloc(&d_songs, n * sizeof(Song));
    cudaMalloc(&d_centroids, k * sizeof(Song));
    cudaMalloc(&d_sumDance, k * sizeof(double));
    cudaMalloc(&d_sumAcoustic, k * sizeof(double));
    cudaMalloc(&d_sumLive, k * sizeof(double));
    cudaMalloc(&d_nSongs, k * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_songs, songs.data(), n * sizeof(Song), cudaMemcpyHostToDevice);

    // Initialize centroids
    std::vector<Song> centroids(k);
    std::mt19937 rng(static_cast<unsigned>(std::time(0))); // Random number generator
    std::uniform_int_distribution<int> uni(0, n - 1);     // Uniform distribution

    // Randomly select initial centroids
    for (int i = 0; i < k; ++i) {
        centroids[i] = songs[uni(rng)];
        centroids[i].cluster = i; // Set the cluster id for the centroid
    }

    // Copy centroids to GPU
    cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Song), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < epochs; ++i) {
        cudaMemset(d_sumDance, 0, k * sizeof(double));
        cudaMemset(d_sumAcoustic, 0, k * sizeof(double));
        cudaMemset(d_sumLive, 0, k * sizeof(double));
        cudaMemset(d_nSongs, 0, k * sizeof(int));

        computeDistances << <blocksPerGrid, threadsPerBlock >> > (d_songs, d_centroids, n, k);
        cudaDeviceSynchronize();

        updateCentroids << <blocksPerGrid, threadsPerBlock >> > (d_songs, n, k, d_sumDance, d_sumAcoustic, d_sumLive, d_nSongs);
        cudaDeviceSynchronize();

        // Copy sums and counts back to host to compute new centroids
        cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Song), cudaMemcpyDeviceToHost);
        std::vector<double> sumDance(k), sumAcoustic(k), sumLive(k);
        std::vector<int> nSongs(k);
        cudaMemcpy(sumDance.data(), d_sumDance, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sumAcoustic.data(), d_sumAcoustic, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sumLive.data(), d_sumLive, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(nSongs.data(), d_nSongs, k * sizeof(int), cudaMemcpyDeviceToHost);

        for (int j = 0; j < k; ++j) {
            centroids[j].feature1 = sumDance[j] / nSongs[j];
            centroids[j].feature2 = sumAcoustic[j] / nSongs[j];
            centroids[j].feature3 = sumLive[j] / nSongs[j];
        }

        // Copy updated centroids back to GPU
        cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Song), cudaMemcpyHostToDevice);
    }

    // Copy final results back to host
    cudaMemcpy(songs.data(), d_songs, n * sizeof(Song), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_songs);
    cudaFree(d_centroids);
    cudaFree(d_sumDance);
    cudaFree(d_sumAcoustic);
    cudaFree(d_sumLive);
    cudaFree(d_nSongs);
}

int main(int argc, char** argv) {
    int maxLines = 250000;
    if (argc > 1) {
        maxLines = std::stoi(argv[1]);
        if (maxLines < 0) {
            maxLines = 250000; 
        }
    }

    std::cout << "Max lines to process: " << maxLines << std::endl;

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Load data
    std::vector<double*> data = parseCSV(maxLines); 
    std::vector<Song> songs;
    for (double* row : data) {
        songs.push_back(Song(row[0], row[6], row[8])); 
    }

    // End timer for parsing
    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parseDuration = endParse - start;
    std::cout << "Parsed data in " << parseDuration.count() << " seconds" << std::endl;

    // Run k-means clustering
    std::cout << "Running k-means..." << std::endl;
    kMeansGPU(songs, 100, 5); // Call to your CUDA k-means function

    // End timer for k-means
    auto endkMeans = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kMeansDuration = endkMeans - endParse;
    std::cout << "Finished k-means in " << kMeansDuration.count() << " seconds" << std::endl;

    // Write output to file
    std::cout << "Writing output to file..." << std::endl;
    std::vector<std::string> featureNames = { "danceability", "acousticness", "liveness" };
    std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";
    std::vector<double*> outputData;
    for (Song& song : songs) {
        double* row = new double[4];
        row[0] = song.feature1;
        row[1] = song.feature2;
        row[2] = song.feature3;
        row[3] = song.cluster;
        outputData.push_back(row);
    }
    writeCSV(outputData, "output.csv", header); 

    // Clean up dynamically allocated memory
    for (double* row : data) {
        delete[] row;
    }
    for (double* row : outputData) {
        delete[] row;
    }

    return 0;
}
