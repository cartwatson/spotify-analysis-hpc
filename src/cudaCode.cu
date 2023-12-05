#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

#include "util.cpp"  
#include "cudaSong.cpp"


// A device function to calculate the Euclidean distance between two Song objects in feature space.
__device__ double distanceGPU(const Song& a, const Song& b) {
    // Calculate the difference for each feature.
    double diff1 = a.feature1 - b.feature1;
    double diff2 = a.feature2 - b.feature2;
    double diff3 = a.feature3 - b.feature3;

    // Return the Euclidean distance using the standard formula.
    return sqrt(diff1 * diff1 + diff2 * diff2 + diff3 * diff3);
}

// A global function to compute the distance of each song to every centroid and assign the song to the nearest centroid.
__global__ void computeDistances(Song* songs, Song* centroids, int n, int k) {
    // Calculate the global index of the thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is within the range of the data set.
    if (idx < n) {
        // Reference to the song at the current index.
        Song& song = songs[idx];

        // Initialize the minimum distance with the maximum possible value.
        // This ensures any actual distance will be smaller.
        double maxDouble = 1.7976931348623157E+308;
        song.minDist = maxDouble;

        // Iterate over all centroids.
        for (int i = 0; i < k; ++i) {
            // Compute the distance from the current song to the current centroid.
            double dist = distanceGPU(song, centroids[i]);

            // If this distance is smaller than the current minimum distance,
            // update minDist and assign the song to this cluster (centroid).
            if (dist < song.minDist) {
                song.minDist = dist;
                song.cluster = i;
            }
        }
    }
}

// A global function to update the sum of features and the count of songs for each cluster.
__global__ void updateCentroids(Song* songs, int n, int k, double* sumDance, double* sumAcoustic, double* sumLive, int* nSongs, int numThreads) {
    // Calculate the global index of the thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // If the index is within the range of the data set.
    if (idx < n) {
        // Reference to the song at the current index.
        Song& song = songs[idx];

        // Update the sum of features for the cluster this song belongs to.
        // Note that each thread updates a unique location to prevent race conditions;
        // this implies that a reduction operation will be needed later.
        sumDance[song.cluster * numThreads + idx] += song.feature1;
        sumAcoustic[song.cluster * numThreads + idx] += song.feature2;
        sumLive[song.cluster * numThreads + idx] += song.feature3;

        // Increment the song count for this cluster.
        nSongs[song.cluster * numThreads + idx] += 1;
    }
}

// This kernel performs a reduction to sum up the accumulators for each feature and the number of songs per cluster.
__global__ void sumAccumulators(int k, double* sumDance, double* sumAcoustic, double* sumLive, int* nSongs, int numThreads) {
    // Calculate the unique index for each thread. This index determines which centroid this thread is responsible for.
    int centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we only process valid centroids.
    if (centroidIdx < k) {
        // Temporary variables to accumulate the sums.
        double danceSum = 0, acousticSum = 0, liveSum = 0;
        int songCount = 0;

        // Loop over the accumulators for each thread.
        // Each thread accumulates its partial sums separately to avoid race conditions.
        for (int i = 0; i < numThreads; ++i) {
            // Calculate the index in the flattened arrays for this thread's accumulator.
            int idx = centroidIdx * numThreads + i;
            // Sum the danceability, acousticness, and liveness for all songs in the current cluster.
            danceSum += sumDance[idx];
            acousticSum += sumAcoustic[idx];
            liveSum += sumLive[idx];
            // Sum the count of songs in the current cluster.
            songCount += nSongs[idx];
        }

        // Only one thread per centroid performs the final write-back to global memory.
        // This check is necessary because the kernel is launched with possibly more threads than centroids.
            // Write the final sums back to the first position for each cluster.
        sumDance[centroidIdx] = danceSum;
        sumAcoustic[centroidIdx] = acousticSum;
        sumLive[centroidIdx] = liveSum;
        nSongs[centroidIdx] = songCount;

    }
}
// This function performs k-means clustering on a set of songs using the GPU.
void kMeansGPU(std::vector<Song>& songs, int epochs, int k) {
    // Number of songs in the dataset.
    int n = songs.size();

    // The number of threads per block. This is usually a power of 2. Here we are using 256.
    int numThreads = 256;

    // Total number of accumulators is the number of clusters times the number of threads,
    // since each thread will have its own set of accumulators to avoid race conditions.
    int numAccumulators = k * numThreads;

    // Device pointers for the songs and centroids in GPU memory.
    Song* d_songs, * d_centroids;

    // Device pointers for feature sums and song counts in GPU memory.
    double* d_sumDance, * d_sumAcoustic, * d_sumLive;
    int* d_nSongs;

    // Allocate GPU memory for the songs and centroids.
    cudaMalloc(&d_songs, n * sizeof(Song));
    cudaMalloc(&d_centroids, k * sizeof(Song));

    // Allocate GPU memory for the feature sums and song counts, using the number of accumulators.
    cudaMalloc(&d_sumDance, numAccumulators * sizeof(double));
    cudaMalloc(&d_sumAcoustic, numAccumulators * sizeof(double));
    cudaMalloc(&d_sumLive, numAccumulators * sizeof(double));
    cudaMalloc(&d_nSongs, numAccumulators * sizeof(int));

    // Copy the song data from host to device memory.
    cudaMemcpy(d_songs, songs.data(), n * sizeof(Song), cudaMemcpyHostToDevice);

    // Vector to store the initial centroids on the host side.
    std::vector<Song> centroids(k);

    // Check if there are songs to process.
    if (n > 0) {
        // Random number generator for selecting initial centroids.
        std::mt19937 rng(static_cast<unsigned>(std::time(0)));
        std::uniform_int_distribution<int> uni(0, n - 1);

        // Initialize centroids to random songs from the dataset.
        for (int i = 0; i < k; ++i) {
            centroids[i] = songs[uni(rng)];
            centroids[i].cluster = i;
        }
    }
    else {
        // If there are no songs, output an error message.
        std::cerr << "No songs to process." << std::endl;
    }

    // Copy the centroids from host to device memory.
    cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Song), cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed for the dataset.
    int blocksPerGrid = (n + numThreads - 1) / numThreads;

    // Perform the clustering for a fixed number of epochs.
    for (int i = 0; i < epochs; ++i) {
        // Zero out the accumulators in GPU memory at the start of each epoch.
        cudaMemset(d_sumDance, 0, numAccumulators * sizeof(double));
        cudaMemset(d_sumAcoustic, 0, numAccumulators * sizeof(double));
        cudaMemset(d_sumLive, 0, numAccumulators * sizeof(double));
        cudaMemset(d_nSongs, 0, numAccumulators * sizeof(int));

        // Assign each song to the nearest centroid.
        computeDistances << <blocksPerGrid, numThreads >> > (d_songs, d_centroids, n, k);
        cudaDeviceSynchronize();

        // Accumulate feature sums and song counts for each cluster.
        updateCentroids << <blocksPerGrid, numThreads >> > (d_songs, n, k, d_sumDance, d_sumAcoustic, d_sumLive, d_nSongs, numThreads);
        cudaDeviceSynchronize();

        // Sum up the accumulators to get the total sums and counts for each cluster.
        sumAccumulators << <k, 1 >> > (k, d_sumDance, d_sumAcoustic, d_sumLive, d_nSongs, numThreads);
        cudaDeviceSynchronize();

        // Copy the updated centroid data back to the host.
        cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Song), cudaMemcpyDeviceToHost);

        // Copy the final sums and counts back to the host.
        std::vector<double> sumDance(k), sumAcoustic(k), sumLive(k);
        std::vector<int> nSongs(k);
        cudaMemcpy(sumDance.data(), d_sumDance, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sumAcoustic.data(), d_sumAcoustic, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sumLive.data(), d_sumLive, k * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(nSongs.data(), d_nSongs, k * sizeof(int), cudaMemcpyDeviceToHost);

        // Update the centroids on the host based on the new sums and counts.
        for (int j = 0; j < k; ++j) {
            centroids[j].feature1 = sumDance[j] / nSongs[j];
            centroids[j].feature2 = sumAcoustic[j] / nSongs[j];
            centroids[j].feature3 = sumLive[j] / nSongs[j];
        }

        // Copy the updated centroids back to the device for the next epoch.
        cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Song), cudaMemcpyHostToDevice);
    }

    // Once all epochs are done, copy the final song data back to the host.
    cudaMemcpy(songs.data(), d_songs, n * sizeof(Song), cudaMemcpyDeviceToHost);

    // Free all allocated GPU memory.
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
    kMeansGPU(songs, 10, 5); // Call to your CUDA k-means function

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
    std::string fn = "output.csv";
    writeCSV(outputData, fn, header);

    // Clean up dynamically allocated memory
    for (double* row : data) {
        delete[] row;
    }
    for (double* row : outputData) {
        delete[] row;
    }

    return 0;
}
