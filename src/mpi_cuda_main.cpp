#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <fstream>
#include <assert.h>

#include "util.cpp"

struct Song {
    float feature1, feature2, feature3;
    int cluster;

    Song() : feature1(0.0), feature2(0.0), feature3(0.0), cluster(-1) {}

    Song(float f1, float f2, float f3) :
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster(-1)
    {}
};

struct Centroid {
    float feature1, feature2, feature3;
    int cluster_size;

    Centroid() : feature1(0.0), feature2(0.0), feature3(0.0), cluster_size(0) {}

    Centroid(float f1, float f2, float f3) :
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster_size(0)
    {}
};

extern "C" {
    void callAssignSongToCluster(Song* songs, Centroid* centroids, int n, int k);
    void callCalculateNewCentroids(Song* songs, Centroid* centroids, int n);
    void allocateMemoryAndCopyToGPU(Song** deviceSongs, Centroid** deviceCentroids, const Song* hostSongs, const Centroid* hostCentroids, int numSongs, int numCentroids);
    void freeGPUMemory(Song* deviceSongs, Centroid* deviceCentroids);
    void gpuErrorCheck();
    void resetCentroids(Centroid* centroids_d, int k);
    void copyCentroidsToHost(Centroid* centroids, Centroid* centroids_d, int k);
    void copyCentroidsToDevice(Centroid* deviceCentroids, Centroid* hostCentroids, int k);
    void copySongsToHost(Song* hostSongs, Song* deviceSongs, int localN);
}

void distributeData(MPI_Datatype MPI_Song, std::vector<Song>& allSongs, std::vector<Song>& localSongs, int world_size, int world_rank) {
    int totalSongs;
    std::vector<int> sendCounts(world_size), displacements(world_size);

    // On the root process, prepare the data for distribution and calculate send counts and displacements
    if (world_rank == 0) {
        totalSongs = allSongs.size();
        int songsPerProcess = totalSongs / world_size;
        int remaining = totalSongs % world_size;

        for (int i = 0; i < world_size; ++i) {
            sendCounts[i] = songsPerProcess + (i < remaining ? 1 : 0);
            displacements[i] = (i == 0 ? 0 : displacements[i - 1] + sendCounts[i - 1]);
        }
    }

    // Broadcast the total number of songs to all processes
    MPI_Bcast(&totalSongs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process prepares its localSongs vector
    int localSongCount = totalSongs / world_size + (world_rank < totalSongs % world_size ? 1 : 0);
    localSongs.resize(localSongCount);

    // Scatter the songs from the root process to all other processes
    MPI_Scatterv(allSongs.data(), sendCounts.data(), displacements.data(), MPI_Song,
        localSongs.data(), sendCounts[world_rank], MPI_Song, 0, MPI_COMM_WORLD);
}

void gatherResults(MPI_Datatype MPI_Song, std::vector<Song>& allSongs, std::vector<Song>& localSongs, int world_size, int world_rank) {
    int totalSongs;
    std::vector<int> recvCounts(world_size), displacements(world_size);

    // The root process prepares to receive the results
    if (world_rank == 0) {
        totalSongs = allSongs.size();
        int songsPerProcess = totalSongs / world_size;
        int remaining = totalSongs % world_size;

        for (int i = 0; i < world_size; ++i) {
            recvCounts[i] = songsPerProcess + (i < remaining ? 1 : 0);
            displacements[i] = (i == 0 ? 0 : displacements[i - 1] + recvCounts[i - 1]);
        }
    }

    // Gather the songs from all processes back to the root process
    MPI_Gatherv(localSongs.data(), localSongs.size(), MPI_Song,
        allSongs.data(), recvCounts.data(), displacements.data(), MPI_Song, 0, MPI_COMM_WORLD);
}

void kMeansCUDAMPI(Song* localSongs, int localN, int epochs, int k, int world_size, int world_rank) {
    Centroid* centroids = new Centroid[k];
    Centroid* centroids_d;
    Song* localSongs_d;
    std::vector<Centroid> allCentroids;

    // Initialize centroids (only on the root process)
    if (world_rank == 0) {
        std::mt19937 rng(static_cast<unsigned>(std::time(0)));
        for (int i = 0; i < k; ++i) {
            int rand_idx = rng() % localN;
            centroids[i] = Centroid(localSongs[rand_idx].feature1, localSongs[rand_idx].feature2, localSongs[rand_idx].feature3);
        }
        allCentroids.resize(k * world_size);
    }

    MPI_Bcast(centroids, k * sizeof(Centroid), MPI_BYTE, 0, MPI_COMM_WORLD);
    allocateMemoryAndCopyToGPU(&localSongs_d, &centroids_d, localSongs, centroids, localN, k * sizeof(Centroid));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        callAssignSongToCluster(localSongs_d, centroids_d, localN, k);
        gpuErrorCheck();
        
        resetCentroids(centroids_d, k);

        callCalculateNewCentroids(localSongs_d, centroids_d, localN);
        gpuErrorCheck();

        copyCentroidsToHost(centroids, centroids_d, k);
        
        MPI_Gather(centroids, k * sizeof(Centroid), MPI_BYTE,
            allCentroids.data(), k * sizeof(Centroid), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // Combine centroids from all processes (only at the root process)
            for (int i = 0; i < k; ++i) {
                float sumFeature1 = 0.0, sumFeature2 = 0.0, sumFeature3 = 0.0;
                int totalSize = 0;

                // Iterate through centroids received from all processes
                for (int j = 0; j < world_size; ++j) {
                    int idx = j * k + i;
                    sumFeature1 += allCentroids[idx].feature1;
                    sumFeature2 += allCentroids[idx].feature2;
                    sumFeature3 += allCentroids[idx].feature3;
                    totalSize += allCentroids[idx].cluster_size;
                }

                // Calculate the average for each feature of the centroid
                if (totalSize > 0) {
                    centroids[i].feature1 = sumFeature1 / totalSize;
                    centroids[i].feature2 = sumFeature2 / totalSize;
                    centroids[i].feature3 = sumFeature3 / totalSize;
                }
            }
        }


        // Broadcast the updated centroids to all processes
        MPI_Bcast(centroids, k * sizeof(Centroid), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Copy updated centroids back to device for next iteration
        copyCentroidsToDevice(centroids_d, centroids, k);
    }

    // Copy final results back to host
    copySongsToHost(localSongs, localSongs_d, localN);

    // Free GPU memory
    freeGPUMemory(localSongs_d, centroids_d);

    // Clean up
    delete[] centroids;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Define MPI datatype for Song
    MPI_Datatype MPI_Song;
    MPI_Type_contiguous(sizeof(Song), MPI_BYTE, &MPI_Song);
    MPI_Type_commit(&MPI_Song);

    // Prepare data
    std::vector<Song> allSongs;
    std::vector<Song> localSongs;
    std::vector<double*> rawData;

    auto start = std::chrono::high_resolution_clock::now();
    if (world_rank == 0)
    {

        int maxLines = 250000;
        if (argc > 1)
        {
            maxLines = std::stoi(argv[1]);
            if (maxLines < 0 || maxLines > MAX_LINES)
                maxLines = MAX_LINES;
            std::cout << "maxLines = " << maxLines << std::endl;
        }
        rawData = parseCSV(maxLines);  // This will hold raw feature data from the CSV

        // Transform rawData into a vector of Song structures
        for (double* features : rawData) {
            allSongs.push_back(Song(features[0], features[6], features[8]));
            delete[] features;
        }
    }

    // Distribute data among MPI processes
    distributeData(MPI_Song, allSongs, localSongs, world_size, world_rank);
    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Process " << world_rank << ": Parsed and distributed data in " << duration.count() << " seconds, received " << localSongs.size() << " songs.\n";

    kMeansCUDAMPI(localSongs.data(), localSongs.size(), 100, 5, world_size, world_rank);

    gatherResults(MPI_Song, allSongs, localSongs, world_size, world_rank);

    if (world_rank == 0) {
        auto endkMeans = std::chrono::high_resolution_clock::now();
        duration = endkMeans - endParse;
        std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;
        std::vector<std::string> featureNames = {"danceability", "acousticness", "liveness"};
        std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";

        std::vector<double*> csvData;
        for (const Song& song : allSongs) {
            double* row = new double[4];
            row[0] = song.feature1;
            row[1] = song.feature2;
            row[2] = song.feature3;
            row[3] = static_cast<double>(song.cluster);  // Cast int to double
            csvData.push_back(row);
        }

        writeCSV(csvData, "src/data/output.csv", header);

        // Clean up the allocated arrays for CSV data
        for (double* row : csvData) {
            delete[] row;
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
