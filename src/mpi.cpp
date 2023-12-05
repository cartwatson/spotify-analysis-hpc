#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

#include "util.cpp"
#include "song.cpp"

MPI_Datatype MPI_Song;
std::ostream& operator<<(std::ostream& os, const Song& song)
{
    os << "Danceability: " << song.feature1 
       << ", Acousticness: " << song.feature2 
       << ", Liveness: " << song.feature3 
       << ", Cluster: " << song.cluster;
    return os;
}

void kMeansDistributed(std::vector<Song>& songs, int epochs, int k, int world_size, int world_rank)
{
    int n = songs.size();
    std::vector<Song> centroids(k);
    std::vector<Song> allCentroids;

    // Validate dataset
    if (songs.empty()) {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return;
    }

    // Randomly initialize centroids
    if (world_rank == 0)
    {
        #ifdef TESTING
        std::mt19937 rng(123);
        #else
        std::mt19937 rng(static_cast<unsigned>(std::time(0)));
        #endif
        std::uniform_int_distribution<int> uni(0, n - 1);

        for (int i = 0; i < k; ++i)
        {
            int randomIndex = uni(rng);
            if (randomIndex < 0 || randomIndex >= n)
            {
                std::cerr << "Error: Random index out of bounds." << std::endl;
                return;
            }
            centroids[i] = songs[randomIndex];
            centroids[i].cluster = i;
        }
    }

    // Broadcast centroids from the root process to all other processes
    int err = MPI_Bcast(centroids.data(), k, MPI_Song, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI_Bcast error: " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err); // Abort on error
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Assign points to the nearest centroid
        for (Song& song : songs) {  // Iterates over each song in the dataset
            double minDist = __DBL_MAX__;  // Initialize minimum distance to the maximum possible double value
            int closestCluster = -1;  // Initialize the closest cluster index to -1 (no cluster)

            for (Song& centroid : centroids) {  // Iterates over each centroid
                double dist = centroid.distance(song);  // Calculate the distance from the current centroid to the song
                if (dist < minDist) {  // Check if this distance is less than the current minimum distance
                    minDist = dist;  // Update the minimum distance
                    closestCluster = centroid.cluster;  // Update the closest cluster to the current centroid's cluster
                }
            }
            song.cluster = closestCluster;  // Assign the song to the cluster of the closest centroid
        }

        // Calculate new centroids locally
        std::vector<Song> newCentroids(k);  // Create a vector to store new centroids for each cluster
        std::vector<int> counts(k, 0);  // Create a vector to count the number of songs in each cluster

        for (Song& song : songs) {  // Iterate over each song
            //std::cout << "Cluster: " << song.cluster << std::endl;
            newCentroids[song.cluster].feature1 += song.feature1;  // Sum danceability for the cluster
            newCentroids[song.cluster].feature2 += song.feature2;  // Sum acousticness for the cluster
            newCentroids[song.cluster].feature3 += song.feature3;  // Sum liveness for the cluster
            newCentroids[song.cluster].cluster = song.cluster;
            counts[song.cluster]++;  // Increment the count of songs in this cluster
        }

        // Check if the current process is the root process (usually process 0)
        if (world_rank == 0)
        {
            allCentroids.resize(k * world_size);
        }

        MPI_Gather(newCentroids.data(), k, MPI_Song,
                allCentroids.data(), k, MPI_Song, 0, MPI_COMM_WORLD);

        // Aggregate centroids at the root and then broadcast them
        if (world_rank == 0) // Check if this is the root process
        {
            // Combine centroids from all processes
            for (int cluster = 0; cluster < k; cluster++) // Iterate over each centroid
            {
                centroids[cluster].feature1 = 0;  // Reset danceability for centroid 'i'
                centroids[cluster].feature2 = 0;  // Reset acousticness for centroid 'i'
                centroids[cluster].feature3 = 0;  // Reset liveness for centroid 'i'
                int totalCount = 0;  // Initialize a counter for the total number of songs in this centroid

                for (Song& centroid : allCentroids)
                {
                    if (centroid.cluster == cluster)
                    {
                        centroids[cluster].feature1 += centroid.feature1;
                        centroids[cluster].feature2 += centroid.feature2;
                        centroids[cluster].feature3 += centroid.feature3;
                        totalCount += counts[cluster];
                        //std::cout << "Total Count " << totalCount << std::endl;
                    }
                }

                if (totalCount > 0) // Check if the centroid has any instances assigned
                {
                    // Average the features of the centroid based on the total count
                    centroids[cluster].feature1 /= totalCount;
                    centroids[cluster].feature2 /= totalCount;
                    centroids[cluster].feature3 /= totalCount;
                }
            }
        }

        // Broadcast the updated centroids to all processes
        MPI_Bcast(centroids.data(), k, MPI_Song, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int maxLines = 250000;
    if (argc > 1)
    {
        maxLines = std::stoi(argv[1]);
        if (maxLines < 0 || maxLines > MAX_LINES)
            maxLines = MAX_LINES;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int totalSongs = 0;

    // Declarations for sendCounts and displacements
    std::vector<int> sendCounts(world_size), displacements(world_size);
    std::vector<Song> allSongs;

    if (world_rank == 0)
    {
        std::cout << "maxLines = " << maxLines << std::endl;
        // Parse CSV and fill allSongs
        std::vector<double*> data = parseCSV(maxLines);
        for (double* row : data)
        {
            // Grab the features for the song
            Song song(row[0], row[6], row[8]);
            allSongs.push_back(song);
        }
        totalSongs = allSongs.size();

        int songsPerProcess = totalSongs / world_size;
        int remaining = totalSongs % world_size;

        // Fill sendCounts and displacements based on allSongs
        for (int i = 0; i < world_size; ++i)
        {
            sendCounts[i] = songsPerProcess + (i < remaining ? 1 : 0);
            displacements[i] = (i == 0 ? 0 : displacements[i - 1] + sendCounts[i - 1]);
        }
    }

    std::vector<std::string> featureNames = {"danceability", "acousticness", "liveness"};

    // Broadcast totalSongs, sendCounts, and displacements to all processes
    MPI_Bcast(&totalSongs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendCounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Define MPI_Song datatype
    MPI_Type_contiguous(sizeof(Song), MPI_BYTE, &MPI_Song);
    MPI_Type_commit(&MPI_Song);

    // Allocate space for local songs on each process
    std::vector<Song> localSongs(sendCounts[world_rank]);

    // Distribute songs to each process
    MPI_Scatterv(allSongs.data(), sendCounts.data(), displacements.data(), MPI_Song, 
             localSongs.data(), sendCounts[world_rank], MPI_Song, 0, MPI_COMM_WORLD);

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Process " << world_rank << ": Parsed and distributed data in " << duration.count() << " seconds, received " << localSongs.size() << " songs.\n";

    kMeansDistributed(localSongs, 100, 4, world_size, world_rank);

    // Prepare for gathering
    std::vector<int> recvCounts(world_size), displs(world_size);
    if (world_rank == 0)
    {
        allSongs.resize(totalSongs);
        for (int i = 0; i < world_size; ++i) {
            recvCounts[i] = sendCounts[i];
            displs[i] = (i == 0 ? 0 : displs[i - 1] + recvCounts[i - 1]);
        }
        auto endkMeans = std::chrono::high_resolution_clock::now();
        duration = endkMeans - endParse;
        std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;
    }

    // Gather the updated songs with correct cluster assignments
    MPI_Gatherv(localSongs.data(), localSongs.size(), MPI_Song,
                allSongs.data(), recvCounts.data(), displs.data(), MPI_Song, 0, MPI_COMM_WORLD);

    MPI_Type_free(&MPI_Song);

    if (world_rank == 0)
    {
        std::cout << "Writing output to file..." << std::endl;
        std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";
        std::vector<double*> data;
        for (Song& song : allSongs)
        {
            double* row = new double[4];
            row[0] = song.feature1;
            row[1] = song.feature2;
            row[2] = song.feature3;
            row[3] = song.cluster;
            data.push_back(row);
        }
        writeCSV(data, "src/data/output.csv", header);
    }

    MPI_Finalize();
    return 0;
}