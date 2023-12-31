#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

#include "util.cpp"
#include "song.cpp"

/**
 * Perform k-means clustering
 * @param songs - pointer to vector of songs
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansParallel(std::vector<Song>& songs, int epochs, int k) {
    int n = songs.size();

    // Randomly initialise centroids
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    #ifdef TESTING
    std::mt19937 rng(123);
    #else
    std::mt19937 rng(static_cast<unsigned>(std::time(0)));
    #endif
    std::vector<Song> centroids;
    std::uniform_int_distribution<int> uni(0, n - 1);
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(songs.at(uni(rng)));
        centroids[i].cluster = i;
    }

    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point
        # pragma omp parallel for
        for (Song& s : songs)
        {
            double minDist = __DBL_MAX__;
            int closestCluster = -1;
            for (Song& c : centroids)
            {
                double dist = c.distance(s);
                if (dist < minDist)
                {
                    minDist = dist;
                    closestCluster = c.cluster;
                }
            }
            s.cluster = closestCluster;
        }


        // Create vectors to keep track of data needed to compute means
        int nSongs[k];
        double sumDance[k], sumAcoustic[k], sumLive[k];
        for (int j = 0; j < k; ++j)
        {
            nSongs[j] = 0;
            sumDance[j] = 0.0;
            sumAcoustic[j] = 0.0;
            sumLive[j] = 0.0;
        }

        // Iterate over points to append data to centroids
        // Use array sum reduction: https://dvalters.github.io/optimisation/code/2016/11/06/OpenMP-array_reduction.html
        # pragma omp parallel for reduction(+:nSongs[:k], sumDance[:k], sumAcoustic[:k], sumLive[:k])
        for (Song& song : songs)
        {
            nSongs[song.cluster] += 1;
            sumDance[song.cluster] += song.feature1;
            sumAcoustic[song.cluster] += song.feature2;
            sumLive[song.cluster] += song.feature3;
        }

        // Compute the new centroids
        for (Song& c : centroids)
        {
            c.feature1 = sumDance[c.cluster] / nSongs[c.cluster];
            c.feature2 = sumAcoustic[c.cluster] / nSongs[c.cluster];
            c.feature3 = sumLive[c.cluster] / nSongs[c.cluster];
        }
    }
}

int main(int argc, char** argv)
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
    
    // Get data from CSV
    std::vector<double*> data = parseCSV(maxLines);
    std::vector<Song> songs;
    std::vector<std::string> featureNames = {"danceability", "acousticness", "liveness"};
    for (double* row : data)
        songs.push_back(Song(row[0], row[6], row[8]));

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;

    kMeansParallel(songs, 100, 5);

    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    // Send data to CSV
    std::cout << "Writing output to file..." << std::endl;
    std::string header = featureNames[0] + "," + featureNames[1] + "," + featureNames[2] + ",cluster";
    data.clear();
    for (Song& song : songs)
    {
        double* row = new double[4];
        row[0] = song.feature1;
        row[1] = song.feature2;
        row[2] = song.feature3;
        row[3] = song.cluster;
        data.push_back(row);
    }
    writeCSV(data, "src/data/output.csv", header);

    return 0;
}