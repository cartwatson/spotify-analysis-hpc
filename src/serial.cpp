#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

#include "parser.cpp"
#include "song.cpp"

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansSerial(std::vector<Song>& songs, int epochs, int k) {
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
        // and update point's cluster if necessary
        for (Song& c : centroids)
            for (Song& song : songs)
            {
                double dist = c.distance(song);
                if (dist < song.minDist)
                {
                    song.minDist = dist;
                    song.cluster = c.cluster;
                }
            }

        // Create vectors to keep track of data needed to compute means
        std::vector<int> nSongs;
        std::vector<double> sumDance, sumAcoustic, sumLive;
        for (int j = 0; j < k; ++j)
        {
            nSongs.push_back(0);
            sumDance.push_back(0.0);
            sumAcoustic.push_back(0.0);
            sumLive.push_back(0.0);
        }

        // Iterate over points to append data to centroids
        for (Song& song : songs)
        {
            nSongs[song.cluster] += 1;
            sumDance[song.cluster] += song.danceability;
            sumAcoustic[song.cluster] += song.acousticness;
            sumLive[song.cluster] += song.liveness;

            song.minDist = __DBL_MAX__;  // reset distance
        }

        // Compute the new centroids
        for (Song& c : centroids)
        {
            c.danceability = sumDance[c.cluster] / nSongs[c.cluster];
            c.acousticness = sumAcoustic[c.cluster] / nSongs[c.cluster];
            c.liveness = sumLive[c.cluster] / nSongs[c.cluster];
        }
    }

    // Write to csv
    std::ofstream myfile;
    myfile.open("src/data/output.csv");
    myfile << "danceability,acousticness,liveness,cluster" << std::endl;

    for (Song& it : songs)
        myfile << it.danceability << "," << it.acousticness << "," << it.liveness << "," << it.cluster << std::endl;

    myfile.close();
}

int main(int argc, char** argv)
{
    int maxLines = 250000;
    if (argc > 1)
        maxLines = std::stoi(argv[1]);

    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<double*> data = parseCSV(maxLines);
    std::vector<Song> songs;
    for (double* row : data)
        songs.push_back(Song(row));

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;
    kMeansSerial(songs, 100, 4);
    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    return 0;
}