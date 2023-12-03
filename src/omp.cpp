#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

#include "parser.cpp"
#include "instance.cpp"

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansParallel(std::vector<Instance>& instances, int epochs, int k) {
    int n = instances.size();

    // Randomly initialise centroids
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    #ifdef TESTING
    std::mt19937 rng(123);
    #else
    std::mt19937 rng(static_cast<unsigned>(std::time(0)));
    #endif
    std::vector<Instance> centroids;
    std::uniform_int_distribution<int> uni(0, n - 1);
    for (int i = 0; i < k; ++i)
    {
        centroids.push_back(instances.at(uni(rng)));
        centroids[i].cluster = i;
    }

    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        for (Instance& c : centroids)
            # pragma omp parallel for
            for (Instance& inst : instances)
            {
                double dist = c.distance(inst);
                if (dist < inst.minDist)
                {
                    inst.minDist = dist;
                    inst.cluster = c.cluster;
                }
            }


        // Create vectors to keep track of data needed to compute means
        int nInsts[k];
        double sumDance[k], sumAcoustic[k], sumLive[k];
        for (int j = 0; j < k; ++j)
        {
            nInsts[j] = 0;
            sumDance[j] = 0.0;
            sumAcoustic[j] = 0.0;
            sumLive[j] = 0.0;
        }

        // Iterate over points to append data to centroids
        // Use array sum reduction: https://dvalters.github.io/optimisation/code/2016/11/06/OpenMP-array_reduction.html
        # pragma omp parallel for reduction(+:nInsts[:k], sumDance[:k], sumAcoustic[:k], sumLive[:k])
        for (Instance& inst : instances)
        {
            nInsts[inst.cluster] += 1;
            sumDance[inst.cluster] += inst.danceability;
            sumAcoustic[inst.cluster] += inst.acousticness;
            sumLive[inst.cluster] += inst.liveness;

            inst.minDist = __DBL_MAX__;  // reset distance
        }

        // Compute the new centroids
        for (Instance& c : centroids)
        {
            c.danceability = sumDance[c.cluster] / nInsts[c.cluster];
            c.acousticness = sumAcoustic[c.cluster] / nInsts[c.cluster];
            c.liveness = sumLive[c.cluster] / nInsts[c.cluster];
        }
    }

    // Write to csv
    std::ofstream myfile;
    myfile.open("src/data/output.csv");
    myfile << "danceability,acousticness,liveness,cluster" << std::endl;

    for (Instance& it : instances)
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
    std::vector<Instance> instances;
    for (double* row : data)
        instances.push_back(Instance(row));

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;
    kMeansParallel(instances, 100, 4);
    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    return 0;
}