#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "parser.cpp"
#include "instance.cpp"

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansSerial(std::vector<Instance>* instances, int epochs, int k) {
    int n = instances->size();

    // Randomly initialise centroids
    // The index of the centroid within the centroids vector
    // represents the cluster label.
    std::vector<Instance> centroids;
    srand(time(0));
    for (int i = 0; i < k; ++i)
        centroids.push_back(instances->at(rand() % n));

    for (int i = 0; i < epochs; ++i)
    {
        // For each centroid, compute distance from centroid to each point
        // and update point's cluster if necessary
        for (std::vector<Instance>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);

            for (std::vector<Instance>::iterator it = instances->begin(); it != instances->end(); ++it)
            {
                Instance inst = *it;
                double dist = c->distance(inst);
                if (dist < inst.minDist)
                {
                    inst.minDist = dist;
                    inst.cluster = clusterId;
                }
                *it = inst;
            }
        }

        // Create vectors to keep track of data needed to compute means
        std::vector<int> nPoints;
        std::vector<double> sumDance, sumAcoustic, sumLive;
        for (int j = 0; j < k; ++j)
        {
            nPoints.push_back(0);
            sumDance.push_back(0.0);
            sumAcoustic.push_back(0.0);
            sumLive.push_back(0.0);
        }

        // Iterate over points to append data to centroids
        for (std::vector<Instance>::iterator it = instances->begin(); it != instances->end(); ++it)
        {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumDance[clusterId] += it->danceability;
            sumAcoustic[clusterId] += it->acousticness;
            sumLive[clusterId] += it->liveness;

            it->minDist = __DBL_MAX__;  // reset distance
        }
        // Compute the new centroids
        for (std::vector<Instance>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            c->danceability = sumDance[clusterId] / nPoints[clusterId];
            c->acousticness = sumAcoustic[clusterId] / nPoints[clusterId];
            c->liveness = sumLive[clusterId] / nPoints[clusterId];
        }
    }

    // Write to csv
    std::ofstream myfile;
    myfile.open("src/data/output.csv");
    myfile << "danceability,acousticness,liveness,cluster" << std::endl;

    for (std::vector<Instance>::iterator it = instances->begin(); it != instances->end(); ++it)
        myfile << it->danceability << "," << it->acousticness << "," << it->liveness << "," << it->cluster << std::endl;

    myfile.close();
}

int main(int argc, char** argv)
{
    int maxLines = 250000;
    if (argc > 1)
        maxLines = std::stoi(argv[1]);

    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<double>> data = parseCSV(maxLines);
    std::vector<Instance> instances;
    for (std::vector<double> row : data)
        instances.push_back(Instance(row));

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Parsed data in " << duration.count() << " seconds" << std::endl;
    
    std::cout << "Running k-means..." << std::endl;
    kMeansSerial(&instances, 100, 4);
    auto endkMeans = std::chrono::high_resolution_clock::now();
    duration = endkMeans - endParse;
    std::cout << "Finished k-means in " << duration.count() << " seconds" << std::endl;

    return 0;
}