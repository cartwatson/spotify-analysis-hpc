#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "csv.hpp"

struct Instance {
    double danceablility, acousticness, liveness;
    int cluster;
    double minDist;

    Instance(double dance, double acoustic, double live) :
        danceablility(dance),
        acousticness(acoustic),
        liveness(live),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    /**
     * Compute the euclidean distance between this instance and another
     */
    double distance(Instance other)
    {
        double danceDiff = other.danceablility - danceablility;
        double acousticDiff = other.acousticness - acousticness;
        double liveDiff = other.liveness - liveness;
        return std::sqrt(danceDiff * danceDiff + acousticDiff * acousticDiff + liveDiff * liveDiff);
    }
};

/**
 * Reads in the CSV file into a vector of instances
 * @param numLines number of lines to read in, -1 for all
 * @return vector of points
 */
std::vector<Instance> readCSV(int numLines) {
    std::vector<Instance> insts;

    csv::CSVReader reader("tracks_features.csv"); // Read in the CSV file
    numLines = numLines > -1 ? numLines : __INT_MAX__; // Only read in the first numLines lines
    for (csv::CSVRow& row: reader)
    {
        int currentRow = reader.n_rows();
        if (currentRow >= numLines) break;
        if (currentRow % 100000 == 0) std::cout << "Read " << currentRow << " lines" << std::endl;
        double danceability = row["danceability"].get<double>();
        double acousticness = row["acousticness"].get<double>();
        double liveness = row["liveness"].get<double>();
        insts.push_back(Instance(danceability, acousticness, liveness));
    }

    return insts;
}

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeans(std::vector<Instance>* instances, int epochs, int k) {
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
            sumDance[clusterId] += it->danceablility;
            sumAcoustic[clusterId] += it->acousticness;
            sumLive[clusterId] += it->liveness;

            it->minDist = __DBL_MAX__;  // reset distance
        }
        // Compute the new centroids
        for (std::vector<Instance>::iterator c = begin(centroids); c != end(centroids); ++c)
        {
            int clusterId = c - begin(centroids);
            c->danceablility = sumDance[clusterId] / nPoints[clusterId];
            c->acousticness = sumAcoustic[clusterId] / nPoints[clusterId];
            c->liveness = sumLive[clusterId] / nPoints[clusterId];
        }
    }

    // Write to csv
    std::ofstream myfile;
    myfile.open("output.csv");
    myfile << "danceability,acousticness,liveness,cluster" << std::endl;

    for (std::vector<Instance>::iterator it = instances->begin(); it != instances->end(); ++it)
        myfile << it->danceablility << "," << it->acousticness << "," << it->liveness << "," << it->cluster << std::endl;

    myfile.close();
}

int main(int argc, char** argv) {
    int numLines = -1;
    if (argc > 1)
        numLines = std::stoi(argv[1]);
    auto start = std::chrono::high_resolution_clock::now(); // Record start time
    std::vector<Instance> insts = readCSV(numLines);
    auto end = std::chrono::high_resolution_clock::now(); // Record end time

    std::chrono::duration<double> duration = end - start; // Calculate duration in seconds
    std::cout << "Read " << insts.size() << " instances in " << duration.count() << " seconds" << std::endl;

    // Run k-means with 100 iterations and for 5 clusters
    std::cout << "Running k-means..." << std::endl;
    kMeans(&insts, 100, 5);

    return 0;
}
