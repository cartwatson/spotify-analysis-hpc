#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <random>

#include "util.cpp"
#include "instance.cpp"

MPI_Datatype MPI_Instance;
std::ostream& operator<<(std::ostream& os, const Instance& inst) {
    os << "Danceability: " << inst.danceability 
       << ", Acousticness: " << inst.acousticness 
       << ", Liveness: " << inst.liveness 
       << ", Cluster: " << inst.cluster;
    return os;
}

void kMeansDistributed(std::vector<Instance>& instances, int epochs, int k, int world_size, int world_rank) {
    int n = instances.size();
    std::vector<Instance> centroids(k);
    std::vector<Instance> allCentroids;

    // Validate dataset
    if (instances.empty()) {
        std::cerr << "Error: Dataset is empty." << std::endl;
        return;
    }

    // Randomly initialize centroids
    if (world_rank == 0) {
        std::mt19937 rng(static_cast<unsigned>(std::time(0)));
        std::uniform_int_distribution<int> uni(0, n - 1);

        for (int i = 0; i < k; ++i) {
            int randomIndex = uni(rng);
            if (randomIndex < 0 || randomIndex >= n) {
                std::cerr << "Error: Random index out of bounds." << std::endl;
                return;
            }
            centroids[i] = instances[randomIndex];
            centroids[i].cluster = i;
        }
    }

    // Broadcast centroids from the root process to all other processes
    int err = MPI_Bcast(centroids.data(), k, MPI_Instance, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS) {
        std::cerr << "MPI_Bcast error: " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, err); // Abort on error
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {        
        // Assign points to the nearest centroid
        for (Instance& inst : instances) {  // Iterates over each instance in the dataset
            double minDist = __DBL_MAX__;  // Initialize minimum distance to the maximum possible double value
            int closestCluster = -1;  // Initialize the closest cluster index to -1 (no cluster)

            for (Instance& centroid : centroids) {  // Iterates over each centroid
                double dist = centroid.distance(inst);  // Calculate the distance from the current centroid to the instance
                if (dist < minDist) {  // Check if this distance is less than the current minimum distance
                    minDist = dist;  // Update the minimum distance
                    closestCluster = centroid.cluster;  // Update the closest cluster to the current centroid's cluster
                }
            }
            inst.cluster = closestCluster;  // Assign the instance to the cluster of the closest centroid
        }



        // Calculate new centroids locally
        std::vector<Instance> newCentroids(k);  // Create a vector to store new centroids for each cluster
        std::vector<int> counts(k, 0);  // Create a vector to count the number of instances in each cluster

        for (Instance& inst : instances) {  // Iterate over each instance
            //std::cout << "Cluster: " << inst.cluster << std::endl;
            newCentroids[inst.cluster].danceability += inst.danceability;  // Sum danceability for the cluster
            newCentroids[inst.cluster].acousticness += inst.acousticness;  // Sum acousticness for the cluster
            newCentroids[inst.cluster].liveness += inst.liveness;  // Sum liveness for the cluster
            newCentroids[inst.cluster].cluster = inst.cluster;
            counts[inst.cluster]++;  // Increment the count of instances in this cluster
        }

        // Check if the current process is the root process (usually process 0)
        if (world_rank == 0) {
            allCentroids.resize(k * world_size);
        }

        //Print newCentroids before MPI_Gather
        // std::cout << "Process " << world_rank << " newCentroids before gather:\n";
        // for (int i = 0; i < newCentroids.size(); i++) {
        //         std::cout << "Cluster " << newCentroids[i].cluster << ": "
        //                 << "Danceability: " << newCentroids[i].danceability
        //                 << ", Acousticness: " << newCentroids[i].acousticness
        //                 << ", Liveness: " << newCentroids[i].liveness << "\n";
        // }

        MPI_Gather(newCentroids.data(), k, MPI_Instance,
                allCentroids.data(), k, MPI_Instance, 0, MPI_COMM_WORLD);

        //Print allCentroids after MPI_Gather on the root process
        // if (world_rank == 0) {
        //     std::cout << "Root process allCentroids after gather:\n";
        //     //for (const auto& centroid : allCentroids) {
        //     for (int i = 0; i < allCentroids.size(); i++) {
        //         std::cout << "Cluster " << allCentroids[i].cluster << ": "
        //                 << "Danceability: " << allCentroids[i].danceability
        //                 << ", Acousticness: " << allCentroids[i].acousticness
        //                 << ", Liveness: " << allCentroids[i].liveness << "\n";
        //     }
        // }

        // Aggregate centroids at the root and then broadcast them
        if (world_rank == 0) {  // Check if this is the root process
            // Combine centroids from all processes
            for (int cluster = 0; cluster < k; cluster++) {  // Iterate over each centroid
                centroids[cluster].danceability = 0;  // Reset danceability for centroid 'i'
                centroids[cluster].acousticness = 0;  // Reset acousticness for centroid 'i'
                centroids[cluster].liveness = 0;  // Reset liveness for centroid 'i'
                int totalCount = 0;  // Initialize a counter for the total number of instances in this centroid

                for (Instance& centroid : allCentroids) {
                    if (centroid.cluster == cluster) {
                        centroids[cluster].danceability += centroid.danceability;
                        centroids[cluster].acousticness += centroid.acousticness;
                        centroids[cluster].liveness += centroid.liveness;
                        totalCount += counts[cluster];
                        //std::cout << "Total Count " << totalCount << std::endl;
                    }
                }

                if (totalCount > 0) {  // Check if the centroid has any instances assigned
                    // Average the features of the centroid based on the total count
                    centroids[cluster].danceability /= totalCount;
                    centroids[cluster].acousticness /= totalCount;
                    centroids[cluster].liveness /= totalCount;
                }
            }

            // std::cout << "Root process centroids after agg:\n";
            // for (const auto& centroid : centroids) {
            //     std::cout << "Cluster " << centroid.cluster << ": "
            //             << "Danceability: " << centroid.danceability
            //             << ", Acousticness: " << centroid.acousticness
            //             << ", Liveness: " << centroid.liveness << "\n";
            // }
        }

        // Broadcast the updated centroids to all processes
        MPI_Bcast(centroids.data(), k, MPI_Instance, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int maxLines = 250000;
    if (argc > 1) {
        maxLines = std::stoi(argv[1]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    int totalInstances = 0;

    // Declarations for sendCounts and displacements
    std::vector<int> sendCounts(world_size), displacements(world_size);
    std::vector<Instance> allInstances;

    if (world_rank == 0) {
        // Parse CSV and fill allInstances
        std::vector<double*> data = parseCSV(maxLines);
        for (double* row : data) {
            allInstances.push_back(Instance(row));
        }
        totalInstances = allInstances.size();

        int instancesPerProcess = totalInstances / world_size;
        int remaining = totalInstances % world_size;

        // Fill sendCounts and displacements based on allInstances
        for (int i = 0; i < world_size; ++i) {
            sendCounts[i] = instancesPerProcess + (i < remaining ? 1 : 0);
            displacements[i] = (i == 0 ? 0 : displacements[i - 1] + sendCounts[i - 1]);
        }
    }

    // Broadcast totalInstances, sendCounts, and displacements to all processes
    MPI_Bcast(&totalInstances, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(sendCounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Define MPI_Instance datatype
    MPI_Type_contiguous(sizeof(Instance), MPI_BYTE, &MPI_Instance);
    MPI_Type_commit(&MPI_Instance);

    // Allocate space for local instances on each process
    std::vector<Instance> localInstances(sendCounts[world_rank]);

    // Distribute instances to each process
    MPI_Scatterv(allInstances.data(), sendCounts.data(), displacements.data(), MPI_Instance, 
             localInstances.data(), sendCounts[world_rank], MPI_Instance, 0, MPI_COMM_WORLD);

    auto endParse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endParse - start;
    std::cout << "Process " << world_rank << ": Parsed and distributed data in " << duration.count() << " seconds, received " << localInstances.size() << " instances\n";

    kMeansDistributed(localInstances, 100, 4, world_size, world_rank);

    // Prepare for gathering
    std::vector<int> recvCounts(world_size), displs(world_size);
    if (world_rank == 0) {
        allInstances.resize(totalInstances);
        for (int i = 0; i < world_size; ++i) {
            recvCounts[i] = sendCounts[i];
            displs[i] = (i == 0 ? 0 : displs[i - 1] + recvCounts[i - 1]);
        }
    }

    // Gather the updated instances with correct cluster assignments
    MPI_Gatherv(localInstances.data(), localInstances.size(), MPI_Instance,
                allInstances.data(), recvCounts.data(), displs.data(), MPI_Instance, 0, MPI_COMM_WORLD);


    MPI_Type_free(&MPI_Instance);

    // Write to file at the root process
    if (world_rank == 0) {
        std::ofstream myfile;
        myfile.open("src/data/output.csv");
        myfile << "danceability,acousticness,liveness,cluster\n";

        for (const Instance& inst : allInstances) {
            myfile << inst.danceability << "," << inst.acousticness << "," << inst.liveness << "," << inst.cluster << "\n";
        }

        myfile.close();
        auto endkMeans = std::chrono::high_resolution_clock::now();
        duration = endkMeans - endParse;
        std::cout << "Finished distributed k-means and output in " << duration.count() << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}