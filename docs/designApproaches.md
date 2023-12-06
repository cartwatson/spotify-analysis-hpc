# Design Approaches

*All approaches seek to accomplish the same task: perform k-means clustering on three features of 1.2M songs from Spotify.*

## 1. Serial - [`Basic Implementation`](/src/serial.cpp)

For our serial implementation we followed closely the advice from the provided site [reasonabledeviations.com](https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/).  Following the instructions provided our first step was to create functions and structs to help store data and accomplish tasks that we would need to do for every implementation.  The struct for storing song data, and the utility functions for parsing from and writing to a csv were created as a part of this step.

Next was initializing the clusters, only slight modifications here were need from the instructions provided, we used our newly created song struct instead of the default point, we also implemented a feature to only randomize the clusters if we're testing, to ensure that we're about to get reproducible results otherwise.  The next two steps were to assign points to a cluster and to compute new centroids, we didn't deviate significantly from the instructions at this step but we did ensure that we included our third axis here.

In conclusion, we successfully adapted the instructions provided to fit our song data.  This was accomplished by creating specific structs and util functions for better code reusability as move forward creating the parallel implementations.  We also made sure to use three features of the songs for a more in depth visualization.  Overall we laid a strong foundation for the upcoming implementations.

## 2. Parallel Shared Memory CPU - [`OpenMP Implementation`](/src/omp.cpp)

For this implementation we took advantage of OpenMP.  The implementation is mostly duplicated code from the serial implementation, though most of this code is not able to be duplicated in an efficient manner due to the nature of OpenMP's pragma statements.

The main difference between this implementation and serial are the inclusion of two `# pragma` statements, however it's not to be understated how much heavy lifting these two statements do.  The first pragma statements comes on line 41 and creates a parallel zone and allows OpenMP to parallelize the loop that calculates the distance between the song and the clusters center and reassigns the song if necessary.  The other pragma statement parallelized the section where we were appending songs to the centroids.  Alongside the parallelization for appending songs it also ensured the methodology we were using wouldn't encounter race conditions.

Overall our implementation for OpenMP utilized heavily the infrastructure put in place by the serial implementation.  With only a few changes to our original code we were able to parallelize the program making it much more time and resource efficient.

## 3. Parallel Shared Memory GPU - [`Cuda Implementation`](/src/cuda.cu)

The cuda implementation keeps almost the exact same driver code as the serial and OpenMP implementations.  The only significant change is the function being called from the driver code.  At this point it's likely there's a much cleaner approach that we could've taken to not duplicate driver code and pass the K-Means Cluster function from each implementation into a util function.

The implementation of the K-Means Cluster Algorithm is exactly as one would expect it to be, it follows the same pattern that's been established with previous implementations.  The benefit that implementing this with Cuda allows is that we can run on the GPU, this reduces memory overhead making this implementation significantly faster.  However it does also limit this implementation to only running on Nvidia GPUs.

To sum it up, Cuda allows for faster parallelization due to it allowing us to run an implementation of the K-Means Cluster Algorithm on the GPU instead of the CPU.

## 4. Distributed Memory CPU - [`MPI Implementation`](/src/mpi.cpp)

The implementation for MPI sticks to the core of what we've built with the serial implementation with a few major tweaks in the driver section and some minor changes in the in the implementation of the algorithm to better leverage parallel processing with distributed memory.

The changes in the driver function are primarily to setup the MPI parallel environment.  The changes ensure that the data is created/parsed in a way that allows MPI to parallelize efficiently.  Data is then broadcast to all nodes in MPI, and the algorithm is called.  The K-Means Clustering Algorithm quite obviously does not fundamentally change but there are small optimizations for MPI that make it different from our other implementations.  The most important tweaks are for error checking and ensuring that there are no race conditions. The driver code finishes by syncing all information back together and writing to the output file.

Ultimately, we retained the core methodology of our serial K-Means algorithm while introducing significant modifications in the driver function and minor optimizations in the algorithm for efficient distributed memory parallel processing on the CPU.

## 5. Distributed Memory GPU - [`Cuda & MPI Implementation`](/src/)

TODO: implement
