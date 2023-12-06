import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from CSV files
cuda_df = pd.read_csv('logs/cuda_scaling_results.csv')
cuda_mpi_df = pd.read_csv('logs/cuda_mpi_scaling_results.csv')

# Plotting K-means Execution Time for CUDA and CUDA/MPI implementations
plt.figure(figsize=(12, 6))
plt.plot(cuda_df['Block Size'], cuda_df['K-means Execution Time'], marker='o', label='CUDA')
plt.plot(cuda_mpi_df['Block Size'], cuda_mpi_df['K-means Execution Time'], marker='o', label='CUDA/MPI')

plt.title('K-means Execution Time vs Block Size')
plt.xlabel('Block Size')
plt.ylabel('K-means Execution Time (seconds)')
plt.legend()
plt.grid(True)

# Save the plot
save_path = 'src/data'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path, 'kmeans_execution_time_vs_block_size.png'))

# plt.show()
