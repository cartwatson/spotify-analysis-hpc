import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from CSV files
serial_df = pd.read_csv('logs/serial_scaling_results.csv')
omp_df = pd.read_csv('logs/omp_scaling_results.csv')
mpi_df = pd.read_csv('logs/mpi_scaling_results.csv')

# Plotting K-means Execution Time for all implementations
plt.figure(figsize=(12, 6))
plt.plot(serial_df['Dataset Size'], serial_df['K-means Execution Time'], marker='o', label='Serial')
plt.plot(omp_df['Dataset Size'], omp_df['K-means Execution Time'], marker='o', label='OpenMP')
plt.plot(mpi_df['Dataset Size'], mpi_df['K-means Execution Time'], marker='o', label='MPI')

plt.title('K-means Execution Time vs Dataset Size')
plt.xlabel('Dataset Size')
plt.ylabel('K-means Execution Time (seconds)')
plt.legend()
plt.grid(True)

# Save the plot
save_path = 'src/data'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path, 'kmeans_execution_time_vs_block_size.png'))

# plt.show()
