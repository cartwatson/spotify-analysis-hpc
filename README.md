# Spotify Analysis, Parallel Implementation Comparison

## Overview

This project aims to implement a parallel K-Means clustering algorithm to categorize a dataset of 1.2 million songs from Spotify using various metrics provided. The algorithm is designed to work flexibly with different numbers of clusters (K), allowing users to experiment and visualize the clustering behavior based on their selected value of K.

## Documentation

- [Design Approaches](docs/designApproaches.md)
- [Build and Run Instructions](src/README.md)
- Scaling Studies
  - [Serial vs Shared Memory](docs/scalingStudies/serialVersusSharedMemory.md)
  - [Distributed Memory: CPU vs GPU](docs/scalingStudies/distributedMemory.md)

## Requirements

- C++11 or greater
- Python >3.10
- Python Libraries: [`scripts/requirements.txt`](scripts/requirements.txt)

## Usage

*All files should be run from the root directory*

- Use `scripts/build_and_run.sh` to build and run your desired implementation
  - To manually build and run implementations you can find instructions at [`src/README.md`](src/README.md)
- Use `scripts/validation.sh` to validate parallel implementations produce the same results as the serial implementation
- Use `scripts/visualize.sh` to visualize results of your most recently run implementation
  1. to run this on wsl (Ubuntu 20.04) run the following commands `sudo apt install x11-apps libxcb-* libxkb*`
  2. then in PowerShell run `wsl --update`, `wsl --shutdown`
  3. back in wsl `pip uninstall pyqt5 pyqt5-qt5 pyqt5-sip`
  4. if on an AMD cpu `pip install pyqt6`
  5. if on an Intel cpu `pip install glfw`

## Contributors

| Name | Github | LinkedIn |
|---|---|---|
| Carter Watson  | [Github](https://www.github.com/cartwatson) | [LinkedIn](https://www.linkedin.com/in/cartwatson) |  
| Caden Maxwell  | [Github](https://github.com/caden-maxwell)  | [LinkedIn](https://www.linkedin.com/in/cadenmaxwell/) |
| Jaxson Madison | [Github](https://github.com/JaxsonM) | LinkedIn |

## References 

- [Kaggle data](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
- [k-means clustering explanation and serial implementation](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
