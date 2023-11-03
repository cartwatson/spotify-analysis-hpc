# Spotify Analysis, Parallel Implementation Comparison

## Overview

This project aims to implement a parallel K-Means clustering algorithm to categorize a dataset of 1.2 million songs from Spotify using various metrics provided. The algorithm is designed to work flexibly with different numbers of clusters (K), allowing users to experiment and visualize the clustering behavior based on their selected value of K.

## Requirements

- C++11 or greater
- Python >3.10.xx
- Python Libraries:
  - matplotlib
  - pandas
  - seaborn

## Usage

1. Ensure all requirements are installed
   - Run `pip install -r src/python/requirements.txt` to install necessary python libraries
2. Run `scripts/build.sh` to build your desired implementation
3. Run `src/python/visualization.py` to visualize results of your implementation

## Contributors

| Name | Github | LinkedIn |
|---|---|---|
| Carter Watson  | [Github](https://www.github.com/cartwatson) | [LinkedIn](https://www.linkedin.com/in/cartwatson) |  
| Caden Maxwell  | [Github](https://github.com/caden-maxwell)  | [LinkedIn](https://www.linkedin.com/in/cadenmaxwell/) |
| Jaxson Madison | Github | LinkedIn |

## References 

- [kaggle data](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
- [k-means clustering](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
