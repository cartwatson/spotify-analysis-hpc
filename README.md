# Spotify Analysis, Parallel Implementation Comparison

## Overview

This project aims to implement a parallel K-Means clustering algorithm to categorize a dataset of 1.2 million songs from Spotify using various metrics provided. The algorithm is designed to work flexibly with different numbers of clusters (K), allowing users to experiment and visualize the clustering behavior based on their selected value of K.

## Requirements

- C++11 or greater
- Python >3.10
- Python Libraries:
  - [`scripts/requirements.txt`](scripts/requirements.txt)

## Usage

*All files should be run from the root directory*

- Use `scripts/build_and_run.sh` to build and run your desired implementation
- Use `scripts/validation.sh` to validate all implementations produce the same results
- Use `scripts/visualize.sh` to visualize results of your most recent implementation
  1. to run this on wsl (ubuntu 20.04) run the following commands `sudo apt install x11-apps libxcb-* libxkb*`
  1. then in powershell run `wsl --update`, `wsl --shutdown`
  1. back in wsl `pip uninstall pyqt5 pyqt5-qt5 pyqt5-sip`
  1. if on an amd cpu `pip install pyqt6`
  1. if on an intel cpu `pip install glfw`

## Contributors

| Name | Github | LinkedIn |
|---|---|---|
| Carter Watson  | [Github](https://www.github.com/cartwatson) | [LinkedIn](https://www.linkedin.com/in/cartwatson) |  
| Caden Maxwell  | [Github](https://github.com/caden-maxwell)  | [LinkedIn](https://www.linkedin.com/in/cadenmaxwell/) |
| Jaxson Madison | [Github](https://github.com/JaxsonM) | LinkedIn |

## References 

- [kaggle data](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
- [k-means clustering](http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/)
