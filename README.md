# TileSpMSpV

 

**TileSpMSpV** is an open source code that uses a tiled structure to optimize sparse matrix-sparse vector multiplication (SpMSpV) on GPUs. 


-------------------
## Paper information

Haonan Ji, Huimin Song, Shibo Lu, Zhou Jin, Guangming Tan and Weifeng Liu, "TileSpMSpV: A Tiled Algorithm for Sparse Matrix-Sparse Vector Multiplication on GPUs,"  Proceedings of the 51st International Conference on Parallel Processing (ICPP), 2022, pp. 1-11, DOI: https://doi.org/10.1145/3545008.3545028.

## Contact us

If you have any questions about running the code, please contact Haonan Ji. 

E-mail: haonan_ji@yeah.net

## Introduction

Sparse matrix-sparse vector multiplication (SpMSpV) is an important primitive for graph algorithms and machine learning applications. The sparsity of the input and output vectors makes its floating point efficiency in general lower than sparse matrix-vector multiplication (SpMV) and sparse matrix-matrix multiplication (SpGEMM). TileSpMSpV provides a tiled algorithm for accelerating SpMSpV on GPUs and proposes a TileBFS algorithm including three kernels called Push-CSC, Push-CSR and Pull-CSC.


<!-- ## Structure
README     instructions on installation
src        C source code
src/main.cu  testing code
Makefile   Makefile that does installation and testing
``` -->

## Installation

<!-- To use this code, you need to modify the Makefile with correct g++ installation path and use make for automatic installation. -->
To better reproduce experiment results, we suggest an NVIDIA GPU with compute capability 8.6. TileSpGEMM evaluation requires the CUDA GPU driver, the nvcc CUDA compiler, and the cuSPARSE library, all of them are included with the CUDA Toolkit. The artifacts have been tested on Ubuntu 18.04/20.04, and are expected to run correctly under other Linux distributions.

## Execution of TileSpMSpV
Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection. 

1. The command 'make' generates an executable file 'test'.
> **make**

3. Run code on matrix data. Running the program requires three parameters: the number of threads, matrix path, and BFS source point.
> **./test 16 test.mtx 0**



## Release version
Apr 13,2023 Version Alpha




 




