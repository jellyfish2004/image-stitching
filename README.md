# Parallel Panorama Stitching

This project implements a high-performance panorama stitching pipeline in C++, optimized using **OpenMP** for multi-core CPU parallelism and **OpenACC** for GPU acceleration.

## Project Structure

- `serial/`: Contains the baseline serial implementation (includes some openacc, but compile with g++ to get serial results)
- `parallel/`: Contains the optimized parallel implementation (OpenMP + OpenACC).
- `include/`: Shared header files.
- `report/`: Technical report and performance analysis.

## Prerequisites

- **Compiler**: NVIDIA HPC SDK (`nvc++`) is required for the parallel version to support OpenACC. Standard `g++` can be used for the serial version.
- **Hardware**: An NVIDIA GPU is required to benefit from OpenACC acceleration.
- **Libraries**: Standard C++17 libraries.

## Compilation

### Parallel Version (Recommended)
To compile the parallel version with OpenMP and OpenACC support:
```bash
nvc++ -mp -acc=gpu -std=c++17 -Iinclude parallel/*.cpp -o stitch -lm
```
*Note: Use `-O2` for production performance or `-O0` for debugging/benchmarking purposes.*

### Serial Version
To compile the baseline serial version:
```bash
g++ -std=c++17 -Iinclude serial/*.cpp -o app -O2
```

## Usage

Run the stitched binary by providing two input images:
```bash
./stitch image1.jpg image2.jpg
```

The program will output:
- `stitched.png`: The final merged panorama.
- `keypoints1.png` / `keypoints2.png`: Visualizations of detected SIFT features.
- `matches.png`: Visualization of feature matching between the two images.

## Performance Benchmarking

A Python script is provided to automate performance testing:
```bash
python3 benchmark.py --runs 15
```
This will generate `benchmark_results.csv` containing average execution times for each pipeline stage.
