#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "csv.h"
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "Timer.h"

static const int ARR_SIZE = 100000;
static const int BLOCK_SIZE = 1024;

enum Operation
{
    SumOfX1, SumOfX2, SumOfY,
    SumOfSquaresX1, SumOfSquaresX2,
    SumOfProductsX1Y, SumOfProductsX2Y, SumofProductsX1X2
};

struct MultivariateCoordinate
{
    double xs[2];
    double y;


    MultivariateCoordinate(double x1, double x2, double y)
    {
        this->xs[0] = x1;
        this->xs[1] = x2;
        this->y = y;
    }
};

__global__ void calculate_sum(const MultivariateCoordinate* coord, double* out, Operation op) {
    int idx = threadIdx.x;
    double sum = 0;

    switch (op) {
    case SumOfX1:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += coord[i].xs[0];
        break;
    case SumOfX2:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += coord[i].xs[1];
        break;
    case SumOfY:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += coord[i].y;
        break;
    }

    __shared__ double r[BLOCK_SIZE];
    r[idx] = sum;
    __syncthreads();
    for (int size = BLOCK_SIZE / 2; size > 0; size /= 2) { //uniform
        if (idx < size)
            r[idx] += r[idx + size];
        __syncthreads();
    }
    if (idx == 0)
        *out = r[0];
}

__global__ void calculate_sum(const MultivariateCoordinate* coord, double* out, double mean, Operation op) {
    int idx = threadIdx.x;
    double sum = 0;

    switch (op) {
    case SumOfSquaresX1:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += (coord[i].xs[0] - mean) * (coord[i].xs[0] - mean);
        break;
    case SumOfSquaresX2:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += (coord[i].xs[1] - mean) * (coord[i].xs[1] - mean);
        break;
    }

    __shared__ double r[BLOCK_SIZE];
    r[idx] = sum;
    __syncthreads();
    for (int size = BLOCK_SIZE / 2; size > 0; size /= 2) { //uniform
        if (idx < size)
            r[idx] += r[idx + size];
        __syncthreads();
    }
    if (idx == 0)
        *out = r[0];
}

__global__ void calculate_sum(const MultivariateCoordinate* coord, double* out, double mean1, double mean2, Operation op) {
    int idx = threadIdx.x;
    double sum = 0;

    switch (op) {
    case SumOfProductsX1Y:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += (coord[i].xs[0] - mean1) * (coord[i].y - mean2);
        break;
    case SumOfProductsX2Y:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += (coord[i].xs[1] - mean1) * (coord[i].y - mean2);
        break;
    case SumofProductsX1X2:
        for (int i = idx; i < ARR_SIZE; i += BLOCK_SIZE)
            sum += (coord[i].xs[0] - mean1) * (coord[i].xs[1] - mean2);
        break;
    }

    __shared__ double r[BLOCK_SIZE];
    r[idx] = sum;
    __syncthreads();
    for (int size = BLOCK_SIZE / 2; size > 0; size /= 2) { //uniform
        if (idx < size)
            r[idx] += r[idx + size];
        __syncthreads();
    }
    if (idx == 0)
        *out = r[0];
}

std::vector<MultivariateCoordinate> read_mock_data(const char* filepath)
{
    std::vector<MultivariateCoordinate> data;

    io::CSVReader<5> in(filepath);
    in.read_header(io::ignore_extra_column, "name", "points", "skill", "assists", "salary");
    std::string name, skill;
    double points, assists, salary;

    // read all rows
    /*while (in.read_row(name, points, skill, assists, salary))
        data.push_back(Coordinate {points, salary});*/

        // read 10k rows
    for (int i = 0; i < ARR_SIZE; i++)
    {
        in.read_row(name, points, skill, assists, salary);
        data.push_back(MultivariateCoordinate(points, assists, salary));
    }

    std::cout << "data size: " << data.size() << std::endl;

    return data;
}

double SumWithCuda(const std::vector<MultivariateCoordinate> &mc, Operation op) {
    // Variables on CPU
    const MultivariateCoordinate* coord = &mc[0];
    double out[ARR_SIZE];
    int n = mc.size();

    // Variables on GPU
    MultivariateCoordinate* dev_coord;
    double* dev_out;
    cudaError_t cudaStatus;

    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Clean;
    }

    // Allocate memory on GPU
    cudaStatus = cudaMalloc((void**)&dev_coord, n * sizeof(MultivariateCoordinate));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    // Copy input data from CPU to GPU
    cudaStatus = cudaMemcpy(dev_coord, coord, n * sizeof(MultivariateCoordinate), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed!");
        goto Clean;
    }

    // Launch kernel
    calculate_sum << <1, BLOCK_SIZE >> > (dev_coord, dev_out, op);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Clean;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Clean;
    }

    // Copy output data from output to input
    cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed!");
        goto Clean;
    }

    return out[0];

Clean:
    cudaFree(dev_coord);
    cudaFree(dev_out);
}

double SumWithCuda(const std::vector<MultivariateCoordinate>& mc, double mean, Operation op) {
    // Variables on CPU
    const MultivariateCoordinate* coord = &mc[0];
    double out[ARR_SIZE];
    int n = mc.size();

    // Variables on GPU
    MultivariateCoordinate* dev_coord;
    double* dev_out;
    cudaError_t cudaStatus;

    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Clean;
    }

    // Allocate memory on GPU
    cudaStatus = cudaMalloc((void**)&dev_coord, n * sizeof(MultivariateCoordinate));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    // Copy input data from CPU to GPU
    cudaStatus = cudaMemcpy(dev_coord, coord, n * sizeof(MultivariateCoordinate), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed!");
        goto Clean;
    }

    // Launch kernel
    calculate_sum << <1, BLOCK_SIZE >> > (dev_coord, dev_out, mean, op);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Clean;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Clean;
    }

    // Copy output data from output to input
    cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed!");
        goto Clean;
    }

    return out[0];

Clean:
    cudaFree(dev_coord);
    cudaFree(dev_out);
}

double SumWithCuda(const std::vector<MultivariateCoordinate>& mc, double mean1, double mean2, Operation op) {
    // Variables on CPU
    const MultivariateCoordinate* coord = &mc[0];
    double out[ARR_SIZE];
    int n = mc.size();

    // Variables on GPU
    MultivariateCoordinate* dev_coord;
    double* dev_out;
    cudaError_t cudaStatus;

    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Clean;
    }

    // Allocate memory on GPU
    cudaStatus = cudaMalloc((void**)&dev_coord, n * sizeof(MultivariateCoordinate));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Clean;
    }

    // Copy input data from CPU to GPU
    cudaStatus = cudaMemcpy(dev_coord, coord, n * sizeof(MultivariateCoordinate), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed!");
        goto Clean;
    }

    // Launch kernel
    calculate_sum << <1, BLOCK_SIZE >> > (dev_coord, dev_out, mean1, mean2, op);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Clean;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Clean;
    }

    // Copy output data from output to input
    cudaStatus = cudaMemcpy(out, dev_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed!");
        goto Clean;
    }

    return out[0];

Clean:
    cudaFree(dev_coord);
    cudaFree(dev_out);
}

std::tuple<std::pair<double, double>, double> CalculateGradientAndYIntercept(const std::vector<MultivariateCoordinate>& mc) {
    auto timer = Timer("CalculateGradient");

    Operation op;

    // Total number of coordinates
    int n = mc.size();

    // Calculate sum of x1, sum of x2 and sum of y
    op = SumOfX1;
    double sum_of_x1 = SumWithCuda(mc, op);
    op = SumOfX2;
    double sum_of_x2 = SumWithCuda(mc, op);
    op = SumOfY;
    double sum_of_y = SumWithCuda(mc, op);

    // Calculate mean
    double mean_x1 = sum_of_x1 / n;
    double mean_x2 = sum_of_x2 / n;
    double mean_y = sum_of_y / n;

    // Calculate sum of squares
    op = SumOfSquaresX1;
    double sum_of_squares_x1 = SumWithCuda(mc, mean_x1, op);
    op = SumOfSquaresX2;
    double sum_of_squares_x2 = SumWithCuda(mc, mean_x2, op);

    // Calculate sum of products
    op = SumOfProductsX1Y;
    double sum_of_products_x1_y = SumWithCuda(mc, mean_x1, mean_y, op);
    op = SumOfProductsX2Y;
    double sum_of_products_x2_y = SumWithCuda(mc, mean_x2, mean_y, op);
    op = SumofProductsX1X2;
    double sum_of_products_x1_x2 = SumWithCuda(mc, mean_x1, mean_x2, op);
    
    // Calculate b1, b2 and a
    double b1 = (sum_of_products_x1_y * sum_of_squares_x2 - sum_of_products_x1_x2 * sum_of_products_x2_y) / (sum_of_squares_x1 * sum_of_squares_x2 - sum_of_products_x1_x2 * sum_of_products_x1_x2);

    double b2 = (sum_of_products_x2_y * sum_of_squares_x1 - sum_of_products_x1_x2 * sum_of_products_x1_y) / (sum_of_squares_x1 * sum_of_squares_x2 - sum_of_products_x1_x2 * sum_of_products_x1_x2);

    double a = mean_y - (b1 * mean_x1) - (b2 * mean_x2);

    // Test return
    return std::make_tuple(std::make_pair(b1, b2), a);
}

int main() {
    std::vector<MultivariateCoordinate> coordinates = read_mock_data("mock.csv");
    
    std::tuple<std::pair<double, double>, double> result = CalculateGradientAndYIntercept(coordinates);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}