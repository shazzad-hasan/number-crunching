#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>


__global__ void kernel_function_a(const double *A, const double *x, double *y, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        double temp = 0.0;
        for (int j = 0; j < N; j++) {
            temp += A[idx * N + j] * x[idx]; 
        }
        y[idx] = temp;
    }
}

__global__ void kernel_function_b(const double a, const double *u, const double *v, double *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = a * u[idx] + v[idx];
    }
}

__global__ void kernel_function_c(const double s, const double *x, const double *y, double *z, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        if (idx % 2 == 0) {
            z[idx] = s * x[idx] + y[idx]; 
        } else {
            z[idx] = x[idx] + y[idx];   
        }
    }
}

__global__ void kernel_function_d(const double *u, const double *v, double *result, int N) {
    extern __shared__ double sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    double sum = 0;
    if (idx < N) {
        sum = u[idx] * v[idx];
    }
    sdata[tid] = sum;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

void init_datastructures(double *u, double *v, double *A, const int N) {
    for (unsigned int i = 0; i < N; i++) {
        u[i] = static_cast<double>(i % 2);
        v[i] = static_cast<double>(i % 4);
    }
    for (unsigned int i = 0; i < N * N; i++) {
        A[i] = static_cast<double>(i % 8);
    }
}

void print_results_to_file(const double s, const double *x, const double *y,
                           const double *z, const double *A, const long long n,
                           std::ofstream &File) {

    unsigned int N = std::min(n, static_cast<long long>(30));

    File << "N: "
        << "\n"
        << n << "\n";

    File << "s: "
        << std::fixed
        << std::setprecision(1)
        << "\n"
        << s << "\n";

    File << "x: "
        << "\n";
    for (unsigned int i = 0; i < N; i++) {
        File << x[i] << " ";
    }
    File << "\n";

    File << "y: "
        << "\n";
    for (unsigned int i = 0; i < N; i++) {
        File << y[i] << " ";
    }
    File << "\n";

    File << "z: "
        << "\n";
    for (unsigned int i = 0; i < N; i++) {
        File << z[i] << " ";
    }
    File << "\n";
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Error: Missing problem size N. Please provide N as a command-line parameter. "
                    "Usage example for N=10: ./number_crunching_task 10" << std::endl;
        return EXIT_FAILURE;
    }

    const int N = std::stoi(argv[1]);

    const int blockSize = 256;

    double *u = new double[N];
    double *v = new double[N];
    double *A = new double[N * N];
    double *x = new double[N];
    double *y = new double[N];
    double *z = new double[N];
    double *d_u, *d_v, *d_A, *d_x, *d_y, *d_z, *d_result;
    double s = 0.0;

    cudaMalloc(&d_u, N * sizeof(double));
    cudaMalloc(&d_v, N * sizeof(double));
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_z, N * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    // Initialize data on the host
    init_datastructures(u, v, A, N);

    // Copy data to the device
    cudaMemcpy(d_u, u, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &s, sizeof(double), cudaMemcpyHostToDevice); 

    // Prepare streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Launch kernels in their respective streams
    kernel_function_b<<<(N + blockSize - 1) / blockSize, blockSize, 0, stream1>>>(2.0, d_u, d_v, d_x, N);
    kernel_function_a<<<(N + blockSize - 1) / blockSize, blockSize, 0, stream2>>>(d_A, d_x, d_y, N);
    int numBlocks = min((N + blockSize - 1) / blockSize, 1024);
    kernel_function_d<<<numBlocks, blockSize, blockSize * sizeof(double), stream3>>>(d_u, d_v, d_result, N);
    cudaStreamSynchronize(stream3);  

    cudaMemcpyAsync(&s, d_result, sizeof(double), cudaMemcpyDeviceToHost, stream3);
    cudaStreamSynchronize(stream3);  

    kernel_function_c<<<(N + blockSize - 1) / blockSize, blockSize, 0, stream4>>>(s, d_x, d_y, d_z, N);
    cudaStreamSynchronize(stream4); 

    // Copy results back to host
    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, N * sizeof(double), cudaMemcpyDeviceToHost);

    std::ofstream File("partial_results.out");
    print_results_to_file(s, x, y, z, A, N, File);

    std::cout << "For correctness checking, partial results have been written to "
                "partial_results.out"
                << std::endl;


    delete[] u; 
    delete[] v; 
    delete[] A; 
    delete[] x; 
    delete[] y; 
    delete[] z;
    cudaFree(d_u); 
    cudaFree(d_v); 
    cudaFree(d_A); 
    cudaFree(d_x); 
    cudaFree(d_y); 
    cudaFree(d_z); 
    cudaFree(d_result);
    cudaStreamDestroy(stream1); 
    cudaStreamDestroy(stream2); 
    cudaStreamDestroy(stream3); 
    cudaStreamDestroy(stream4);

    return EXIT_SUCCESS;
}
