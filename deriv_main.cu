#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKSIZE = 1000
#define DERIVLEN = 4
#define N 1000

// Initialize the kernel
__global__ void first_deriv (double *out, double *in, double delta, int len);

// Inline check for CUDA error
static inline void is_cuda_success (int i)
{
        if (i != cudaSuccess)
                printf("CUDA error %d\n" i);
}

int main()
{
        size_t mem_size = N * sizeof(double);

        // Initialize pointers for host
        double *in, *out;
        in = (double *) malloc(mem_size);
        out = (double *) malloc(mem_size);

        // Input function is x^2 with 0.01 inc between points
        for (int i = 0; i < N; i++)
                a[i] = ((double) i)/100 * ((double) i)/100.0;

        double delta = 0.01;

        // Initialize pointers for device and copy input
        int = err;
        double *d_in, *d_out; 

        cudaMalloc((void **) &d_in, size);
        cudaMalloc((void **) &d_out, size);
        err = cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
        is_cuda_success(err);

        // With device initialized, run the kernel
        first_deriv<<<N/BLOCKSIZE, BLOCKSIZE>>>(d_out, d_in, delta, N);

        // Finally, copy the output from device to host
        err = cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
        is_cuda_success(err);

        // clean up device
        cudaFree(d_in);
        cudaFree(d_out);

        // clean up host
        free(in);
        free(out);

        return 0;
}
