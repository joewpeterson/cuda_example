#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKSIZE = 100
#define DERIVLEN = 4
#define N 1000

__global__ void first_deriv (double *out, double *in, double delta, int len)
{
        // With 8th order deriv, we will access each memory location multiple
        // time. This, then, is a good candidate for defining the block shared
        // memory before execution 

        __shared__ double blktmp[BLOCKSIZE + 2 * DERIVLEN];
        
        int g_index = blockIdx.x * blockDim.x + threadIdx.x; //global index
        int pb_index = threadIdx.x + DERIVLEN; // padded block index  

        // each thread copies memory into the shared space
        blktmp[pb_index] = in[g_index];

        //The threads on the edge copy into the padded zones
        if(threadIdx.x < DERIVLEN)
                if(g_index - DERIVLEN > 0) //make sure you are in g_index
                        blktmp[pb_index - DERIVLEN] = in[g_index - DERIVLEN];
                else
                        blktmp[pb_index - DERIVLEN] = 0;
        if(threadIdx.x < blockDim.x - DERIVLEN)
                if(g_index + DERIVLEN < len) //make sure you are in g_index
                        blktmp[pb_index + DERIVLEN] = in[g_index + DERIVLEN];
                else
                        blktmp[pb_index + DERIVLEN] = 0;

        // Before we compute the derivative, make sure all threads are synced
        __syncthreads();

        // Compute the 8th order FD derivative. Edges done with forward/backward
        // Otherwise, use central

        if (g_index < DERIVLEN) {
                out[g_index] = (-49./20.)*blktmp[pb_index] + (6.)*blktmp[pb_index + 1] +
                               (-15./2.)*blktmp[pb_index + 2] + (20./3.)*blktmp[pb_index + 3] +
                               (-15./4.)*blktmp[pb_index + 4] + (6./5.)*blktmp[pb_index + 5] +
                               (-1./6.)*blktmp[pb_index + 6];
        } else if (g_index > N - DERIVLEN) {
                out[g_index] = (49./20.)*blktmp[pb_index] + (-6.)*blktmp[pb_index - 1] +
                               (15./2.)*blktmp[pb_index - 2] + (-20./3.)*blktmp[pb_index - 3] +
                               (15./4.)*blktmp[pb_index - 4] + (-6./5.)*blktmp[pb_index - 5] +
                               (1./6.)*blktmp[pb_index - 6];
        } else {
                out[g_index] = (1./280.)*blktmp[pb_index - 4] + (-4./105.)*blktmp[pb_index - 3] +
                               (1./5.)*blktmp[pb_index - 2] + (-4./5.)*blktmp[pb_index - 1] +
                               (4./5.)*blktmp[pb_index + 1] + (-1./5.)*blktmp[pb_index + 2] +
                               (4./105.)*blktmp[pb_index + 3] + (-1./280.)*blktmp[pb_index + 4];

        }


