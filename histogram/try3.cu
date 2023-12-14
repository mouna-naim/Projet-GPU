#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
//#include <helper_cuda.h>
#define ELEMENT_TYPE float  // Assurez-vous que cela correspond à la définition dans votre programme principal

// Kernel défini dans votre code
__global__ void compute_histogram_kernel(ELEMENT_TYPE *bounds, const int nb_bins, const ELEMENT_TYPE lower_bound, const ELEMENT_TYPE upper_bound) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    ELEMENT_TYPE scale = (upper_bound - lower_bound) / nb_bins;

    if (j < nb_bins) {
        bounds[j] = lower_bound + j * scale;
    }
}

// Fonction pour tester le kernel
void test_compute_histogram_kernel() {
    int nb_bins = 6;  // Exemple: nombre de bins
    ELEMENT_TYPE lower_bound = 0.0;
    ELEMENT_TYPE upper_bound = 10.0;

    // Allocation de la mémoire sur le CPU
    ELEMENT_TYPE *bounds = (ELEMENT_TYPE *)malloc(nb_bins * sizeof(ELEMENT_TYPE));

    // Allocation de la mémoire sur le GPU
    ELEMENT_TYPE *gpu_bounds;
    cudaMalloc(&gpu_bounds, nb_bins * sizeof(ELEMENT_TYPE));
    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
       fprintf(stderr, "Failed to allocate memory - %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    // Calcul du nombre de blocs et de threads (1 thread par bin)
    int threadsPerBlock = 256;
    int blocksPerGrid = (nb_bins + threadsPerBlock - 1) / threadsPerBlock;

    // Appel du kernel
    compute_histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(gpu_bounds, nb_bins, lower_bound, upper_bound);

    // Attendez que le GPU finisse
    cudaDeviceSynchronize();

    // Copie des résultats du GPU vers le CPU
    cudaMemcpy(bounds, gpu_bounds, nb_bins * sizeof(ELEMENT_TYPE), cudaMemcpyDeviceToHost);

    // Affichage des bornes calculées
    for (int i = 0; i < nb_bins; i++) {
        printf("Bin %d: %f\n", i, bounds[i]);
    }

    // Libération de la mémoire
    cudaFree(gpu_bounds);
    free(bounds);
}

int main() {
    test_compute_histogram_kernel();
    return 0;
}

//salloc --nodes --time=01:00:00
