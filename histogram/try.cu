#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define ELEMENT_TYPE float
#define THREAD_PER_BLOCK 265 

#define DEFAULT_ARRAY_LEN 8 //10
#define DEFAULT_NB_BINS 6
#define DEFAULT_LOWER_BOUND 0.0
#define DEFAULT_UPPER_BOUND 10.0
#define DEFAULT_NB_REPEAT 10

#define MAX_DISPLAY_COLUMNS 10
#define MAX_DISPLAY_ROWS 20

#define PRINT_ERROR(MSG)                                                    \
        do                                                                  \
        {                                                                   \
                fprintf(stderr, "%s:%d - %s\n", __FILE__, __LINE__, (MSG)); \
                exit(EXIT_FAILURE);                                         \
        } while (0)


struct s_settings
{
        int array_len;
        int nb_bins;
        double lower_bound;
        double upper_bound;
        int nb_repeat;
        int enable_output;
        int enable_verbose;
};

__global__ void compute_histogram_kernel(ELEMENT_TYPE *bounds, int nb_bins, ELEMENT_TYPE lower_bound, ELEMENT_TYPE upper_bound) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j <= nb_bins) {
        ELEMENT_TYPE scale = (upper_bound - lower_bound) / nb_bins;
        bounds[j] = lower_bound + j * scale;
    }
}


static void cuda_compute_histogram(const ELEMENT_TYPE *array, int *histogram, struct s_settings *p_settings) {
    memset(histogram, 0, p_settings->nb_bins * sizeof(*histogram));
    ELEMENT_TYPE *bounds = (ELEMENT_TYPE *)malloc((p_settings->nb_bins + 1) * sizeof(ELEMENT_TYPE));

    ELEMENT_TYPE *gpu_bounds;
    cudaMalloc(&gpu_bounds, (p_settings->nb_bins + 1) * sizeof(ELEMENT_TYPE));
    compute_histogram_kernel<<<(p_settings->nb_bins + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(gpu_bounds, p_settings->nb_bins, p_settings->lower_bound, p_settings->upper_bound);
    cudaDeviceSynchronize();
    cudaMemcpy(bounds, gpu_bounds, (p_settings->nb_bins + 1) * sizeof(ELEMENT_TYPE), cudaMemcpyDeviceToHost);
        
        
         for (int i = 0; i < DEFAULT_NB_BINS; i++) {
            printf("Le bound est  %f\n", bounds[i]);
        }


        int i;
        for (i = 0; i < p_settings->array_len; i++)
        {
                ELEMENT_TYPE value = array[i];

                int j;
                for (j = 0; j < p_settings->nb_bins; j++)
                {
                        if (value >= bounds[j] && value < bounds[j + 1])
                        {
                                histogram[j]++;
                                break;
                        }
                }
        }


        // Libérer la mémoire allouée sur le CPU
        cudaFree(gpu_bounds);
        free(bounds);

}

int main() {
    ELEMENT_TYPE test_array[] = {1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 9.0};
    int histogram[DEFAULT_NB_BINS] = {0};
    struct s_settings settings = {6, DEFAULT_NB_BINS, DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND, DEFAULT_NB_REPEAT, 1, 1};

    // Appeler la fonction CUDA
    cuda_compute_histogram(test_array, histogram, &settings);

    // Afficher les résultats
    printf("Histogramme calculé :\n");
    for (int i = 0; i < DEFAULT_NB_BINS; i++) {
        printf("Bin %d : %d\n", i, histogram[i]);
    }

    // Afficher l'histogramme attendu
    printf("\nHistogramme attendu :\n");
    for (int i = 0; i < DEFAULT_NB_BINS; i++) {
        printf("Bin %d : %d\n", i, 1); // Comme on attend 1 dans chaque bin
    }

    return 0;
}
