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

#define DEFAULT_ARRAY_LEN 8000 //10
#define DEFAULT_NB_BINS 6000
#define DEFAULT_LOWER_BOUND 0.0
#define DEFAULT_UPPER_BOUND 10.0
#define DEFAULT_NB_REPEAT 10

#define MAX_DISPLAY_COLUMNS 10
#define MAX_DISPLAY_ROWS 20

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

#define PRINT_ERROR(MSG)                                                    \
        do                                                                  \
        {                                                                   \
                fprintf(stderr, "%s:%d - %s\n", __FILE__, __LINE__, (MSG)); \
                exit(EXIT_FAILURE);                                         \
        } while (0)

#define IO_CHECK(OP, RET)                   \
        do                                  \
        {                                   \
                if ((RET) < 0)              \
                {                           \
                        perror((OP));       \
                        exit(EXIT_FAILURE); \
                }                           \
        } while (0)

static void usage(void)
{
        fprintf(stderr, "usage: histogram [OPTIONS...]\n");
        fprintf(stderr, "    --array-len  ARRAY_LENGTH\n");
        fprintf(stderr, "    --nb-bins  NB_BINS\n");
        fprintf(stderr, "    --lower-bound  LOWER_BOUND\n");
        fprintf(stderr, "    --higher-bound  HIGHER_BOUND\n");
        fprintf(stderr, "    --nb-repeat NB_REPEAT\n");
        fprintf(stderr, "    --output\n");
        fprintf(stderr, "    --verbose\n");
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
}
static void init_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings == NULL);

        struct s_settings *p_settings = (struct s_settings *)calloc(1, sizeof(struct s_settings));        
        if (p_settings == NULL)
        {
                PRINT_ERROR("memory allocation failed");
                exit(EXIT_FAILURE);
        }
        p_settings->array_len = DEFAULT_ARRAY_LEN;
        p_settings->nb_bins = DEFAULT_NB_BINS;
        p_settings->lower_bound = DEFAULT_LOWER_BOUND;
        p_settings->upper_bound = DEFAULT_UPPER_BOUND;
        p_settings->nb_repeat = DEFAULT_NB_REPEAT;
        p_settings->enable_verbose = 0;
        p_settings->enable_output = 0;
        *pp_settings = p_settings;
}

static void parse_cmd_line(int argc, char *argv[], struct s_settings *p_settings)
{
        int i = 1;
        while (i < argc)
        {
                if (strcmp(argv[i], "--array-len") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid ARRAY_LENGTH argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->array_len = value;
                }
                else if (strcmp(argv[i], "--nb-bins") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid NB_BINS argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->nb_bins = value;
                }
                else if (strcmp(argv[i], "--lower-bound") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atof(argv[i]);
                        int class_value =fpclassify(value);
                        if ((class_value != FP_NORMAL) && (class_value != FP_ZERO))
                        {
                                fprintf(stderr, "invalid LOWER_BOUND argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->lower_bound = value;
                }
                else if (strcmp(argv[i], "--upper-bound") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        double value = atof(argv[i]);
                        int class_value = fpclassify(value);
                        if ((class_value != FP_NORMAL) && (class_value != FP_ZERO))
                        {
                                fprintf(stderr, "invalid UPPER_BOUND argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->upper_bound = value;
                }
                else if (strcmp(argv[i], "--nb-repeat") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid NB_REPEAT argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->nb_repeat = value;
                }
                else if (strcmp(argv[i], "--output") == 0)
                {
                        p_settings->enable_output = 1;
                }
                else if (strcmp(argv[i], "--verbose") == 0)
                {
                        p_settings->enable_verbose = 1;
                }
                else
                {
                        usage();
                }

                i++;
        }

        if (p_settings->upper_bound <= p_settings->lower_bound)
        {
                fprintf(stderr, "invalid histogram bounds\n");
                exit(EXIT_FAILURE);
        }

        if (p_settings->enable_output)
        {
                p_settings->nb_repeat = 1;
        }
}

static void delete_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings != NULL);
        free(*pp_settings);
        pp_settings = NULL;
}

static void allocate_array(ELEMENT_TYPE **p_array, struct s_settings *p_settings)
{
        assert(*p_array == NULL);
        ELEMENT_TYPE *array = (float *)calloc(p_settings->array_len, sizeof(*array));
        if (array == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        *p_array = array;
}

static void delete_array(ELEMENT_TYPE **p_array)
{
        assert(*p_array != NULL);
        free(*p_array);
        p_array = NULL;
}

static void init_array_random(ELEMENT_TYPE *array, struct s_settings *p_settings)
{
        const ELEMENT_TYPE offset = p_settings->lower_bound;
        const ELEMENT_TYPE scale = p_settings->upper_bound - p_settings->lower_bound;

        int i;
        for (i = 0; i < p_settings->array_len; i++)
        {
                ELEMENT_TYPE value = scale * ((ELEMENT_TYPE)rand()) / (1.0 + (ELEMENT_TYPE)(RAND_MAX)) + offset;
                array[i] = value;
        }
}

static void print_array(const ELEMENT_TYPE *array, struct s_settings *p_settings)
{
        printf("[");
        int j = 0;
        int i;
        for (i = 0; i < p_settings->array_len; i++)
        {
                if (i > 0)
                {
                        printf(",");
                        if ((i % MAX_DISPLAY_COLUMNS == 0))
                        {
                                printf("\n");
                                printf(" ");
                                j++;

                                if (j >= MAX_DISPLAY_ROWS)
                                {
                                        printf("  ...\n");
                                        break;
                                }
                        }
                }
                printf(" %8.3lg", array[i]);
        }
        printf(" ]");
}

static void write_array_to_file(FILE *file, const ELEMENT_TYPE *array, struct s_settings *p_settings)
{
        int i;
        int ret;

        for (i = 0; i < p_settings->array_len; i++)
        {
                ret = fprintf(file, "%lf\n", array[i]);
                IO_CHECK("fprintf", ret);
        }
}

static void allocate_histogram(int **p_histogram, struct s_settings *p_settings)
{
        assert(*p_histogram == NULL);
        int *histogram =(int *)calloc(p_settings->nb_bins, sizeof(*histogram));
        if (histogram == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        *p_histogram = histogram;
}

static void delete_histogram(int **p_histogram)
{
        assert(*p_histogram != NULL);
        free(*p_histogram);
        p_histogram = NULL;
}

static void print_histogram(const int *histogram, struct s_settings *p_settings)
{
        const ELEMENT_TYPE offset = p_settings->lower_bound;
        const ELEMENT_TYPE scale = p_settings->upper_bound - p_settings->lower_bound;

        printf("<\n");
        int i;
        for (i = 0; i < p_settings->nb_bins; i++)
        {
                ELEMENT_TYPE lower = offset + i * scale / p_settings->nb_bins;
                ELEMENT_TYPE upper = offset + (i + 1) * scale / p_settings->nb_bins;

                printf(" [ %8.2lg ... %8.2lg [ :  %d\n", lower, upper, histogram[i]);
        }
        printf(">");
}

static void write_bins_to_file(FILE *file, struct s_settings *p_settings)
{
        int i;
        int ret;

        const ELEMENT_TYPE offset = p_settings->lower_bound;
        const ELEMENT_TYPE scale = p_settings->upper_bound - p_settings->lower_bound;

        ret = fprintf(file, "%lf\n", offset);
        IO_CHECK("fprintf", ret);
        for (i = 0; i < p_settings->nb_bins; i++)
        {
                ELEMENT_TYPE bound = offset + (i + 1) * scale / p_settings->nb_bins;
                ret = fprintf(file, "%lf\n", bound);
                IO_CHECK("fprintf", ret);
        }
}

static void write_histogram_to_file(FILE *file, const int *histogram, struct s_settings *p_settings)
{
        int i;
        int ret;

        for (i = 0; i < p_settings->nb_bins; i++)
        {
                ret = fprintf(file, "%d\n", histogram[i]);
                IO_CHECK("fprintf", ret);
        }
}

static void print_settings_csv_header(void)
{
        printf("array_len,nb_bins,nb_repeat");
}

static void print_settings_csv(struct s_settings *p_settings)
{
        printf("%d,%d,%d", p_settings->array_len, p_settings->nb_bins, p_settings->nb_repeat);
}

static void print_results_csv_header(void)
{
        printf("rep,timing,check_status");
}

static void print_results_csv(int rep, double timing_in_seconds, int check_status)
{
        printf("%d,%le,%d", rep, timing_in_seconds, check_status);
}

static void print_csv_header(void)
{
        print_settings_csv_header();
        printf(",");
        print_results_csv_header();
        printf("\n");
}

static void naive_compute_histogram(const ELEMENT_TYPE *array, int *histogram, struct s_settings *p_settings)
{
        //Intialiser l'histogramme à zéro
        memset(histogram, 0, p_settings->nb_bins * sizeof(*histogram));

        ELEMENT_TYPE *bounds = NULL;
        bounds = (float *)malloc((p_settings->nb_bins + 1) * sizeof(*bounds));
        if (bounds == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }

        {
                const ELEMENT_TYPE offset = p_settings->lower_bound;
                const ELEMENT_TYPE scale = p_settings->upper_bound - p_settings->lower_bound;

                bounds[0] = offset;

                //On peut paralléliser avec Openmp
                int j;
                //#pragma omp parallel for
                for (j = 0; j < p_settings->nb_bins; j++)
                {
                        bounds[j + 1] = offset + (j + 1) * scale / p_settings->nb_bins;
                }
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

        free(bounds);
}

//Appelée depuis le CPU mais executée sur le GPU__Déclaration du kernel Cuda
__global__ void compute_histogram_kernel(const ELEMENT_TYPE *array, int *histogram, const ELEMENT_TYPE *bounds, int array_len, int nb_bins) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < array_len) {

        ELEMENT_TYPE value = array[i];

        for (int j = 0; j < nb_bins; j++) {

            if (value >= bounds[j] && value < bounds[j + 1]) {

                atomicAdd(&histogram[j], 1);

                break;
            }
        }
    }
}

static void cuda_compute_histogram(const ELEMENT_TYPE *array, int *histogram, struct s_settings *p_settings) 
{
    //Intialiser l'histogramme à zéro
        memset(histogram, 0, p_settings->nb_bins * sizeof(*histogram));

        ELEMENT_TYPE *bounds = NULL;
        bounds = (float *)malloc((p_settings->nb_bins + 1) * sizeof(*bounds));
        if (bounds == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }

        {
                const ELEMENT_TYPE offset = p_settings->lower_bound;
                const ELEMENT_TYPE scale = p_settings->upper_bound - p_settings->lower_bound;

                bounds[0] = offset;

                //On peut paralléliser avec Openmp
                int j;
                //#pragma omp parallel for
                for (j = 0; j < p_settings->nb_bins; j++)
                {
                        bounds[j + 1] = offset + (j + 1) * scale / p_settings->nb_bins;
                }
        }

        ELEMENT_TYPE *gpu_array, *gpu_bounds;
        int *gpu_histogram;

        //Allouer la mémoire sur le GPU
        cudaMalloc((void **)&gpu_array, p_settings ->array_len * sizeof(ELEMENT_TYPE));
        cudaMalloc((void**)&gpu_bounds, (p_settings->nb_bins + 1) * sizeof(ELEMENT_TYPE));
        cudaMalloc((void**)&gpu_histogram, p_settings->nb_bins * sizeof(int));

        //Copier les données depuis le CPU vers le GPU
        cudaMemcpy(gpu_array, array, p_settings->array_len * sizeof(ELEMENT_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_bounds, bounds, (p_settings->nb_bins + 1) * sizeof(ELEMENT_TYPE), cudaMemcpyHostToDevice);

        int num_blocks = (p_settings->array_len + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

        //Appeler le kernel cuda
        compute_histogram_kernel<<<num_blocks, THREAD_PER_BLOCK>>>(gpu_array, gpu_histogram, gpu_bounds, p_settings->array_len, p_settings->nb_bins);

        //Copierles résultats depuis le GPU vers le CPU
        cudaMemcpy(histogram, gpu_histogram, p_settings->nb_bins * sizeof(int), cudaMemcpyDeviceToHost);

        //Libérer la mémoire allouée sur le GPU
        cudaFree(gpu_array);
        cudaFree(gpu_bounds);
        cudaFree(gpu_histogram);

        // Libérer la mémoire allouée sur le CPU
        free(bounds);

}


static void run(const ELEMENT_TYPE *array, int *run_histogram, struct s_settings *p_settings)
{
        //mp_compute_histogram(array, run_histogram, p_settings);
        //naive_compute_histogram(array, run_histogram, p_settings);
        cuda_compute_histogram(array, run_histogram, p_settings);
        if (p_settings->enable_output)
        {
                FILE *file = fopen("run_histogram.csv", "w");
                if (file == NULL)
                {
                        perror("fopen");
                        exit(EXIT_FAILURE);
                }
                write_histogram_to_file(file, run_histogram, p_settings);
                fclose(file);
        }

        if (p_settings->enable_verbose)
        {
                printf("run histogram:\n");
                print_histogram(run_histogram, p_settings);
                printf("\n\n");
        }
}

static int check(const ELEMENT_TYPE *array, int *check_histogram, const int *run_histogram, struct s_settings *p_settings)
{
        naive_compute_histogram(array, check_histogram, p_settings);

        if (p_settings->enable_output)
        {
                FILE *file = fopen("check_histogram.csv", "w");
                if (file == NULL)
                {
                        perror("fopen");
                        exit(EXIT_FAILURE);
                }
                write_histogram_to_file(file, check_histogram, p_settings);
                fclose(file);
        }

        if (p_settings->enable_verbose)
        {
                printf("check histogram:\n");
                print_histogram(check_histogram, p_settings);
                printf("\n\n");
        }

        int check = 0;
        int i;
        for (i = 0; i < p_settings->nb_bins; i++)
        {
                if (run_histogram[i] != check_histogram[i])
                {
                        fprintf(stderr, "check failed [bin: %d]: run = %d, check = %d\n", i,
                                run_histogram[i], check_histogram[i]);
                        check = 1;
                }
        }
        return check;
}

int main(int argc, char *argv[])
{
        struct s_settings *p_settings = NULL;

        init_settings(&p_settings);
        parse_cmd_line(argc, argv, p_settings);

        ELEMENT_TYPE *array = NULL;
        allocate_array(&array, p_settings);

        int *histogram = NULL;
        allocate_histogram(&histogram, p_settings);

        int *check_histogram = NULL;
        allocate_histogram(&check_histogram, p_settings);

        {
                if (!p_settings->enable_verbose)
                {
                        print_csv_header();
                }

                if (p_settings->enable_output)
                {
                        FILE *file = fopen("bins.csv", "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_bins_to_file(file, p_settings);
                        fclose(file);
                }

                int rep;
                for (rep = 0; rep < p_settings->nb_repeat; rep++)
                {
                        if (p_settings->enable_verbose)
                        {
                                printf("repeat %d\n", rep);
                        }

                        init_array_random(array, p_settings);

                        if (p_settings->enable_output)
                        {
                                FILE *file = fopen("array.csv", "w");
                                if (file == NULL)
                                {
                                        perror("fopen");
                                        exit(EXIT_FAILURE);
                                }
                                write_array_to_file(file, array, p_settings);
                                fclose(file);
                        }

                        if (p_settings->enable_verbose)
                        {
                                printf("array:\n");
                                print_array(array, p_settings);
                                printf("\n\n");
                        }

                        struct timespec timing_start, timing_end;
                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                        run(array, histogram, p_settings);
                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                        double timing_in_seconds = (timing_end.tv_sec - timing_start.tv_sec) + 1.0e-9 * (timing_end.tv_nsec - timing_start.tv_nsec);

                        int check_status = check(array, check_histogram, histogram, p_settings);

                        if (p_settings->enable_verbose)
                        {
                                print_csv_header();
                        }
                        print_settings_csv(p_settings);
                        printf(",");
                        print_results_csv(rep, timing_in_seconds, check_status);
                        printf("\n");
                }
        }

        delete_histogram(&check_histogram);
        delete_histogram(&histogram);

        delete_array(&array);
        delete_settings(&p_settings);

        return 0;
}
