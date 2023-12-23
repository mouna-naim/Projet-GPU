#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <starpu.h>
#include <immintrin.h>
// Nombre de parts si on le mets pas en arguments
#define DEFAULT_NB_PARTS 12
#define DEFAULT_VERIF 1
#define ELEMENT_TYPE float
#define DEFAULT_MESH_WIDTH 2000
#define DEFAULT_MESH_HEIGHT 1000
#define DEFAULT_NB_ITERATIONS 10
#define DEFAULT_NB_REPEAT 1

#define STENCIL_WIDTH 3
#define STENCIL_HEIGHT 3

#define TOP_BOUNDARY_VALUE 10
#define BOTTOM_BOUNDARY_VALUE 5
#define LEFT_BOUNDARY_VALUE -10
#define RIGHT_BOUNDARY_VALUE -5

#define MAX_DISPLAY_COLUMNS 20
#define MAX_DISPLAY_LINES 100

#define EPSILON 1e-3
//taille des bits dans le SIMD register
#define REG_BYTES (sizeof(__m256))
// nombre d'élements dans un SIMD register
#define REG_NB_ELEMENTS (REG_BYTES / sizeof(float))
static const ELEMENT_TYPE stencil_coefs[STENCIL_HEIGHT * STENCIL_WIDTH] =
    {
        0,     0.25, 0   ,
        0.25, -1.00, 0.25,
        0,     0.25, 0
    };

enum e_initial_mesh_type
{
        initial_mesh_zero = 1,
        initial_mesh_random = 2
};
enum v_version
{
	naive = 1,
	starpu = 2,
        nvx=3,
        starpu_nvx=4
};
struct s_settings
{
        int mesh_width;
        int mesh_height;
        enum e_initial_mesh_type initial_mesh_type;
        //_____________

        enum v_version vcode;
        int parts;
        int verif;

        //______________
        int nb_iterations;
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
        fprintf(stderr, "usage: stencil [OPTIONS...]\n");
        fprintf(stderr, "    --mesh-width  MESH_WIDTH\n");
        fprintf(stderr, "    --mesh-height MESH_HEIGHT\n");
        fprintf(stderr, "    --initial-mesh <zero|random>\n");
        fprintf(stderr, "    --nb-iterations NB_ITERATIONS\n");
        fprintf(stderr, "    --nb-repeat NB_REPEAT\n");
        fprintf(stderr, "    --output\n");
        fprintf(stderr, "    --verbose\n");
        //La verification permet de valider les données avec la fonction naive mettez non pour calculer la vitesse
        fprintf(stderr, "    --verification <1|0>\n");
        fprintf(stderr, "    --taskpart NB_PARTS\n");
        fprintf(stderr, "    --version <naive|starpu|nvx|starpu_nvx>\n");
        //_____________________
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
}

static void init_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings == NULL);
        struct s_settings *p_settings = calloc(1, sizeof(*p_settings));
        if (p_settings == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        p_settings->mesh_width = DEFAULT_MESH_WIDTH;
        p_settings->mesh_height = DEFAULT_MESH_HEIGHT;
        p_settings->initial_mesh_type = initial_mesh_zero;
        p_settings->nb_iterations = DEFAULT_NB_ITERATIONS;
        p_settings->nb_repeat = DEFAULT_NB_REPEAT;
        p_settings->enable_verbose = 0;
        p_settings->enable_output = 0;
        //___________________
        p_settings->parts = DEFAULT_NB_PARTS;
        p_settings->verif =DEFAULT_VERIF;
        p_settings->vcode = starpu;
        //______________________
        *pp_settings = p_settings;
}

static void parse_cmd_line(int argc, char *argv[], struct s_settings *p_settings)
{
        int i = 1;
        while (i < argc)
        {
                if (strcmp(argv[i], "--mesh-width") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < STENCIL_WIDTH)
                        {
                                fprintf(stderr, "invalid MESH_WIDTH argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->mesh_width = value;
                }
                else if (strcmp(argv[i], "--mesh-height") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < STENCIL_HEIGHT)
                        {
                                fprintf(stderr, "invalid MESH_HEIGHT argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->mesh_height = value;
                }


                //______________ mes modifs_____________________
                else if (strcmp(argv[i], "--taskpart") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "Nombre de parts invalide\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->parts = value;
                }

                else if (strcmp(argv[i], "--verification") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if ((value != 1)   && (value != 0)  )
                        {
                                fprintf(stderr, "Verification invalide mettez 1 pour verifier 0 sinon\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->verif = value;
                }
                else if (strcmp(argv[i], "--version") == 0)
                {
                        i++;
                        if (strcmp(argv[i], "starpu") == 0)
			{
				p_settings->vcode = starpu;
			}
			else if (strcmp(argv[i], "naive") == 0)
			{
				p_settings->vcode = naive;
			}
                        else if (strcmp(argv[i], "nvx") == 0)
			{
				p_settings->vcode = nvx;
			}
                        else if (strcmp(argv[i], "starpu_nvx") == 0)
			{
				p_settings->vcode = starpu_nvx;
			}
			else
			{
				fprintf(stderr, "La version est invalide\n");
				exit(EXIT_FAILURE);
			}
                }







                else if (strcmp(argv[i], "--initial-mesh") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        if (strcmp(argv[i], "zero") == 0)
                        {
                                p_settings->initial_mesh_type = initial_mesh_zero;
                        }
                        else if (strcmp(argv[i], "random") == 0)
                        {
                                p_settings->initial_mesh_type = initial_mesh_random;
                        }
                        else
                        {
                                fprintf(stderr, "invalid initial mesh type\n");
                                exit(EXIT_FAILURE);
                        }
                }
                else if (strcmp(argv[i], "--nb-iterations") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid NB_ITERATIONS argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->nb_iterations = value;
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

        if (p_settings->enable_output)
        {
                p_settings->nb_repeat = 1;
                if (p_settings->nb_iterations > 100)
                {
                        p_settings->nb_iterations = 100;
                }
        }
}

static void delete_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings != NULL);
        free(*pp_settings);
        pp_settings = NULL;
}

static void allocate_mesh(ELEMENT_TYPE **pp_mesh, struct s_settings *p_settings)
{
        assert(*pp_mesh == NULL);
        ELEMENT_TYPE *p_mesh = calloc(p_settings->mesh_width * p_settings->mesh_height, sizeof(*p_mesh));
        if (p_mesh == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        *pp_mesh = p_mesh;
}

static void delete_mesh(ELEMENT_TYPE **pp_mesh)
{
        assert(*pp_mesh != NULL);
        free(*pp_mesh);
        pp_mesh = NULL;
}

static void init_mesh_zero(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;
        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = 0;
                }
        }
}

static void init_mesh_random(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;
        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
                {
                        ELEMENT_TYPE value = rand() / (ELEMENT_TYPE)RAND_MAX * 20 - 10;
                        p_mesh[y * p_settings->mesh_width + x] = value;
                }
        }
}
static void init_mesh_values(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        switch (p_settings->initial_mesh_type)
        {
        case initial_mesh_zero:
                init_mesh_zero(p_mesh, p_settings);
                break;

        case initial_mesh_random:
                init_mesh_random(p_mesh, p_settings);
                break;

        default:
                PRINT_ERROR("invalid initial mesh type");
        }
}

static void copy_mesh(ELEMENT_TYPE *p_dst_mesh, const ELEMENT_TYPE *p_src_mesh, struct s_settings *p_settings)
{
        memcpy(p_dst_mesh, p_src_mesh, p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_dst_mesh));
}

static void apply_boundary_conditions(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;

        for (x = 0; x < p_settings->mesh_width; x++)
        {
                for (y = 0; y < margin_y; y++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = TOP_BOUNDARY_VALUE;
                        p_mesh[(p_settings->mesh_height - 1 - y) * p_settings->mesh_width + x] = BOTTOM_BOUNDARY_VALUE;
                }
        }

        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = 0; x < margin_x; x++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = LEFT_BOUNDARY_VALUE;
                        p_mesh[y * p_settings->mesh_width + (p_settings->mesh_width - 1 - x)] = RIGHT_BOUNDARY_VALUE;
                }
        }
}

static void print_settings_csv_header(void)
{
        printf("mesh_width,mesh_height,nb_iterations,nb_repeat");
}

static void print_settings_csv(struct s_settings *p_settings)
{
        printf("%d,%d,%d,%d", p_settings->mesh_width, p_settings->mesh_height, p_settings->nb_iterations, p_settings->nb_repeat);
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

static void print_mesh(const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int x;
        int y;

        printf("[\n");
        for (y = 0; y < p_settings->mesh_height; y++)
        {
                if (y >= MAX_DISPLAY_LINES)
                {
                        printf("...\n");
                        break;
                }
                printf("[%03d: ", y);
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        if (x >= MAX_DISPLAY_COLUMNS)
                        {
                                printf("...");
                                break;
                        }
                        printf(" %+8.2lf", p_mesh[y * p_settings->mesh_width + x]);
                }
                printf("]\n");
        }
        printf("]");
}

static void write_mesh_to_file(FILE *file, const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int x;
        int y;
        int ret;

        for (y = 0; y < p_settings->mesh_height; y++)
        {
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        if (x > 0)
                        {
                                ret = fprintf(file, ",");
                                IO_CHECK("fprintf", ret);
                        }

                        ret = fprintf(file, "%lf", p_mesh[y * p_settings->mesh_width + x]);
                        IO_CHECK("fprintf", ret);
                }

                ret = fprintf(file, "\n");
                IO_CHECK("fprintf", ret);
        }
}



static void naive_stencil_func(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;

        ELEMENT_TYPE *p_temporary_mesh = malloc(p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_mesh));
        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
                for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
                {
                        ELEMENT_TYPE value = p_mesh[y * p_settings->mesh_width + x];
                        int stencil_x, stencil_y;
                        for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
                        {
                                for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
                                {
                                        value +=
                                            p_mesh[(y + stencil_y - margin_y) * p_settings->mesh_width + (x + stencil_x - margin_x)] * stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                                }
                        }
                        p_temporary_mesh[y * p_settings->mesh_width + x] = value;
                }
        }

        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
                for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = p_temporary_mesh[y * p_settings->mesh_width + x];
                }
        }
        free(p_temporary_mesh);
}

static void run_naive(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                naive_stencil_func(p_mesh, p_settings);

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "run_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("mesh after iteration %d\n", i);
                        print_mesh(p_mesh, p_settings);
                        printf("\n\n");
                }
        }
}







//____________________________________Partie StarPu _____________________________________________________________________________________




// dans cette partie on a optimisé la fonction naive_stencil pour améliorer le calcul d'un premier lieu et pour mieux facilite le parcour de notre boucle

// On a réussi a optimiser mathématiquement le parcour de nos valeur en jouant sur le reste du division euclidienne

// Notre méthode sera mieux détaillé lors de la présentation
// p_temporary mesh contient que les valeurs qui se calcul dans notre boucle d'origine donc elle est moins grande que p_mesh
//On applique un filtre à p_temporary_mesh pour le diviser sur plusieurs blocs puis on applique cette fonction sur chaque bloc où on stoque les valeurs à p_temp_mesh puis on les redonne apres à p_mesh

//On a pensé a appliquer le filtre block_shadow sur p_mesh pour eviter les conflits lors de la lecture des valeurs de cette matrice
//mais les performances ont été meilleurs quand on laisse p_mesh en mode STARPU_R que en le divisant



// Definition de la fonction de base qui calcul les blocs diviser par le filtre StarPu
void operation(ELEMENT_TYPE *p_temporary_mesh, ELEMENT_TYPE *p_mesh, struct s_settings *p_settings, int len)
{

    // On commence par definir nos variables    
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    const int beg = margin_x + margin_y * p_settings->mesh_width;
    // on a définit p_temporary_mesh avec des valeurs de 0 à len pour pouvoir reprendre notre indice d'origine et prendre la bonne valeur de p_mesh (avec Iorigine=Ig+ibeg+Itot)
    int itot= (int)p_temporary_mesh[0];
    for (int Ig = 0; Ig < len; Ig++)
    {
        p_temporary_mesh[Ig]=p_mesh[Ig + itot+beg];
        int r = (Ig +beg+ itot) % p_settings->mesh_width;
        if (r > margin_x - 1 && r < p_settings->mesh_width)
        {
            ELEMENT_TYPE value = p_mesh[Ig +beg+ itot];
            int stencil_x, stencil_y;

            for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
            {
                for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
                {
                    value += p_mesh[(stencil_y - margin_y) * p_settings->mesh_width + (stencil_x - margin_x) + Ig+beg + itot] * stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                }
            }
            p_temporary_mesh[Ig] = value;
        }
    }
}

// On définit notre kernel qui est appelé dans notre codelet
void stencil_starpu_kernel(void *buffers[], void *cl_args)
{
    struct starpu_vector_interface *vector_handle = buffers[0];
    struct starpu_vector_interface *vector_handle_temp = buffers[1];
    ELEMENT_TYPE *p_mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(vector_handle);
    ELEMENT_TYPE *p_temporary_mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(vector_handle_temp);
    int len = (int)STARPU_VECTOR_GET_NX(vector_handle_temp);
    // on extrait les arguments
    struct s_settings *p_settings;
    
    starpu_codelet_unpack_args(cl_args, &p_settings);
        /// appel de la fonction qui fait les operations sur les blocs
    operation(p_temporary_mesh, p_mesh, p_settings, len);
}

// definition de notre codelet
struct starpu_codelet stencil_codelet =
    {
        .cpu_funcs = {stencil_starpu_kernel},
        .nbuffers = 2,                       // 2 buffers p_mesh et des blocs de p_mesh_temp 
        .modes = {STARPU_R, STARPU_RW},   // P_mesh est read-only pour optimiser les conflits de lecture, output buffer est Read-Write car on lit les valeurs de nos indices et on les remplaces par la valeurs calculé
};






//la fonction qui est appelé dans chaque boucle du nouveau run(pour eviter de unregister et register et definir notre partitionnement a chaque boucle et donc gagner en temps de calcul)
void parallel_stencil_func(ELEMENT_TYPE* restrict p_mesh,ELEMENT_TYPE* restrict p_temporary_mesh, struct s_settings *p_settings, starpu_data_handle_t *handle_mesh,starpu_data_handle_t *sub_x_handles,int len)
{
    
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    const int beg = margin_x + margin_y * p_settings->mesh_width;

    int nb_parts = p_settings->parts;


    // crée nb_part (vous pouvez le changer soit avec --nb_part lors de l'execution ou en haut du code dans une macro)
    for (int i = 0; i < nb_parts; i++)
    {
      
        starpu_task_insert(&stencil_codelet, STARPU_R, handle_mesh[0], STARPU_RW, sub_x_handles[i], STARPU_VALUE, &p_settings, sizeof(p_settings),0);

    }

        // on attend que tout les taches finissent leur calcul pour regrouper notre vecteur
    starpu_task_wait_for_all();

// On avait 3 choix ici soit de faire le boucle commenter ci-dessous naive soit de créer des taches StarPu 
//et la meilleur options qu'on a estimé et de parcourir les lignes et copier rapidement avec memcpy notre ligne

 /*  
    for (int x = margin_x; x < p_settings->mesh_width - margin_x; x++) {
        for (int y = margin_y; y < p_settings->mesh_height - margin_y; y++) {
            p_mesh[y * p_settings->mesh_width + x] = p_temporary_mesh[y * p_settings->mesh_width + x - beg];
        }
    }
*/


for (int y = margin_y; y < p_settings->mesh_height - margin_y; y++) {
        
            memcpy(&p_mesh[(y * p_settings->mesh_width)+margin_x],&p_temporary_mesh[y * p_settings->mesh_width+margin_x- beg],(p_settings->mesh_width-(2*margin_x))*sizeof(ELEMENT_TYPE));
        
    }



    
}




// ________________Modification du run pour StarPu
static void run_starpu(ELEMENT_TYPE *p_mesh,ELEMENT_TYPE *p_temporary_mesh, struct s_settings *p_settings,int len)
{
    int nb_parts = p_settings->parts;
    // On crée un nouveau p_temporary_mesh1 pour pouvoir copier p_mesh rapidement à chaque boucle on utilise plus de memoire mais ca reste plus aventageux
    ELEMENT_TYPE *p_temporary_mesh1 = malloc((len) * sizeof(*p_temporary_mesh1));
    for (int i = 0; i < len; i++) {
        p_temporary_mesh1[i] = i;
    }

    // On crée nos handle et on enregistre nos données dans les registres de StarPu
    starpu_data_handle_t handle_mesh[2];
	starpu_vector_data_register(&handle_mesh[0], STARPU_MAIN_RAM, (uintptr_t)p_mesh, p_settings->mesh_width * p_settings->mesh_height, sizeof(p_mesh[0]));
    starpu_vector_data_register(&handle_mesh[1], STARPU_MAIN_RAM, (uintptr_t)p_temporary_mesh, len , sizeof(p_mesh[0]));

    // On définit le filtre pour partitionner les taches
    struct starpu_data_filter f_part =
        {
            .filter_func = starpu_vector_filter_block,
            .nchildren = nb_parts,
        };
        //On partitionne
    starpu_data_handle_t sub_x_handles[nb_parts];
    starpu_data_partition_plan(handle_mesh[1], &f_part, sub_x_handles);
    starpu_data_partition_submit(handle_mesh[1], nb_parts, sub_x_handles);
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                // Comme on l'as mentionner on copie nos données rapidement à chaque itération
                memcpy(p_temporary_mesh,p_temporary_mesh1,len*sizeof(ELEMENT_TYPE));
                // On calcule avec la fonction qu'on a définit a la place du naive
                parallel_stencil_func(p_mesh,p_temporary_mesh, p_settings, handle_mesh,sub_x_handles,len );

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "run_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("mesh after iteration %d\n", i);
                        print_mesh(p_mesh, p_settings);
                        printf("\n\n");
                }
        }
        // Le travail pour cette repetition est fini on désalloue les mémoires
    starpu_data_unpartition_submit(handle_mesh[1], nb_parts, sub_x_handles, -1);
    starpu_data_partition_clean(handle_mesh[1], nb_parts, sub_x_handles);
    //free(sub_x_handles);
    //free(handle_mesh);
    //starpu_data_unregister(handle_mesh);
	//starpu_data_unregister(p_temporary_mesh);
 
        starpu_data_unregister(handle_mesh[0]);


    free(p_temporary_mesh1);
}

//______________________________ Fin starPu________________ on a fait quelques modif en main aussi pour initialiser starpu et qlq matrices



//____________________________________Partie NVX2 seule _____________________________________________________________________________________


__attribute__((noinline)) void func_simd_avx2(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings) {

    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    
    //Définir la taille du tableau p_temporary_mesh
    int size_temp = p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_mesh);

    //Allocation de la mémoire pour le tableau p_temporary_mesh
    ELEMENT_TYPE *p_temporary_mesh = malloc(size_temp);

    //Indice de début de la boucle for 
    const int beg = margin_x + margin_y * p_settings->mesh_width;

    //Indice final pour la boucle for
    const int end = p_settings->mesh_width * (p_settings->mesh_height - margin_y) - margin_x;

    int Ig;

    // Boucle vectorisée

    for (Ig = beg; Ig < end-REG_NB_ELEMENTS ; Ig += REG_NB_ELEMENTS) {
            //Définir un registre pour la variable value 
            __m256 reg_value = _mm256_loadu_ps(&p_mesh[Ig]);

            for (int stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++) {

                for (int stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++) {

                    //Loader p_mesh dans reg_p_mesh
                    __m256 reg_p_mesh = _mm256_loadu_ps(&p_mesh[(stencil_y - margin_y)
                                        * p_settings->mesh_width + (stencil_x - margin_x) + Ig]);
                    
                    //Remplir la totalité du reg_stencil_coefs par stencil_coefs[..]
                    __m256 reg_stencil_coefs = _mm256_set1_ps(stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x]);

                    //Multiplier les deux registres 
                    __m256 reg_value2 = _mm256_mul_ps(reg_stencil_coefs, reg_p_mesh);
                    
                    //Ajouter reg_value2 à reg_value 
                    reg_value = _mm256_add_ps(reg_value, reg_value2);
                }
            }
            //Stocker reg_value dans temporary_mesh
            _mm256_storeu_ps(&p_temporary_mesh[Ig], reg_value);
    }

     
     
    // Et si le reste n'est pas calculer on ajoute une boucle pour calculer le reste
    //Cette boucle effectue le calcul de manière séquentielle sur le reste du tableau


    for (; Ig < end; Ig++) {
        int r = Ig % p_settings->mesh_width;
        if (r > margin_x - 1 && r < p_settings->mesh_width - margin_x) {
            ELEMENT_TYPE sum = p_mesh[Ig];
            for (int stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++) {
                for (int stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++) 
                {
                    int mesh_index = (stencil_y - margin_y) * p_settings->mesh_width + (stencil_x - margin_x) + Ig;
                    int stencil_index = stencil_y * STENCIL_WIDTH + stencil_x;
                    sum += p_mesh[mesh_index] * stencil_coefs[stencil_index];
                }
            }
            p_temporary_mesh[Ig] = sum;
        }
    }


    //Copier le tableau p_temporary_mesh dans p_mesh avec une maniére améliorer

    for (int y = margin_y; y < p_settings->mesh_height - margin_y; y++) {
        
            memcpy(&p_mesh[(y * p_settings->mesh_width)+margin_x],&p_temporary_mesh[y * p_settings->mesh_width+margin_x],(p_settings->mesh_width-(2*margin_x))*sizeof(ELEMENT_TYPE));
        
    }

    free(p_temporary_mesh);
}

//___________run nvx2
static void run_nvx2(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                func_simd_avx2(p_mesh, p_settings);

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "run_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("mesh after iteration %d\n", i);
                        print_mesh(p_mesh, p_settings);
                        printf("\n\n");
                }
        }
}



//_______________________________Debut StarPu et Nvx __________ 
//on a recopier l'architecture de StarPu et modifier la fonction operation par operation_starpu_nvx
/// qui reprend le même principe que la version nvx


// On a enlever les commentaires sur les parties similaires et on a changer les noms des fonction tout les changement ont été faites sur la partie operation
// qui fusionne les indices de la méthodes starpu et la vectorisation de nvx
__attribute__((noinline)) void operation_starpu_nvx(ELEMENT_TYPE *p_temporary_mesh, ELEMENT_TYPE *p_mesh, struct s_settings *p_settings, int len)
{  
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    const int beg = margin_x + margin_y * p_settings->mesh_width;
    
    int itot= (int)p_temporary_mesh[0];
    int Ig=0;
    for (; Ig < len-REG_NB_ELEMENTS; Ig+=REG_NB_ELEMENTS)
    {
        p_temporary_mesh[Ig]=p_mesh[Ig + itot+beg];
        __m256 reg_value = _mm256_loadu_ps(&p_mesh[Ig +beg+ itot]);
        int stencil_x, stencil_y;
        __m256 reg_p_mesh;

        __m256 reg_stencil_coefs;
            for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
            {
                for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
                {  
                        reg_p_mesh = _mm256_loadu_ps(&p_mesh[(stencil_y - margin_y) * p_settings->mesh_width + (stencil_x - margin_x) + Ig+beg + itot]); 
        
                        reg_stencil_coefs= _mm256_set1_ps(stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x]);

                        __m256 reg_value2 = _mm256_mul_ps(reg_stencil_coefs, reg_p_mesh);

                        reg_value = _mm256_add_ps(reg_value, reg_value2);
                   
                }
            }
            _mm256_storeu_ps(&p_temporary_mesh[Ig], reg_value);
    }

    for (; Ig < len; Ig++)
    {
        p_temporary_mesh[Ig]=p_mesh[Ig + itot+beg];
        int r = (Ig +beg+ itot) % p_settings->mesh_width;
        if (r > margin_x - 1 && r < p_settings->mesh_width)
        {
            ELEMENT_TYPE value = p_mesh[Ig +beg+ itot];
            int stencil_x, stencil_y;

            for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
            {
                for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
                {
                    value += p_mesh[(stencil_y - margin_y) * p_settings->mesh_width + (stencil_x - margin_x) + Ig+beg + itot] * stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                }
            }
            p_temporary_mesh[Ig] = value;
        }
    }
    
}

void starpu_nvx_starpu_kernel(void *buffers[], void *cl_args)
{
    struct starpu_vector_interface *vector_handle = buffers[0];
    struct starpu_vector_interface *vector_handle_temp = buffers[1];
    ELEMENT_TYPE *p_mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(vector_handle);
    ELEMENT_TYPE *p_temporary_mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(vector_handle_temp);
    int len = (int)STARPU_VECTOR_GET_NX(vector_handle_temp);
    struct s_settings *p_settings;
    starpu_codelet_unpack_args(cl_args, &p_settings);
    operation_starpu_nvx(p_temporary_mesh, p_mesh, p_settings, len);
}

struct starpu_codelet starpu_nvx_codelet =
    {
        .cpu_funcs = {starpu_nvx_starpu_kernel},
        .nbuffers = 2,                      
        .modes = {STARPU_R, STARPU_RW},   
    };
void starpu_nvx_stencil_func(ELEMENT_TYPE* restrict p_mesh,ELEMENT_TYPE* restrict p_temporary_mesh, struct s_settings *p_settings, starpu_data_handle_t *handle_mesh,starpu_data_handle_t *sub_x_handles,int len)
{
    
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    const int beg = margin_x + margin_y * p_settings->mesh_width;

    int nb_parts = p_settings->parts;
    for (int i = 0; i < nb_parts; i++)
    {    
        starpu_task_insert(&starpu_nvx_codelet, STARPU_R, handle_mesh[0], STARPU_RW, sub_x_handles[i], STARPU_VALUE, &p_settings, sizeof(p_settings),0);

    }
    starpu_task_wait_for_all();
        for (int y = margin_y; y < p_settings->mesh_height - margin_y; y++) {
            memcpy(&p_mesh[(y * p_settings->mesh_width)+margin_x],&p_temporary_mesh[y * p_settings->mesh_width+margin_x- beg],(p_settings->mesh_width-(2*margin_x))*sizeof(ELEMENT_TYPE));       
    }
}
static void run_starpu_nvx(ELEMENT_TYPE *p_mesh,ELEMENT_TYPE *p_temporary_mesh, struct s_settings *p_settings,int len)
{
    int nb_parts = p_settings->parts;
    ELEMENT_TYPE *p_temporary_mesh1 = malloc((len) * sizeof(*p_temporary_mesh1));
    for (int i = 0; i < len; i++) {
        p_temporary_mesh1[i] = i;
    }
    starpu_data_handle_t handle_mesh[2];
	starpu_vector_data_register(&handle_mesh[0], STARPU_MAIN_RAM, (uintptr_t)p_mesh, p_settings->mesh_width * p_settings->mesh_height, sizeof(p_mesh[0]));
    starpu_vector_data_register(&handle_mesh[1], STARPU_MAIN_RAM, (uintptr_t)p_temporary_mesh, len , sizeof(p_mesh[0]));

    struct starpu_data_filter f_part =
        {
            .filter_func = starpu_vector_filter_block,
            .nchildren = nb_parts,
        };
    starpu_data_handle_t sub_x_handles[nb_parts];
    starpu_data_partition_plan(handle_mesh[1], &f_part, sub_x_handles);
    starpu_data_partition_submit(handle_mesh[1], nb_parts, sub_x_handles);
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                memcpy(p_temporary_mesh,p_temporary_mesh1,len*sizeof(ELEMENT_TYPE));
                // On change ici l'appelation
                starpu_nvx_stencil_func(p_mesh,p_temporary_mesh, p_settings, handle_mesh,sub_x_handles,len );
                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "run_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh, p_settings);
                        fclose(file);
                }
                if (p_settings->enable_verbose)
                {
                        printf("mesh after iteration %d\n", i);
                        print_mesh(p_mesh, p_settings);
                        printf("\n\n");
                }
        }
    starpu_data_unpartition_submit(handle_mesh[1], nb_parts, sub_x_handles, -1);
    starpu_data_partition_clean(handle_mesh[1], nb_parts, sub_x_handles);
        starpu_data_unregister(handle_mesh[0]);
    free(p_temporary_mesh1);
}




/////////// Fin starpu_nvx














static int check(const ELEMENT_TYPE *p_mesh, ELEMENT_TYPE *p_mesh_copy, struct s_settings *p_settings)
{
    
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                naive_stencil_func(p_mesh_copy, p_settings);

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "check_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh_copy, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("check mesh after iteration %d\n", i);
                        print_mesh(p_mesh_copy, p_settings);
                        printf("\n\n");
                }
        }

        int check = 0;
        int x;
        int y;
        for (y = 0; y < p_settings->mesh_height; y++)
        {
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        ELEMENT_TYPE diff = fabs(p_mesh[y * p_settings->mesh_width + x] - p_mesh_copy[y * p_settings->mesh_width + x]);
                        if (diff > EPSILON)
                        {
                                fprintf(stderr, "check failed [x: %d, y: %d]: run = %lf, check = %lf\n", x, y,
                                        p_mesh[y * p_settings->mesh_width + x],
                                        p_mesh_copy[y * p_settings->mesh_width + x]);
                                check = 1;
                        }
                }
        }

        return check;
}

int main(int argc, char *argv[])
{
        struct s_settings *p_settings = NULL;

        init_settings(&p_settings);
        parse_cmd_line(argc, argv, p_settings);

        ELEMENT_TYPE *p_mesh = NULL;
        allocate_mesh(&p_mesh, p_settings);
        
        
        ELEMENT_TYPE *p_mesh_copy = NULL;
        allocate_mesh(&p_mesh_copy, p_settings);
        int ret;

// Je défini directement p_temporary_mesh et je l'ajoute dans les arguments pour ne pas l'allouer et le desallouer a chaque itération
    const int margin_x = (STENCIL_WIDTH - 1) / 2;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    const int beg = margin_x + margin_y * p_settings->mesh_width;

    const int end = p_settings->mesh_width * (p_settings->mesh_height - margin_y) - margin_x;
    const int len = end - beg + 1;
    

    ELEMENT_TYPE *p_temporary_mesh = malloc((len) * sizeof(*p_temporary_mesh));
    // On initialise starpu
	ret = starpu_init(NULL);
    if (ret != 0)
	{
		exit(EXIT_FAILURE);
	}

	
        {
                if (!p_settings->enable_verbose)
                {
                        print_csv_header();
                }

                int rep;
                for (rep = 0; rep < p_settings->nb_repeat; rep++)
                {
                        if (p_settings->enable_verbose)
                        {
                                printf("repeat %d\n", rep);
                        }

                        init_mesh_values(p_mesh, p_settings);
                        apply_boundary_conditions(p_mesh, p_settings);
                        copy_mesh(p_mesh_copy, p_mesh, p_settings);

                        if (p_settings->enable_verbose)
                        {
                                printf("initial mesh\n");
                                print_mesh(p_mesh, p_settings);
                                printf("\n\n");
                        }

                        struct timespec timing_start, timing_end;
                        // on a définit une fonction run differente pour starpu donc on change les run
                        switch(p_settings->vcode) {
                                case starpu:
                                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                                        run_starpu(p_mesh,p_temporary_mesh, p_settings,len);
                                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                                        break;
                                case naive:
                                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                                        run_naive(p_mesh, p_settings);
                                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                                        break;
                                case nvx:
                                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                                        run_nvx2(p_mesh, p_settings);
                                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                                        break;
                                case starpu_nvx:
                                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                                        run_starpu_nvx(p_mesh,p_temporary_mesh, p_settings,len);
                                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                        }       
                        
                        double timing_in_seconds = (timing_end.tv_sec - timing_start.tv_sec) + 1.0e-9 * (timing_end.tv_nsec - timing_start.tv_nsec);
                        // on verifie pas si verif==0
                        int check_status=0;
                        if(p_settings->verif==1){
                        
                        check_status = check(p_mesh, p_mesh_copy, p_settings);
                        }
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
    
    delete_mesh(&p_temporary_mesh);
        delete_mesh(&p_mesh_copy);
        delete_mesh(&p_mesh);
        delete_settings(&p_settings);
        starpu_shutdown();
        return 0;
}