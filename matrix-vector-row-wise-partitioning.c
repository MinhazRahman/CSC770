#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

// Useful macros
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW(id + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_LOW(id + 1, p, n) - BLOCK_LOW(id, p, n))
#define BLOCK_OWNER(index, p, n) (((p * index + 1) - 1) / n)

// Function declarations
void read_row_striped_matrix(int matrix[], int rows, int cols, int id, int p);
void read_vector(int vector[], int n, int id, int p);
void multiply_matrix_vector(int matrix[], int vector[], int result[], int id, int p, int rows, int cols);
void gather_result(int result[], int id, int p, int n);

int main(int argc, char *argv[])
{
    int id, p, rows, cols;
    double start_time;

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Barrier so all processes start together from this point and then record start time
    MPI_Barrier(MPI_COMM_WORLD);
    if (!id)
        start_time = MPI_Wtime();

    // root process = p - 1
    // Only root process can read the size fo the matrix and vector from files
    if (!id)
    {
        FILE *ftr;
        ftr = fopen("matrix.txt", "r");
        fscanf(ftr, "%d", &rows);
        fscanf(ftr, "%d", &cols);
        fclose(ftr);

        ftr = fopen("vector.txt", "r");
        int vec_rows;
        fscanf(ftr, "%d", &vec_rows);
        fclose(ftr);

        // If number of columns of matrix does not equal number of rows of vector then exit
        if (vec_rows != cols)
            exit(1);
    }

    // Broadcast the number of rows and columns to other processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for local matrix, vector and result vector
    int *matrix = (int *)malloc(BLOCK_SIZE(id, p, rows) * cols * sizeof(int));
    int *vector = (int *)malloc(cols * sizeof(int));
    int *result = (int *)malloc(BLOCK_SIZE(id, p, rows) * sizeof(int));
    for (int i = 0; i < BLOCK_SIZE(id, p, rows); i++)
        result[i] = 0;

    // Read the matrix and vector from input files
    read_row_striped_matrix(matrix, rows, cols, id, p);
    read_vector(vector, cols, id, p);

    // Multiply the matrix and vector, gather and then print the result
    multiply_matrix_vector(matrix, vector, result, id, p, rows, cols);
    gather_result(result, id, p, rows);

    // Process 0 will print the result
    if (!id)
    {
        printf("Time for %d process and %d rows:\n ", p, rows);
        double t = (MPI_Wtime() - start_time) * 1000;
        printf("%f ms\n", t);
    }
    MPI_Finalize();
    return 0;
}

void read_row_striped_matrix(int matrix[], int rows, int cols, int id, int p)
{
    // read the matrix from the file
    int value;
    int *temp_mat = NULL;
    int sum = 0;

    int *sendcounts = (int *)malloc(sizeof(int) * p);
    int *displs = (int *)malloc(sizeof(int) * p);

    // This loop calculates the displacement and count of the various blocks to be sent to other processes
    for (int i = 0; i < p; i++)
    {
        sendcounts[i] = cols * BLOCK_SIZE(i, p, rows);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // If process 0 then read from file to temp_mat and scatter blocks to other processes
    if (!id)
    {
        temp_mat = (int *)malloc(rows * cols * sizeof(int));
        int num;
        FILE *ftr;
        ftr = fopen("matrix.txt", "r");
        fscanf(ftr, "%d", &num);
        fscanf(ftr, "%d", &num);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                fscanf(ftr, "%d", &num);
                temp_mat[i * cols + j] = num;
            }
        }
        fclose(ftr);
        MPI_Scatterv(temp_mat, sendcounts, displs, MPI_INT, matrix, sendcounts[id], MPI_INT, 0, MPI_COMM_WORLD);
        free(temp_mat);
    }
    // If not process 0 then receive from process 0
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, NULL, matrix, sendcounts[id], MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void read_vector(int vector[], int n, int id, int p)
{
    // If process 0 then read vector from file and broadcast to other processes
    if (!id)
    {
        int num;
        FILE *ftr;
        ftr = fopen("vector.txt", "r");
        fscanf(ftr, "%d", &num);
        fscanf(ftr, "%d", &num);
        for (int i = 0; i < n; i++)
        {
            fscanf(ftr, "%d", &num);
            vector[i] = num;
        }
        fclose(ftr);
    }
    MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
}

void multiply_matrix_vector(int matrix[], int vector[], int result[], int id, int p, int rows, int cols)
{
    for (int i = 0; i < BLOCK_SIZE(id, p, rows); i++)
        for (int j = 0; j < cols; j++)
            result[i] += matrix[i * cols + j] * vector[j];
}

void gather_result(int result[], int id, int p, int n)
{
    int sum = 0;
    int *temp_mat = (int *)malloc(n * sizeof(int));
    int *sendcounts = (int *)malloc(p * sizeof(int));
    int *displs = (int *)malloc(p * sizeof(int));

    for (int i = 0; i < p; i++)
    {
        sendcounts[i] = BLOCK_SIZE(i, p, n);
        displs[i] = sum;
        sum += sendcounts[i];
    }
    MPI_Gatherv(result, sendcounts[id], MPI_INT, temp_mat, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (!id)
    {
        printf("Result vector:\n");
        for (int i = 0; i < n; i++)
            printf("%d\n", temp_mat[i]);
    }
}
