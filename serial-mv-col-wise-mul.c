#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * The code reads matrix from file matrix.txt and vector from vector.txt
 * the first line of each file indicates the number of rows and columns
 * followed by the matrix or vector
 *
 *  To compile and run:
 *  mpicc SerialMatrixMul.c -o SerialMatrixMul
 *  mpirun -np 2 SerialMatrixMul
 */

int main(int argc, char *argv[])
{
    int nproces, mypid, matrix_rows, matrix_cols, vector_rows;
    double **matrix;
    double *vector;
    double *result_vector;

    MPI_Init(&argc, &argv);
    // determines the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproces);
    // determines the identifier of the calling process
    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);

    // root process = p - 1
    // only root node reads the size of the matrix as well as the size of the vector
    if (!mypid)
    {
        FILE *file;
        file = fopen("matrix.txt", "r");
        fscanf(file, "%d", &matrix_rows);
        fscanf(file, "%d", &matrix_cols);
        fclose(file);

        file = fopen("vector.txt", "r");
        fscanf(file, "%d", &vector_rows);
        fclose(file);
        printf("matrix_rows: %d matrix_cols: %d vector_rows: %d\n", matrix_rows, matrix_cols, vector_rows);

        // If the number of columns of the matrix is not equal to the number of rows of the vector
        // then matrix-vector multiplication can't be performed
        if (matrix_cols != vector_rows)
        {
            exit(1);
        }

        // initialize the matrix
        matrix = (double **)malloc(matrix_rows * sizeof(double *));
        for (int i = 0; i < matrix_rows; i++)
        {
            matrix[i] = (double *)malloc(matrix_rows * sizeof(double));
        }

        // read the values from the matrix.txt file
        int val;
        file = fopen("matrix.txt", "r");
        fscanf(file, "%d", &val);
        fscanf(file, "%d", &val);
        for (int i = 0; i < matrix_rows; i++)
        {
            for (int j = 0; j < matrix_cols; j++)
            {
                fscanf(file, "%d", &val);
                matrix[i][j] = val;
            }
        }
        fclose(file);

        // initialize the vector
        vector = (double *)malloc(vector_rows * sizeof(double));

        // read the values from vector.txt file
        int num;
        file = fopen("vector.txt", "r");
        fscanf(file, "%d", &num);
        fscanf(file, "%d", &num);
        for (int i = 0; i < vector_rows; i++)
        {
            fscanf(file, "%d", &num);
            vector[i] = num;
        }
        fclose(file);

        // initialize the result vector
        result_vector = (double *)malloc(vector_rows * sizeof(double));

        // multiply matrix-vector column-wise
        // The tasks are associated with a column of the matrix and a single vector element
        // Each iteration will compute a partial sum associated with several elements of result_vector[n].
        // The partial sums will need to be added to other iterations
        for (int i = 0; i < matrix_rows; i++)
        {
            for (int j = 0; j < matrix_cols; j++)
            {
                result_vector[j] += matrix[j][i] * vector[i];
            }
        }

        // print the result vector
        for (int i = 0; i < matrix_rows; i++)
        {
            printf("%f\n", result_vector[i]);
        }
    }

    MPI_Finalize();

    return 0;
}
