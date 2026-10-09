#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/**
 * Parallel LU Decomposition with MPI
 * 
 * Approach: 1D row distribution (simple, educational)
 * - Each MPI process owns a contiguous set of rows
 * - Gaussian elimination with partial pivoting
 * - PA = LU factorization
 * 
 * For production: use ScaLAPACK (PDGETRF) or SuperLU_DIST
 */

typedef struct {
    double *data;      // Matrix data (stored row-major)
    int n;             // Global matrix dimension
    int local_rows;    // Number of rows owned by this rank
    int start_row;     // Global row index of first local row
    int *pivot;        // Pivot permutation array
} Matrix;

// Function prototypes
Matrix* create_matrix(int n, int local_rows, int start_row);
void destroy_matrix(Matrix *A);
void initialize_random_matrix(Matrix *A, int rank, int size);
void print_matrix(Matrix *A, int rank);
void lu_decompose_mpi(Matrix *A, int rank, int size);
void forward_substitution(Matrix *A, double *b, double *y, int rank, int size);
void back_substitution(Matrix *A, double *y, double *x, int rank, int size);

/**
 * Create matrix structure
 * n: global dimension
 * local_rows: number of rows for this process
 * start_row: global row index where this process's rows begin
 */
Matrix* create_matrix(int n, int local_rows, int start_row)
{
    Matrix *A = (Matrix *)malloc(sizeof(Matrix));
    A->n = n;
    A->local_rows = local_rows;
    A->start_row = start_row;
    A->data = (double *)malloc(local_rows * n * sizeof(double));
    A->pivot = (int *)malloc(n * sizeof(int));
    
    // Initialize pivot array
    for (int i = 0; i < n; i++) {
        A->pivot[i] = i;
    }
    
    return A;
}

void destroy_matrix(Matrix *A)
{
    free(A->data);
    free(A->pivot);
    free(A);
}

/**
 * Initialize random matrix (for testing)
 */
void initialize_random_matrix(Matrix *A, int rank, int size)
{
    srand(rank + 42);
    for (int i = 0; i < A->local_rows * A->n; i++) {
        A->data[i] = (double)rand() / RAND_MAX;
    }
}

/**
 * Print matrix (gather to rank 0 for display)
 */
void print_matrix(Matrix *A, int rank)
{
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== Matrix (first 5x5) ===\n");
    }
    
    for (int row = 0; row < (A->n < 5 ? A->n : 5); row++) {
        int owner = (row * (int)ceil((double)A->n / (int)ceil((double)A->n / A->local_rows))) / A->n;
        
        if (row >= A->start_row && row < A->start_row + A->local_rows) {
            if (rank == 0) {
                printf("Row %d: ", row);
                for (int col = 0; col < (A->n < 5 ? A->n : 5); col++) {
                    printf("%8.4f ", A->data[(row - A->start_row) * A->n + col]);
                }
                printf("\n");
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Parallel LU Decomposition with Partial Pivoting
 * 
 * Factorizes A into PA = LU
 * L stored in strict lower triangle (implicit unit diagonal)
 * U stored in upper triangle
 * pivot[] records row permutations
 */
void lu_decompose_mpi(Matrix *A, int rank, int size)
{
    int n = A->n;
    MPI_Status status;
    
    for (int k = 0; k < n; k++) {
        // ========== Step 1: Find pivot ==========
        double pivot_val = 0.0;
        int pivot_row_global = k;
        
        // Each process searches its eligible local rows (row index >= k)
        if (k >= A->start_row && k < A->start_row + A->local_rows) {
            for (int i = k; i < A->start_row + A->local_rows; i++) {
                int local_idx = i - A->start_row;
                double abs_val = fabs(A->data[local_idx * A->n + k]);
                if (abs_val > pivot_val) {
                    pivot_val = abs_val;
                    pivot_row_global = i;
                }
            }
        }
        
        // Global reduction to find maximum pivot across all processes
        struct {
            double val;
            int idx;
        } local_max, global_max;
        
        local_max.val = pivot_val;
        local_max.idx = pivot_row_global;
        
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        pivot_row_global = global_max.idx;
        pivot_val = global_max.val;
        
        // Check for singular matrix
        if (pivot_val == 0.0) {
            if (rank == 0) {
                printf("Error: Matrix is singular at column %d\n", k);
            }
            return;
        }
        
        // ========== Step 2: Swap rows k and pivot_row_global ==========
        int owner_k = (k >= A->start_row && k < A->start_row + A->local_rows) ? rank : -1;
        int owner_p = (pivot_row_global >= A->start_row && pivot_row_global < A->start_row + A->local_rows) ? rank : -1;
        
        MPI_Allreduce(MPI_IN_PLACE, &owner_k, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &owner_p, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        double *row_k = (double *)malloc(n * sizeof(double));
        double *row_p = (double *)malloc(n * sizeof(double));
        
        // Row k owner broadcasts it
        if (rank == owner_k) {
            memcpy(row_k, &A->data[(k - A->start_row) * A->n], n * sizeof(double));
        }
        MPI_Bcast(row_k, n, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
        
        // Row p owner broadcasts it
        if (rank == owner_p) {
            memcpy(row_p, &A->data[(pivot_row_global - A->start_row) * A->n], n * sizeof(double));
        }
        MPI_Bcast(row_p, n, MPI_DOUBLE, owner_p, MPI_COMM_WORLD);
        
        // Perform swaps locally
        if (rank == owner_k) {
            memcpy(&A->data[(k - A->start_row) * A->n], row_p, n * sizeof(double));
        }
        if (rank == owner_p) {
            memcpy(&A->data[(pivot_row_global - A->start_row) * A->n], row_k, n * sizeof(double));
        }
        
        // Update pivot record
        int temp_pivot = A->pivot[k];
        A->pivot[k] = A->pivot[pivot_row_global];
        A->pivot[pivot_row_global] = temp_pivot;
        
        free(row_k);
        free(row_p);
        
        // ========== Step 3: Broadcast pivot row A[k, k:n] ==========
        double *pivot_row = (double *)malloc((n - k) * sizeof(double));
        
        if (rank == owner_k) {
            memcpy(pivot_row, &A->data[(k - A->start_row) * A->n + k], (n - k) * sizeof(double));
        }
        MPI_Bcast(pivot_row, n - k, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
        
        // ========== Step 4: Update local rows independently (no communication) ==========
        for (int i = k + 1; i < A->n; i++) {
            if (i >= A->start_row && i < A->start_row + A->local_rows) {
                int local_idx = i - A->start_row;
                
                // Compute multiplier L[i,k] = A[i,k] / A[k,k]
                double multiplier = A->data[local_idx * A->n + k] / pivot_row[0];
                A->data[local_idx * A->n + k] = multiplier;
                
                // Update row i: A[i, k+1:n] -= L[i,k] * A[k, k+1:n]
                for (int j = k + 1; j < n; j++) {
                    A->data[local_idx * A->n + j] -= multiplier * pivot_row[j - k];
                }
            }
        }
        
        free(pivot_row);
    }
    
    if (rank == 0) {
        printf("LU decomposition completed successfully.\n");
    }
}

/**
 * Forward substitution: Solve Ly = Pb
 * L is lower triangular with unit diagonal, stored in strict lower triangle of A
 */
void forward_substitution(Matrix *A, double *b, double *y, int rank, int size)
{
    // Apply permutation: y = P*b
    double *b_perm = (double *)malloc(A->n * sizeof(double));
    for (int i = 0; i < A->n; i++) {
        b_perm[i] = b[A->pivot[i]];
    }
    
    // Ly = b_perm
    for (int i = 0; i < A->n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            if (i >= A->start_row && i < A->start_row + A->local_rows) {
                int local_idx = i - A->start_row;
                // L[i,j] is stored in A[i,j] for i > j
                sum += A->data[local_idx * A->n + j] * y[j];
            }
        }
        // This part requires gathering y values; simplified for local operation
        y[i] = b_perm[i] - sum;
    }
    
    free(b_perm);
}

/**
 * Back substitution: Solve Ux = y
 * U is upper triangular, stored in upper triangle of A
 */
void back_substitution(Matrix *A, double *y, double *x, int rank, int size)
{
    // Ux = y
    for (int i = A->n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < A->n; j++) {
            if (i >= A->start_row && i < A->start_row + A->local_rows) {
                int local_idx = i - A->start_row;
                sum += A->data[local_idx * A->n + j] * x[j];
            }
        }
        x[i] = (y[i] - sum) / A->data[(i - A->start_row) * A->n + i];
    }
}

/**
 * Main program
 */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 8;  // Matrix dimension (should be divisible by size for simplicity)
    
    if (rank == 0) {
        printf("=== Parallel LU Decomposition with MPI ===\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Number of processes: %d\n\n", size);
    }
    
    // Distribute rows evenly
    int local_rows = n / size;
    int start_row = rank * local_rows;
    
    // Create matrix
    Matrix *A = create_matrix(n, local_rows, start_row);
    initialize_random_matrix(A, rank, size);
    
    if (rank == 0) {
        printf("Original matrix (first 5x5):\n");
    }
    print_matrix(A, rank);
    
    // Perform LU decomposition
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    lu_decompose_mpi(A, rank, size);
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("LU factorization time: %f seconds\n", end_time - start_time);
        printf("\nFactorized matrix (first 5x5):\n");
    }
    print_matrix(A, rank);
    
    // Cleanup
    destroy_matrix(A);
    
    MPI_Finalize();
    return 0;
}
