# Parallel LU Decomposition with MPI

A C++ implementation of **LU decomposition with partial pivoting** using **Message Passing Interface (MPI)** for distributed memory parallelism.

## Overview

This project demonstrates the **1D row distribution** approach for parallel LU factorization:

- **PA = LU** factorization using Gaussian elimination with partial pivoting
- Each MPI process owns a contiguous set of matrix rows
- Distributed pivot selection using `MPI_Allreduce` with `MPI_MAXLOC`
- Row exchange via `MPI_Sendrecv`
- Pivot row broadcast with `MPI_Bcast`
- Local independent updates with **no communication** between processes

### Key MPI Operations

| Operation | MPI Call | Purpose |
|-----------|----------|---------|
| Find global pivot | `MPI_Allreduce(MPI_MAXLOC)` | Locate largest entry across all processes |
| Broadcast pivot row | `MPI_Bcast` | Share row data with all processes |
| Row exchange | `MPI_Sendrecv` | Swap pivot rows between processes |
| Synchronization | `MPI_Barrier` | Ensure all processes reach checkpoint |

## Algorithm

### Gaussian Elimination with Partial Pivoting (Column k)

```
1. Find pivot:
   - Each process finds max(|A[i,k]|) for its local rows i ≥ k
   - MPI_Allreduce(MAXLOC) returns global maximum and its row
   
2. Swap rows k and pivot_row:
   - Identify which processes own these rows
   - Exchange full rows (including stored multipliers)
   
3. Broadcast pivot row:
   - Owner of row k broadcasts A[k, k:n] to all processes
   
4. Update trailing matrix (parallel, no communication):
   - Each process computes multipliers: L[i,k] = A[i,k] / A[k,k]
   - Updates: A[i,j] -= L[i,k] * A[k,j] for j > k
```

### Result Storage

After factorization:
- **Strict lower triangle**: multipliers (L with implicit unit diagonal)
- **Upper triangle**: U matrix
- **Diagonal**: pivot values from U
- **`pivot[]` array**: row permutation records

## Building and Running

### Prerequisites

- C/C++ compiler with MPI support (OpenMPI or MPICH)
- CMake 3.10 or higher
- Standard math library

### Build

```bash
mkdir build
cd build
cmake ..
make
```

### Run

```bash
# Serial (for testing)
./mpi_lu

# Parallel with 4 processes
mpirun -np 4 ./mpi_lu

# Parallel with 8 processes
mpirun -np 8 ./mpi_lu
```

## Code Structure

### `src/mpi_lu.cpp`

- **`Matrix` struct**: Encapsulates matrix data, dimensions, and pivot information
- **`create_matrix()` / `destroy_matrix()`**: Memory management
- **`lu_decompose_mpi()`**: Main factorization routine
- **`forward_substitution()` / `back_substitution()`**: Solve systems using factors
- **`main()`**: Driver program

### Key Functions

#### `void lu_decompose_mpi(Matrix *A, int rank, int size)`

Performs parallel LU factorization:
- Loop over columns k = 0 to n-1
- Each iteration:
  1. Find global pivot with `MPI_Allreduce(MPI_DOUBLE_INT, MPI_MAXLOC, ...)`
  2. Determine process owners via reduction
  3. Broadcast rows for swapping
  4. Perform local multiplier and update computations

## Communication Complexity

For an **n × n** matrix with **p** processes:

- **Number of communication rounds**: O(n)
- **Per-round collective ops**: 3–4 (allreduce, bcast, barrier)
- **Per-process computation**: O(n²/p) in elimination phase

**Scaling characteristics:**
- Suitable for **moderate p** (2–16 on distributed memory)
- Beyond this, blocked LU with 2D process grid (see Advanced section) is preferred

## Example Output

```
=== Parallel LU Decomposition with MPI ===
Matrix size: 8 x 8
Number of processes: 4

Original matrix (first 5x5):
Row 0:   0.1234   0.5678   0.9012   0.3456   0.7890
Row 1:   0.2341   0.6789   0.0123   0.4567   0.8901
Row 2:   0.3452   0.7890   0.1234   0.5678   0.9012
Row 3:   0.4563   0.8901   0.2345   0.6789   0.0123
Row 4:   0.5674   0.9012   0.3456   0.7890   0.1234

LU decomposition completed successfully.
LU factorization time: 0.000234 seconds
```

## Advanced Topics

### For Production Use

This implementation is **educational**. For real applications, use:

- **Dense matrices**: `ScaLAPACK PDGETRF` (double) or `PZGETRF` (complex)
  - 2D block-cyclic distribution
  - Blocked algorithm for efficiency
  - Optimized BLAS kernels

- **Sparse matrices**: `SuperLU_DIST` or `MUMPS`
  - Handles fill-in and ordering
  - Distributed sparse storage
  - Much higher complexity but far better performance on large sparse systems

### Blocked LU (2D Distribution)

Replace column-by-column with panel processing:

```text
For each panel of b columns:
  1. Factor panel (small local LU)
  2. Apply row permutations to trailing matrix
  3. Compute block row of U (triangular solve)
  4. Update trailing submatrix: A22 -= L21 * U12 (GEMM, highly parallel)
```

**Benefits:**
- Most work becomes matrix–matrix multiplication (GEMM)
- Better load balance
- Reduced communication per flop
- Scales to 100s of processes

## References

1. **ScaLAPACK**: [http://www.netlib.org/scalapack/](http://www.netlib.org/scalapack/)
2. **SuperLU_DIST**: [https://portal.nersc.gov/project/sparse/superlu/](https://portal.nersc.gov/project/sparse/superlu/)
3. **Golub & Van Loan** (1996). *Matrix Computations*, 3rd ed. Johns Hopkins University Press.
4. **Demmel et al.** (1993). "A Supernodal Approach to Sparse Partial Pivoting." SIAM J. Matrix Anal. Appl.

## License

MIT License - see LICENSE file for details.
