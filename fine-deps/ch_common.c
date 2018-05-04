
#define MAIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "ch_common.h"
#include "cholesky.h"

static void get_block_rank(int *block_rank, int nt);

void omp_potrf(double * const A, int ts, int ld)
{
    static int INFO;
    static const char L = 'L';
    dpotrf_(&L, &ts, A, &ld, &INFO);
}
void omp_trsm(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    static double DONE = 1.0;
    dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}
void omp_gemm(double *A, double *B, double *C, int ts, int ld)
{
    static const char TR = 'T', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}
void omp_syrk(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}

void cholesky_single(const int ts, const int nt, double* A[nt][nt])
{
    for (int k = 0; k < nt; k++) {
#pragma omp task depend(out: A[k][k])
{
        omp_potrf(A[k][k], ts, ts);
#ifdef DEBUG
        if (mype == 0) printf("potrf:out:A[%d][%d]\n", k, k);
#endif
}
        for (int i = k + 1; i < nt; i++) {
#pragma omp task depend(in: A[k][k]) depend(out: A[k][i])
{
            omp_trsm(A[k][k], A[k][i], ts, ts);
#ifdef DEBUG
        if (mype == 0) printf("trsm :in:A[%d][%d]:out:A[%d][%d]\n", k, k, k, i);
#endif
}
        }
        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {
#pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i])
{
                omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
#ifdef DEBUG
                if (mype == 0) printf("gemm :in:A[%d][%d]:A[%d][%d]:out:A[%d][%d]\n", k, i, k, j, j, i);
#endif
}
            }
#pragma omp task depend(in: A[k][i]) depend(out: A[i][i])
{
            omp_syrk(A[k][i], A[i][i], ts, ts);
#ifdef DEBUG
            if (mype == 0) printf("syrk :in:A[%d][%d]:out:A[%d][%d]\n", k, i, i, i);
#endif
}
        }
    }
#pragma omp taskwait
}

inline void wait(MPI_Request *comm_req)
{
    int comm_comp = 0;

    MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    while (!comm_comp) {
#pragma omp taskyield
        MPI_Test(comm_req, &comm_comp, MPI_STATUS_IGNORE);
    }
//    MPI_Wait(comm_req, MPI_STATUS_IGNORE);
}

inline void reset_send_flags(char *send_flags)
{
    for (int i = 0; i < np; i++) send_flags[i] = 0;
}

int main(int argc, char *argv[])
{
    /* MPI Initialize */
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("This Compiler does not support MPI_THREAD_MULTIPLE\n");
        exit(0);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* cholesky init */
    const char *result[3] = {"n/a","successful","UNSUCCESSFUL"};
    const double eps = BLAS_dfpinfo(blas_eps);

    if (argc < 4) {
        printf("cholesky matrix_size block_size check\n");
        exit(-1);
    }
    const int  n = atoi(argv[1]); // matrix size
    const int ts = atoi(argv[2]); // tile size
    int check    = atoi(argv[3]); // check result?

    const int nt = n / ts;

    if (mype == 0)
        printf("nt = %d, ts = %d\n", nt, ts);

    /* Set block rank */
    int *block_rank = malloc(nt * nt * sizeof(int));
    get_block_rank(block_rank, nt);

#ifdef DEBUG
    if (mype == 0) {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                printf("%d ", block_rank[i * nt + j]);
            }
            printf("\n");
        }
    }
#endif

    double *A[nt][nt], *B, *C[nt], *Ans[nt][nt];

#pragma omp parallel
{
#pragma omp single
{
  for (int i = 0; i < nt; i++) {
    for (int j = 0; j < nt; j++) {
#pragma omp task depend(out: A[i][j]) shared(Ans, A)
{
      if (check) {
        MPI_Alloc_mem(ts * ts * sizeof(double), MPI_INFO_NULL, &Ans[i][j]);
        initialize_tile(ts, Ans[i][j]);
      }
      if (block_rank[i*nt+j] == mype) {
        MPI_Alloc_mem(ts * ts * sizeof(double), MPI_INFO_NULL, &A[i][j]);
        if (!check) {
          initialize_tile(ts, A[i][j]);
        } else {
          for (int k = 0; k < ts * ts; k++) {
            A[i][j][k] = Ans[i][j][k];
          }
        }
      }
}
    }
#pragma omp task depend(inout: A[i][i]) shared(Ans, A)
{
    // add to diagonal
    if (check) {
      Ans[i][i][i*ts+i] = (double)nt;
    }
    if (block_rank[i*nt+i] == mype) {
      A[i][i][i*ts+i] = (double)nt;
    }
}
  }
} // omp single
} // omp parallel

  MPI_Alloc_mem(ts * ts * sizeof(double), MPI_INFO_NULL, &B);
  for (int i = 0; i < nt; i++) {
    MPI_Alloc_mem(ts * ts * sizeof(double), MPI_INFO_NULL, &C[i]);
  }

#pragma omp single
    num_threads = omp_get_num_threads();

    const float t3 = get_time();
    if (check) cholesky_single(ts, nt, (double* (*)[nt]) Ans);
    const float t4 = get_time() - t3;

    MPI_Barrier(MPI_COMM_WORLD);

    if (mype == 0)
      printf("Starting parallel computation\n");
    const float t1 = get_time();
    cholesky_mpi(ts, nt, (double* (*)[nt])A, B, C, block_rank);
    const float t2 = get_time() - t1;
    if (mype == 0)
      printf("Finished parallel computation\n");

    MPI_Barrier(MPI_COMM_WORLD);

    /* Verification */
    if (check) {
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < nt; j++) {
                if (block_rank[i * nt + j] == mype) {
                    for (int k = 0; k < ts*ts; k++) {
                        if (Ans[i][j][k] != A[i][j][k]) check = 2;
                    }
                }
            }
        }
    }

    float time_mpi = t2;
    float gflops_mpi = (((1.0 / 3.0) * n * n * n) / ((time_mpi) * 1.0e+9));
    float time_ser = t4;
    float gflops_ser = (((1.0 / 3.0) * n * n * n) / ((time_ser) * 1.0e+9));

    printf("test:%s-%d-%d-%d:mype:%2d:np:%2d:threads:%2d:result:%s:gflops:%f:time:%f:gflops_ser:%f:time_ser:%f\n", argv[0], n, ts, num_threads, mype, np, num_threads, result[check], gflops_mpi, t2, gflops_ser, t4);

    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            if (block_rank[i*nt+j] == mype) {
                free(A[i][j]);
            }
            if (check)
              free(Ans[i][j]);
        }
        free(C[i]);
    }
    free(B);
    free(block_rank);

    MPI_Finalize();

    return 0;
}

static void get_block_rank(int *block_rank, int nt)
{
    int row, col;
    row = col = np;

    if (np != 1) {
        while (1) {
            row = row / 2;
            if (row * col == np) break;
            col = col / 2;
            if (row * col == np) break;
        }
    }
    if (mype == 0) printf("row = %d, col = %d\n", row, col);

    int i, j, tmp_rank = 0, offset = 0;
    for (i = 0; i < nt; i++) {
        for (j = 0; j < nt; j++) {
            block_rank[i*nt + j] = tmp_rank + offset;
            tmp_rank++;
            if (tmp_rank >= col) tmp_rank = 0;
        }
        tmp_rank = 0;
        offset = (offset + col >= np) ? 0 : offset + col;
    }
}
