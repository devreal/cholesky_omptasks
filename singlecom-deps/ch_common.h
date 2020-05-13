#ifndef _BENCH_CHOLESKY_COMMON_
#define _BENCH_CHOLESKY_COMMON_

#include <mkl.h>
#include <mpi.h>
#include <omp.h>

// #define DEBUG_PRINT(STR, ...)
// #define DEBUG_PRINT_WINFO(STR, ...)

#define DEBUG_PRINT(STR, ...) do { \
    fprintf(stderr, STR, __VA_ARGS__); \
} while(0)
#define DEBUG_PRINT_WINFO(STR, ...) do { \
    char* tmp_str = malloc(sizeof(char)*512); \
    tmp_str[0] = '\0'; \
    strcat(tmp_str,"R#%02d T#%02d (OS_TID:%06ld): --> "); \
    strcat(tmp_str,STR); \
    fprintf(stderr, tmp_str, mype, omp_get_thread_num(), syscall(SYS_gettid), __VA_ARGS__); \
    free(tmp_str); \
} while(0)

#ifdef _USE_HBW
#include <hbwmalloc.h>
#endif

void dgemm_ (const char *transa, const char *transb, int *l, int *n, int *m, double *alpha,
             const void *a, int *lda, void *b, int *ldb, double *beta, void *c, int *ldc);

void dtrsm_ (char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha,
             double *a, int *lda, double *b, int *ldb);

void dsyrk_ (char *uplo, char *trans, int *n, int *k, double *alpha, double *a, int *lda,
             double *beta, double *c, int *ldc);

void cholesky_single(const int ts, const int nt, double* A[nt][nt]);
void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank);

void omp_potrf(double * const A, int ts, int ld);
void omp_trsm(double *A, double *B, int ts, int ld);
void omp_gemm(double *A, double *B, double *C, int ts, int ld);
void omp_syrk(double *A, double *B, int ts, int ld);

inline static void waitall(MPI_Request *comm_req, int n)
{
#ifdef DISABLE_TASKYIELD
  MPI_Waitall(n, comm_req, MPI_STATUSES_IGNORE);
#else
  while (1) {
    int flag = 0;
    MPI_Testall(n, comm_req, &flag, MPI_STATUSES_IGNORE);
    if (flag) break;
    (void)flag; // <-- make the Cray compiler happy
#pragma omp taskyield
  }
#endif
}
void reset_send_flags(char *send_flags);

#ifdef MAIN
int np;
int mype;
int num_threads;
#else
extern int np;
extern int mype;
extern int num_threads;
#endif

#endif
