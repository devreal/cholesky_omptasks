
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "ch_common.h"
#include "../timing.h"

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
#pragma omp parallel
{
#pragma omp single
{
	INIT_TIMING(omp_get_num_threads());
    START_TIMING(TIME_TOTAL);
    {
    START_TIMING(TIME_CREATE);
    for (int k = 0; k < nt; k++) {
        if (block_rank[k*nt+k] == mype) {
#pragma omp task depend(out: A[k][k]) firstprivate(k)
{
        //printf("Computing potrf in k=%d\n", k);
		START_TIMING(TIME_POTRF);
        omp_potrf(A[k][k], ts, ts);
		END_TIMING(TIME_POTRF);
}
        }

        int comm_sentinel; // <-- sentinel, never actual referenced

        if (block_rank[k*nt+k] == mype && np != 1) {
          // use comm_sentinel to make sure this task runs before the communication tasks below
#pragma omp task depend(in: A[k][k], comm_sentinel) firstprivate(k)
{
          //printf("Communicating potrf in k=%d\n", k);
          START_TIMING(TIME_COMM);
          MPI_Request *reqs = NULL;
          int nreqs = 0;
          char send_flags[np];
          reset_send_flags(send_flags);
          for (int kk = k+1; kk < nt; kk++) {
            if (!send_flags[block_rank[k*nt+kk]]) {
              ++nreqs;
              send_flags[block_rank[k*nt+kk]] = 1;
            }
          }
          reqs = malloc(sizeof(MPI_Request)*nreqs);
          nreqs = 0;
          for (int dst = 0; dst < np; dst++) {
            if (send_flags[dst] && dst != mype) {
              MPI_Request send_req;
              //printf("Sending potrf block to %d in k=%d\n", dst, k);
              MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &send_req);
              reqs[nreqs++] = send_req;
            }
          }
          //printf("Waiting for potrf block in k=%d\n", k);
          waitall(reqs, nreqs);
          free(reqs);
          END_TIMING(TIME_COMM);
}
        } else if (block_rank[k*nt+k] != mype) {
          // use comm_sentinel to make sure this task runs before the communication tasks below
#pragma omp task depend(out: B) depend(in:comm_sentinel) firstprivate(k)
{
          START_TIMING(TIME_COMM);
          int recv_flag = 0;
          for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
              recv_flag = 1;
              break;
            }
          }
          if (recv_flag) {
            MPI_Request recv_req;
            MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD, &recv_req);
            //printf("Receiving potrf block from %d in k=%d\n", block_rank[k*nt+k], k);
            waitall(&recv_req, 1);
          }
          END_TIMING(TIME_COMM);
}
        }

        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
#pragma omp task depend(in: A[k][k], comm_sentinel) depend(out: A[k][i]) firstprivate(k, i)
{
			        START_TIMING(TIME_TRSM);
                    omp_trsm(A[k][k], A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
}
                } else {
#pragma omp task depend(in: B, comm_sentinel) depend(out: A[k][i]) firstprivate(k, i)
{
			        START_TIMING(TIME_TRSM);
                    omp_trsm(B, A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
}
                }
            }
        }

#pragma omp task depend(inout: comm_sentinel) firstprivate(k) shared(A)
{
        START_TIMING(TIME_COMM);
        char send_flags[np];
        reset_send_flags(send_flags);
        int nreqs = 0;
        // upper bound in case all our blocks have to be sent
        int max_req = (nt-k)*(np-1);
        MPI_Request *reqs = malloc(sizeof(*reqs)*max_req);
        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype && np != 1) {
                for (int ii = k + 1; ii < i; ii++) {
                    if (!send_flags[block_rank[ii*nt+i]]) {
                      send_flags[block_rank[ii*nt+i]] = 1;
                    }
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (!send_flags[block_rank[i*nt+ii]]) {
                      send_flags[block_rank[i*nt+ii]] = 1;
                    }
                }
                if (!send_flags[block_rank[i*nt+i]]) send_flags[block_rank[i*nt+i]] = 1;
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                        MPI_Request send_req;
                        MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &send_req);
                        reqs[nreqs++] = send_req;
                    }
                }
                reset_send_flags(send_flags);
            }
            if (block_rank[k*nt+i] != mype) {
                int recv_flag = 0;
                for (int ii = k + 1; ii < i; ii++) {
                    if (block_rank[ii*nt+i] == mype) recv_flag = 1;
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (block_rank[i*nt+ii] == mype) recv_flag = 1;
                }
                if (block_rank[i*nt+i] == mype) recv_flag = 1;
                if (recv_flag) {
                    MPI_Request recv_req;
                    MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                    reqs[nreqs++] = recv_req;
                }
            }
        }

        //printf("Waiting for trsm blocks in k=%d\n", k);
        waitall(reqs, nreqs);
        free(reqs);
        END_TIMING(TIME_COMM);
}

        for (int i = k + 1; i < nt; i++) {

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
{
						START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
						END_TIMING(TIME_GEMM);
}
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in: A[k][j], comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
{
						START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], A[k][j], A[j][i], ts, ts);
						END_TIMING(TIME_GEMM);
}
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#pragma omp task depend(in: A[k][i], comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
{
						START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], C[j], A[j][i], ts, ts);
						END_TIMING(TIME_GEMM);
}
                    } else {
#pragma omp task depend(in: comm_sentinel) depend(out: A[j][i]) firstprivate(k, j, i)
{
						START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], C[j], A[j][i], ts, ts);
						END_TIMING(TIME_GEMM);
}
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
#pragma omp task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i)
{
						START_TIMING(TIME_SYRK);
                    omp_syrk(A[k][i], A[i][i], ts, ts);
						END_TIMING(TIME_SYRK);
}
                } else {
#pragma omp task depend(in: comm_sentinel) depend(out: A[i][i]) firstprivate(k, i)
{
						START_TIMING(TIME_SYRK);
                    omp_syrk(C[i], A[i][i], ts, ts);
						END_TIMING(TIME_SYRK);
}
                }
            }
        }
    }
    END_TIMING(TIME_CREATE);
    }
#pragma omp taskwait
    END_TIMING(TIME_TOTAL);
    MPI_Barrier(MPI_COMM_WORLD);
	PRINT_TIMINGS();
	FREE_TIMING();

}// pragma omp single
}// pragma omp parallel
}

