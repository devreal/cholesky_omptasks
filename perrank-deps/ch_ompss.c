
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "ch_common.h"
#include "../extrae.h"
#include "../timing.h"

/**
 * TODO: What is the lower bound for a circular deadlock? (0 waits for 1 waits for 2 waits for 0)
 * Example: Execution order on 1 is reversed:
 *    0 waits for 1/2/3,
 *    1 waits for 3/2/0,
 *    2 waits for 0/1/3,
 *    3 waits for 0/1/2
 * OR
 *    0 waits for 1/2/3/4,
 *    1 waits for 0/2/3/4,
 *    2 waits for 4/3/2/0,
 *    3 waits for 0/1/2/4,
 *    4 waits for 0/1/2/3
 * OR
 *    0 waits for 1/2/3/4/5,
 *    1 waits for 0/2/3/4/5,
 *    2 waits for 5/4/3/2/0,
 *    3 waits for 0/1/2/4/5,
 *    4 waits for 0/1/2/3/5,
 *    5 waits for 0/1/2/3/5
 *
 * NOTE: circular dependencies may happen if at least one of the inner ranks
 *       (1 or 2, not 0 or 3) reverse their order
 * HYPOTHESIS: we need at least (p-(p/2)) (ceil(0.5p)) threads to avoid deadlock from reversal
 * Generalization to some ordered graph traversal problem?
 */

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
  int *send_blocks = malloc((nt) * sizeof(int));
  int *recv_blocks = malloc((nt) * sizeof(int));
  REGISTER_EXTRAE();
#pragma omp parallel
{
#pragma omp single
{
	INIT_TIMING(omp_get_num_threads());
    char dst_sentinels[np];
    START_TIMING(TIME_TOTAL);
    {
    START_TIMING(TIME_CREATE);
    for (int k = 0; k < nt; k++) {
        if (block_rank[k*nt+k] == mype) {
#pragma omp task out(A[k][k]) firstprivate(k) no_copy_deps
{
			EXTRAE_ENTER(EVENT_POTRF);
			START_TIMING(TIME_POTRF);
            omp_potrf(A[k][k], ts, ts);
			END_TIMING(TIME_POTRF);
			EXTRAE_EXIT(EVENT_POTRF);
}
        }

        if (block_rank[k*nt+k] == mype && np != 1) {
#pragma omp task in(A[k][k]) firstprivate(k) no_copy_deps untied
{
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
              MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &send_req);
              reqs[nreqs++] = send_req;
            }
          }
          waitall(reqs, nreqs);
          free(reqs);
          END_TIMING(TIME_COMM);
}
        }

        if (block_rank[k*nt+k] != mype) {
#pragma omp task out(B) firstprivate(k) no_copy_deps untied
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
              waitall(&recv_req, 1);
            }
            END_TIMING(TIME_COMM);
}
        }

        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                if (block_rank[k*nt+k] == mype) {
#pragma omp task in(A[k][k]) out(A[k][i]) firstprivate(k, i) no_copy_deps
{
					EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(A[k][k], A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
					EXTRAE_EXIT(EVENT_TRSM);
}
                } else {
#pragma omp task in(B) out(A[k][i]) firstprivate(k, i) no_copy_deps
{
					EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(B, A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
					EXTRAE_EXIT(EVENT_TRSM);
}
                }
            }
        }

      for (int dst = 0; dst < np; dst++) {
        if (dst == mype) continue;

        int send_cnt = 0;
        int recv_cnt = 0;
        // populate list of blocks to send/recv to/from this unit
        for (int i = k + 1; i < nt; i++) {
          if (block_rank[k*nt+i] == mype && np != 1) {
            int send_flag = 0;
            for (int ii = k + 1; ii < i; ii++) {
                if (!send_flag && block_rank[ii*nt+i] == dst) {
                  send_flag = 1;
                  break;
                }
            }
            for (int ii = i + 1; ii < nt; ii++) {
                if (!send_flag && block_rank[i*nt+ii] == dst) {
                  send_flag = 1;
                  break;
                }
            }
            if (!send_flag && block_rank[i*nt+i] == dst) send_flag = 1;
            if (send_flag) {
              send_blocks[send_cnt++] = i;
            }
          }
          if (block_rank[k*nt+i] != mype && block_rank[k*nt+i] == dst) {
            int recv_flag = 0;
            for (int ii = k + 1; ii < i; ii++) {
              if (block_rank[ii*nt+i] == mype) recv_flag = 1;
            }
            for (int ii = i + 1; ii < nt; ii++) {
              if (block_rank[i*nt+ii] == mype) recv_flag = 1;
            }
            if (block_rank[i*nt+i] == mype) recv_flag = 1;
            if (recv_flag) {
              recv_blocks[recv_cnt++] = i;
            }
          }
        }
        //printf("send_cnt: %d, recv_cnt: %d, blocks: %d\n", send_cnt, recv_cnt, (nt-(k+1)));
          // NOTE: we have to wait for all of the above tasks using comm_sentinel
          //       dependency iterators might help here
#pragma omp task no_copy_deps firstprivate(k, dst) out({C[recv_blocks[it]], it=0;recv_cnt}) in({A[k][send_blocks[it]], it=0;send_cnt}) untied
{
        START_TIMING(TIME_COMM);
        int nreqs = 0;
        // upper bound in case all our blocks have to be sent
        int max_req = (nt-k);
        MPI_Request *reqs = malloc(sizeof(*reqs)*max_req);
        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype && np != 1) {
                int send_flag = 0;
                for (int ii = k + 1; ii < i; ii++) {
                    if (!send_flag && block_rank[ii*nt+i] == dst) {
                      send_flag = 1;
                    }
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (!send_flag && block_rank[i*nt+ii] == dst) {
                      send_flag = 1;
                    }
                }
                if (!send_flag && block_rank[i*nt+i] == dst) send_flag = 1;
                if (send_flag) {
                    MPI_Request send_req;
                    MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &send_req);
                    reqs[nreqs++] = send_req;
                }
            }
            if (block_rank[k*nt+i] != mype && block_rank[k*nt+i] == dst) {
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

        //printf("Waiting for trsm blocks from %d in k=%d\n", dst, k);
        waitall(reqs, nreqs);
        free(reqs);
        END_TIMING(TIME_COMM);
}
      }

        for (int i = k + 1; i < nt; i++) {

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#pragma omp task in(A[k][i], A[k][j]) out(A[j][i]) firstprivate(k, j, i) no_copy_deps
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#pragma omp task in(A[k][j], C[i]) out(A[j][i]) firstprivate(k, j, i) no_copy_deps
{
  						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
                      // TODO: the content of C[j] may be overwritten but we cannot specify a dependency on it :(
#pragma omp task in(A[k][i], C[j]) out(A[j][i]) firstprivate(k, j, i) no_copy_deps
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], C[j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else {
#pragma omp task in(C[i], C[j]) out(A[j][i]) firstprivate(k, j, i) no_copy_deps
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], C[j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    }
                }
            }

            if (block_rank[i*nt+i] == mype) {
                if (block_rank[k*nt+i] == mype) {
#pragma omp task in(A[k][i]) out(A[i][i]) firstprivate(k, i) no_copy_deps
{
					EXTRAE_ENTER(EVENT_SYRK);
			        START_TIMING(TIME_SYRK);
                    omp_syrk(A[k][i], A[i][i], ts, ts);
			        END_TIMING(TIME_SYRK);
					EXTRAE_EXIT(EVENT_SYRK);
}
                } else {
#pragma omp task in(C[i]) out(A[i][i]) firstprivate(k, i) no_copy_deps
{
					EXTRAE_ENTER(EVENT_SYRK);
			        START_TIMING(TIME_SYRK);
                    omp_syrk(C[i], A[i][i], ts, ts);
			        END_TIMING(TIME_SYRK);
					EXTRAE_EXIT(EVENT_SYRK);
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
  free(send_blocks);
  free(recv_blocks);
}

