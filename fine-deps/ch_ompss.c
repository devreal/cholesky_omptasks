
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "ch_common.h"
#include "../extrae.h"
#include "../timing.h"

//#ifdef _OMPSS
//#warning "Compiling for OMPSS"
//#endif


//TODO: adjust wait() for timing
static int depth;
#pragma omp threadprivate(depth)

static int comm_round_sentinel; // <-- used to limit parallel communication tasks

void cholesky_mpi(const int ts, const int nt, double *A[nt][nt], double *B, double *C[nt], int *block_rank)
{
	REGISTER_EXTRAE();
#pragma omp parallel
{
    depth = 0;
#pragma omp single
{
	INIT_TIMING(omp_get_num_threads());
    char *send_flags = malloc(sizeof(char) * np);
    char recv_flag = 0;
    int num_send_tasks = 0;
    int num_recv_tasks = 0;
    int max_send_tasks = 0;
    int max_recv_tasks = 0;
    int num_comp_tasks = 0;
    reset_send_flags(send_flags);

    START_TIMING(TIME_TOTAL);
    {
    START_TIMING(TIME_CREATE);
    for (int k = 0; k < nt; k++) {
        int send_tasks = 0, recv_tasks = 0;
        // sentinel task to limit communication task parallelism
#ifdef HAVE_COMM_SENTINEL
#pragma omp task depend(out: comm_round_sentinel)
        { if (comm_round_sentinel < 0) comm_round_sentinel = 0; }
#endif // HAVE_COMM_SENTINEL
        if (block_rank[k*nt+k] == mype) {
            num_comp_tasks++;
#pragma omp task depend(out: A[k][k]) firstprivate(k)
{
			EXTRAE_ENTER(EVENT_POTRF);
			START_TIMING(TIME_POTRF);
            omp_potrf(A[k][k], ts, ts);
			END_TIMING(TIME_POTRF);
			EXTRAE_EXIT(EVENT_POTRF);
}
        }

        if (block_rank[k*nt+k] == mype && np != 1) {
#pragma omp task depend(in: A[k][k]) firstprivate(k) depend(in: comm_round_sentinel) untied
{
            START_TIMING(TIME_COMM);
            MPI_Request reqs[np];
            int nreqs = 0;
            //printf("[%d:%d:%d] Sending k=%d block (tag %d)\n", mype, omp_get_thread_num(), depth, k, k*nt+k);
            for (int dst = 0; dst < np; dst++) {
                int send_flag = 0;
                for (int kk = k+1; kk < nt; kk++) {
                   if (dst == block_rank[k*nt+kk]) { send_flag = 1; break; }
                }
                if (send_flag && dst != mype) {
 		    depth++;
                    MPI_Request send_req;
                    MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &send_req);
                    //wait(&send_req);
                    reqs[nreqs++] = send_req;
 		    depth--;
                }
            }
            for (int i = 0; i < nreqs; ++i) {
              wait(&reqs[i]);
            }
			END_TIMING(TIME_COMM);
            //printf("[%d:%d:%d] Done Sending k=%d block (tag %d)\n", mype, omp_get_thread_num(), depth, k, k*nt+k);
}
            reset_send_flags(send_flags);
        }

        if (block_rank[k*nt+k] != mype) {
            for (int i = k + 1; i < nt; i++) {
                if (block_rank[k*nt+i] == mype) recv_flag = 1;
            }
            if (recv_flag) {
#pragma omp task depend(out: B) firstprivate(k) depend(in: comm_round_sentinel) untied
{
		//printf("[%d:%d:%d] Receiving k=%d block from %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, block_rank[k*nt+k], k*nt+k);
            START_TIMING(TIME_COMM);
 		    depth++;
                MPI_Request recv_req;
                MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD, &recv_req);
                wait(&recv_req);
 		    depth--;
			END_TIMING(TIME_COMM);
		//printf("[%d:%d:%d] Done Receiving k=%d block from %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, block_rank[k*nt+k], k*nt+k);
}
                recv_flag = 0;
            }
        }

#ifdef HAVE_INTERMEDIATE_COMM_SENTINEL
        // sentinel task to limit communication task parallelism
#pragma omp task depend(out: comm_round_sentinel)
        { if (comm_round_sentinel < 0) comm_round_sentinel = 0; }
#endif

        for (int i = k + 1; i < nt; i++) {
            if (block_rank[k*nt+i] == mype) {
                num_comp_tasks++;
                if (block_rank[k*nt+k] == mype) {
#pragma omp task depend(in: A[k][k]) depend(out: A[k][i]) firstprivate(k, i)
{
					EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(A[k][k], A[k][i], ts, ts);
			        END_TIMING(TIME_TRSM);
					EXTRAE_EXIT(EVENT_TRSM);
}
                } else {
#pragma omp task depend(in: B) depend(out: A[k][i]) firstprivate(k, i)
{
                    EXTRAE_ENTER(EVENT_TRSM);
			        START_TIMING(TIME_TRSM);
                    omp_trsm(B, A[k][i], ts, ts);
                    END_TIMING(TIME_TRSM);
                    EXTRAE_EXIT(EVENT_TRSM);
}
                }
            }

            if (block_rank[k*nt+i] == mype && np != 1) {
                for (int ii = k + 1; ii < i; ii++) {
                    if (!send_flags[block_rank[ii*nt+i]]) send_flags[block_rank[ii*nt+i]] = 1;
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (!send_flags[block_rank[i*nt+ii]]) send_flags[block_rank[i*nt+ii]] = 1;
                }
                if (!send_flags[block_rank[i*nt+i]]) send_flags[block_rank[i*nt+i]] = 1;
                for (int dst = 0; dst < np; dst++) {
                    if (send_flags[dst] && dst != mype) {
                       send_tasks++;
                       num_send_tasks++;
#pragma omp task depend(in: A[k][i]) firstprivate(k, i, dst) depend(in: comm_round_sentinel) untied
{
		        //printf("[%d:%d:%d] Sending k=%d i=%d block to %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, i, dst, k*nt+i);
            START_TIMING(TIME_COMM);
 		    depth++;
                        MPI_Request send_req;
                        MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &send_req);
                        wait(&send_req);
 		    depth--;
			END_TIMING(TIME_COMM);
		        //printf("[%d:%d:%d] Done Sending k=%d i=%d block to %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, i, dst, k*nt+i);
}
                    }
                }
                reset_send_flags(send_flags);
            }
            if (block_rank[k*nt+i] != mype) {
                for (int ii = k + 1; ii < i; ii++) {
                    if (block_rank[ii*nt+i] == mype) recv_flag = 1;
                }
                for (int ii = i + 1; ii < nt; ii++) {
                    if (block_rank[i*nt+ii] == mype) recv_flag = 1;
                }
                if (block_rank[i*nt+i] == mype) recv_flag = 1;
                if (recv_flag) {
                    recv_tasks++;
                    num_recv_tasks++;
#pragma omp task depend(out: C[i]) firstprivate(k, i) depend(in: comm_round_sentinel) untied
{
		    //printf("[%d:%d:%d] Receiving k=%d i=%d block from %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, i, block_rank[k*nt+i], k*nt+i);
            START_TIMING(TIME_COMM);
 		    depth++;
                    MPI_Request recv_req;
                    MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &recv_req);
                    wait(&recv_req);
 		    depth--;
			END_TIMING(TIME_COMM);
		    //printf("[%d:%d:%d] Done Receiving k=%d i=%d block from %d (tag %d)\n", mype, omp_get_thread_num(), depth, k, i, block_rank[k*nt+i], k*nt+i);
}
                    recv_flag = 0;
                }
            }
        }

        if ((max_send_tasks + max_recv_tasks) < (send_tasks + recv_tasks)) {
          max_send_tasks = send_tasks;
          max_recv_tasks = recv_tasks;
        }

        for (int i = k + 1; i < nt; i++) {

            for (int j = k + 1; j < i; j++) {
                if (block_rank[j*nt+i] == mype) {
                    num_comp_tasks++;
                    if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype) {
#pragma omp task depend(in: C[i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i)
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(C[i], A[k][j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype) {
#pragma omp task depend(in: A[k][i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
{
						EXTRAE_ENTER(EVENT_GEMM);
			            START_TIMING(TIME_GEMM);
                        omp_gemm(A[k][i], C[j], A[j][i], ts, ts);
			            END_TIMING(TIME_GEMM);
						EXTRAE_EXIT(EVENT_GEMM);
}
                    } else {
#pragma omp task depend(in: C[i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i)
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
                num_comp_tasks++;
                if (block_rank[k*nt+i] == mype) {
#pragma omp task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i)
{
					EXTRAE_ENTER(EVENT_SYRK);
			        START_TIMING(TIME_SYRK);
                    omp_syrk(A[k][i], A[i][i], ts, ts);
			        END_TIMING(TIME_SYRK);
					EXTRAE_EXIT(EVENT_SYRK);
}
                } else {
#pragma omp task depend(in: C[i]) depend(out: A[i][i]) firstprivate(k, i)
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
#ifdef USE_TIMING
	PRINT_TIMINGS();
	FREE_TIMING();
#endif 
    printf("[%d] max_send_tasks %d, max_recv_tasks %d, num_send_tasks %d, num_recv_tasks %d, num_comp_tasks %d\n", 
           mype, max_send_tasks, max_recv_tasks, num_send_tasks, num_recv_tasks, num_comp_tasks);

    free(send_flags);

}// pragma omp single
}// pragma omp parallel
}

