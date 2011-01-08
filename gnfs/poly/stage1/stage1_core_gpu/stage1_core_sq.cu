/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id$
--------------------------------------------------------------------*/

#include "cuda_intrinsics.h"
#include "stage1_core_sq.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __CUDA_ARCH__ >= 200
#define SHARED_BATCH_SIZE 100
#else
#define SHARED_BATCH_SIZE 32
#endif

typedef struct {
	uint32 p[SHARED_BATCH_SIZE];
	float lsize[SPECIALQ_BATCH_SIZE][SHARED_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE][SHARED_BATCH_SIZE];
} p_soa_shared_t;

/* note that num_q in each function below must be a
   multiple of the block size (we want either all threads
   or no threads to execute the __syncthreads() calls) */

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_48(p_soa_t *pbatch, 
		uint32 num_p,
		q_soa_t *qbatch,
		uint32 num_q,
		uint32 num_roots,
		found_t *found_array)
{
	__shared__ p_soa_shared_t pbatch_cache;
	uint32 my_threadid;
	uint32 num_threads;
	uint32 i, j, k;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;
	found_array[my_threadid].p = 0;

	for (i = my_threadid; i < num_q; i += num_threads) {
		uint32 q, q2_w, p_done = 0;
		uint64 q2, q2_r;

		q = qbatch->p[i];
		q2 = wide_sqr32(q);
		q2_w = montmul24_w((uint32)q2);
		q2_r = montmul48_r(q2, q2_w);

		while (p_done < num_p) {

			uint32 curr_num_p = MIN(SHARED_BATCH_SIZE,
						num_p - p_done);

			__syncthreads();

			for (j = threadIdx.x; j < curr_num_p;
							j += blockDim.x) {

				pbatch_cache.p[j] = pbatch->p[p_done + j];

				for (k = 0; k < num_roots; k++) {

					pbatch_cache.roots[k][j] =
						pbatch->roots[k][p_done + j]; 

					pbatch_cache.lsize[k][j] =
						pbatch->lsize[k][p_done + j];
				}
			}

			__syncthreads();

			for (j = 0; j < curr_num_p; j++) {
				uint64 prefetch = qbatch->roots[0][i]; 
				uint32 p = pbatch_cache.p[j];
				uint64 p2 = wide_sqr32(p);
				uint32 pinvmodq = modinv32(p, q);

				uint64 pinv, tmp;

				tmp = wide_sqr32(pinvmodq);
				tmp = montmul48(tmp, q2_r, q2, q2_w);
				pinv = montmul48(p2, tmp, q2, q2_w);
				pinv = modsub64((uint64)2, pinv, q2);
				pinv = montmul48(pinv, tmp, q2, q2_w);
				pinv = montmul48(pinv, q2_r, q2, q2_w);

				for (k = 0; k < num_roots; k++) {

					uint64 qroot;
					uint64 proot;
					uint64 res;
					float lsize;

					qroot = prefetch;
					prefetch = qbatch->roots[k+1][i]; 
					proot = pbatch_cache.roots[k][j];
					lsize = pbatch_cache.lsize[k][j];
					res = montmul48(pinv, 
							modsub64(qroot, proot,
							q2), q2, q2_w);

					if (res < lsize) {
						found_t *f = found_array + 
								my_threadid;
						f->p = p;
						f->q = q;
						f->k = k;
						f->offset = res;
						f->proot = proot;
					}
				}
			}

			p_done += curr_num_p;
		}
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_64(p_soa_t *pbatch, 
		uint32 num_p,
		q_soa_t *qbatch,
		uint32 num_q,
		uint32 num_roots,
		found_t *found_array)
{
	__shared__ p_soa_shared_t pbatch_cache;
	uint32 my_threadid;
	uint32 num_threads;
	uint32 i, j, k;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;
	found_array[my_threadid].p = 0;

	for (i = my_threadid; i < num_q; i += num_threads) {
		uint32 q, q2_w, p_done = 0;
		uint64 q2, q2_r;

		q = qbatch->p[i];
		q2 = wide_sqr32(q);
		q2_w = montmul32_w((uint32)q2);
		q2_r = montmul64_r(q2, q2_w);

		while (p_done < num_p) {

			uint32 curr_num_p = MIN(SHARED_BATCH_SIZE,
						num_p - p_done);

			__syncthreads();

			for (j = threadIdx.x; j < curr_num_p;
							j += blockDim.x) {

				pbatch_cache.p[j] = pbatch->p[p_done + j];

				for (k = 0; k < num_roots; k++) {

					pbatch_cache.roots[k][j] =
						pbatch->roots[k][p_done + j]; 

					pbatch_cache.lsize[k][j] =
						pbatch->lsize[k][p_done + j];
				}
			}

			__syncthreads();

			for (j = 0; j < curr_num_p; j++) {
				uint64 prefetch = qbatch->roots[0][i]; 
				uint32 p = pbatch_cache.p[j];
				uint64 p2 = wide_sqr32(p);
				uint32 pinvmodq = modinv32(p, q);

				uint64 pinv, tmp;

				tmp = wide_sqr32(pinvmodq);
				tmp = montmul64(tmp, q2_r, q2, q2_w);
				pinv = montmul64(p2, tmp, q2, q2_w);
				pinv = modsub64((uint64)2, pinv, q2);
				pinv = montmul64(pinv, tmp, q2, q2_w);
				pinv = montmul64(pinv, q2_r, q2, q2_w);

				for (k = 0; k < num_roots; k++) {

					uint64 qroot;
					uint64 proot;
					uint64 res;
					float lsize;

					qroot = prefetch;
					prefetch = qbatch->roots[k+1][i]; 
					proot = pbatch_cache.roots[k][j];
					lsize = pbatch_cache.lsize[k][j];
					res = montmul64(pinv, 
							modsub64(qroot, proot,
							q2), q2, q2_w);

					if (res < lsize) {
						found_t *f = found_array + 
								my_threadid;
						f->p = p;
						f->q = q;
						f->k = k;
						f->offset = res;
						f->proot = proot;
					}
				}
			}

			p_done += curr_num_p;
		}
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 sq[SPECIALQ_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE];
} sq_soa_shared_t;

__global__ void
trans_batch_sq(q_soa_t *qbatch,
		uint32 num_q,
		p_soa_t *sqbatch,
		uint32 num_sq,
		float sieve_size)
{
	__shared__ sq_soa_shared_t sqbatch_cache;
	uint32 my_threadid;
	uint32 num_threads;
	uint32 i, j;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;

	for (i = my_threadid; i < num_q; i+= num_threads) {
		uint32 q, q2_w;
		uint64 q2, q2_r;
		uint64 qroot;
		float tmp_lsize;

		q = qbatch->p[i];
		q2 = wide_sqr32(q);
		q2_w = montmul32_w((uint32)q2);
		q2_r = montmul64_r(q2, q2_w);
		qroot = qbatch->start_root[i];
		tmp_lsize = sieve_size / q2;

		__syncthreads();

		for (j = threadIdx.x; j < num_sq; j += blockDim.x) {

			sqbatch_cache.sq[j] = sqbatch->p[j];
			sqbatch_cache.roots[j] =
				sqbatch->start_root[j];
		}

		__syncthreads();

		for (j = 0; j < num_sq; j++) {
			uint32 sq = sqbatch_cache.sq[j];
			uint64 sq2 = wide_sqr32(sq);
			uint32 sqinvmodq = modinv32(sq % q, q);

			uint64 sqinv, tmp;
			uint64 sqroot, res;

			tmp = wide_sqr32(sqinvmodq);
			tmp = montmul64(tmp, q2_r, q2, q2_w);
			sqinv = montmul64(sq2, tmp, q2, q2_w);
			sqinv = modsub64((uint64)2, sqinv, q2);
			sqinv = montmul64(sqinv, tmp, q2, q2_w);
			sqinv = montmul64(sqinv, q2_r, q2, q2_w);

			sqroot = sqbatch_cache.roots[j];
			res = montmul64(sqinv,
					modsub64(qroot, 
					sqroot % q2,
					q2), q2, q2_w);

			qbatch->roots[j][i] = res;
			qbatch->lsize[j][i] = tmp_lsize / sq2;
		}
	}
}

#ifdef __cplusplus
}
#endif
