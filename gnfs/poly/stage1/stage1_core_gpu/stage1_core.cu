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

#include "stage1_core.h"

#ifdef __cplusplus
extern "C" {
#endif

__constant__ specialq_t q_batch[BATCH_SPECIALQ_MAX];

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_trans(uint32 *p_array, uint32 num_p, uint64 *start_roots,
			uint32 num_roots, uint32 *p_out, int64 *roots_out,
			uint32 num_specialq, uint32 num_entries)
{
	uint32 offset, i, j, p, pp_w, q, end, gcd;
	uint64 pp, pp_r, qq, tmp, inv;
	int64 newroot;

	offset = blockIdx.x * blockDim.x + threadIdx.x;
	if (offset >= num_p)
		return;

	p = p_array[offset];
	pp = wide_sqr32(p);
	pp_w = montmul32_w((uint32)pp);
	pp_r = montmul64_r(pp, pp_w);
	end = num_p * num_roots;

	q = 0;
	for (i = 0; i < num_specialq; i++) {
		if (q != q_batch[i].p) {
			q = q_batch[i].p;
			gcd = gcd32(p, q);

			if (gcd == 1) {
				qq = wide_sqr32(q) % pp;
				tmp = modinv32(q % p, p);
				tmp = wide_sqr32(tmp);
				tmp = montmul64(tmp, pp_r, pp, pp_w);
				inv = montmul64(qq, tmp, pp, pp_w);
				inv = modsub64((uint64)2, inv, pp);
				inv = montmul64(inv, tmp, pp, pp_w);
				inv = montmul64(inv, pp_r, pp, pp_w);
			}
		}

		for (j = offset; j < end; j += num_p) {

			if (gcd == 1) {
				newroot = modsub64(start_roots[j],
						q_batch[i].root % pp, pp);
				newroot = montmul64(newroot, inv, pp, pp_w);

				if (newroot > pp / 2)
					newroot -= pp;

				p_out[j + num_entries * i] = p;
				roots_out[j + num_entries * i] = newroot;
			}
		}
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_sort(uint32 *p_array, uint64 *roots)
{
	uint32 my_threadid, offset, j, k, u, dir, tmp;
	extern __shared__ char shared_cache[];
	uint32 *p_cache;
	uint64 *root_cache;
	uint64 root_1, root_2;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = 2 * (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x +
							threadIdx.x;

	p_cache = (uint32 *)shared_cache;
	root_cache = (uint64 *)(p_cache + blockDim.x * 2);

	p_cache[threadIdx.x] = p_array[offset];
	p_cache[threadIdx.x + blockDim.x] = p_array[offset + blockDim.x];

	root_cache[threadIdx.x] = roots[offset];
	root_cache[threadIdx.x + blockDim.x] = roots[offset + blockDim.x];

	__syncthreads();

	for (j = 1; j <= blockDim.x; j *= 2) {

		dir = !!(my_threadid & j);
		for (k = j; k; k /= 2) {

			u = (threadIdx.x & ~(k - 1)) * 2 +
						(threadIdx.x & (k - 1));

			root_1 = root_cache[u];
			root_2 = root_cache[u + k];

			if (dir != (root_1 > root_2)) {

				tmp = p_cache[u];
				p_cache[u] = p_cache[u + k];
				p_cache[u + k] = tmp;

				root_cache[u] = root_2;
				root_cache[u + k] = root_1;
			}

			__syncthreads();
		}
	}

	p_array[offset] = p_cache[threadIdx.x];
	p_array[offset + blockDim.x] = p_cache[threadIdx.x + blockDim.x];

	roots[offset] = root_cache[threadIdx.x];
	roots[offset + blockDim.x] = root_cache[threadIdx.x + blockDim.x];
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_merge(uint32 *p_array, uint64 *roots, uint32 j)
{
	uint32 my_threadid, offset, k, u, dir, tmp;
	extern __shared__ char shared_cache[];
	uint32 *p_cache;
	uint64 *root_cache;
	uint64 root_1, root_2;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = 2 * (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x +
							threadIdx.x;

	p_cache = (uint32 *)shared_cache;
	root_cache = (uint64 *)(p_cache + blockDim.x * 2);

	p_cache[threadIdx.x] = p_array[offset];
	p_cache[threadIdx.x + blockDim.x] = p_array[offset + blockDim.x];

	root_cache[threadIdx.x] = roots[offset];
	root_cache[threadIdx.x + blockDim.x] = roots[offset + blockDim.x];

	__syncthreads();

	dir = !!(my_threadid & j);
	for (k = blockDim.x; k; k /= 2) {

		u = (threadIdx.x & ~(k - 1)) * 2 +
					(threadIdx.x & (k - 1));

		root_1 = root_cache[u];
		root_2 = root_cache[u + k];

		if (dir != (root_1 > root_2)) {

			tmp = p_cache[u];
			p_cache[u] = p_cache[u + k];
			p_cache[u + k] = tmp;

			root_cache[u] = root_2;
			root_cache[u + k] = root_1;
		}

		__syncthreads();
	}

	p_array[offset] = p_cache[threadIdx.x];
	p_array[offset + blockDim.x] = p_cache[threadIdx.x + blockDim.x];

	roots[offset] = root_cache[threadIdx.x];
	roots[offset + blockDim.x] = root_cache[threadIdx.x + blockDim.x];
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_merge1(uint32 *p_array, uint64 *roots, uint32 j, uint32 k)
{
	uint32 my_threadid, offset, tmp;
	uint64 root_1, root_2;

	my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	offset = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x +
							threadIdx.x;
	offset = (offset & ~(k - 1)) * 2 + (offset & (k - 1));

	root_1 = roots[offset];
	root_2 = roots[offset + k];

	if ((!!(my_threadid & j)) != (root_1 > root_2)) {

		tmp = p_array[offset];
		p_array[offset] = p_array[offset + k];
		p_array[offset + k] = tmp;

		roots[offset] = root_2;
		roots[offset + k] = root_1;
	}
}

/*------------------------------------------------------------------------*/
__global__ void
sieve_kernel_final(uint32 *p_array, int64 *roots, uint32 num_entries,
			uint32 num_specialq, found_t *found_array)
{
	uint32 i, my_threadid, num_threads, p_1, p_2;
	int64 root_1, root_2;

	i = my_threadid = blockIdx.x * blockDim.x + threadIdx.x;
	num_threads = gridDim.x * blockDim.x;

	while (i < num_entries * num_specialq - 1) {
		p_1 = p_array[i];
		p_2 = p_array[i + 1];
		root_1 = roots[i];
		root_2 = roots[i + 1];

		if (p_1 > 0 && p_2 > 0 && root_1 == root_2) {

			if (gcd32(p_1, p_2) == 1) {
				found_t *f = found_array + my_threadid;

				f->p1 = p_1;
				f->p2 = p_2;
				f->q = q_batch[i / num_entries].p;
				f->qroot = q_batch[i / num_entries].root;
				f->offset = root_1;
			}
		}

		i += num_threads;
	}
}

#ifdef __cplusplus
}
#endif
