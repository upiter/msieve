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

#include <stage1.h>
#include <cpu_intrinsics.h>
#include <stage1_core_gpu/stage1_core.h>

#define MAX_P ((uint32)(-1))

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 num_p_alloc;
	uint32 curr;

	uint32 k[POLY_BATCH_SIZE];

	uint32 *p;
	uint32 *mont_w;
	uint64 *mont_r;
	uint64 *p2;
	uint32 *lattice_size;
	uint64 *roots[POLY_BATCH_SIZE];
} p_soa_var_t;

static void
p_soa_var_init(p_soa_var_t *soa, uint32 batch_size)
{
	uint32 i;

	memset(soa, 0, sizeof(soa));
	soa->num_p_alloc = batch_size;
	soa->p = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->mont_w = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->mont_r = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	soa->p2 = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	soa->lattice_size = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		soa->roots[i] = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	free(soa->mont_w);
	free(soa->mont_r);
	free(soa->p2);
	free(soa->lattice_size);
	for (i = 0; i < POLY_BATCH_SIZE; i++) {
		free(soa->roots[i]);
	}
}

static void
p_soa_var_reset(p_soa_var_t *soa)
{
	uint32 i;

	soa->num_p = soa->curr = soa->p[0] = 0;
	for (i = 0; i < POLY_BATCH_SIZE; i++)
		memset(soa->roots[i], 0, soa->num_p_alloc *
							sizeof(uint64));
}

static void 
store_p_soa(uint32 p, uint32 num_roots, uint32 which_poly, 
		mpz_t *roots, void *extra)
{
	p_soa_var_t *soa = (p_soa_var_t *)extra;
	uint32 i, j, curr;
	uint64 p2 = (uint64)p * p;
	uint32 mont_w = montmul32_w((uint32)p2);
	uint64 mont_r = montmul64_r(p2);

	if (p != soa->p[soa->curr]) {
		soa->curr = soa->num_p;
		soa->num_p = MIN(soa->num_p + num_roots, soa->num_p_alloc);
	}
	curr = soa->curr;
	j = soa->num_p - curr;
	for (i = 0; i < j; i++, curr++) {
		soa->p[curr] = p;
		soa->mont_w[curr] = mont_w;
		soa->mont_r[curr] = mont_r;
		soa->p2[curr] = p2;
		soa->roots[which_poly][curr] = gmp2uint64(roots[i]);
	}
}

/*------------------------------------------------------------------------*/
static uint32
trans_batch_ad_one_sq(p_soa_var_t *orig_p_array, p_soa_var_t *trans_p_array,
			p_soa_var_t *special_q_array, uint32 which_special_q,
			uint32 start, uint32 len)
{
	uint32 i, j, k = 0, end;
	uint64 sq2 = special_q_array->p2[which_special_q];

	p_soa_var_reset(trans_p_array);
	end = start + len;
	for (i = start; i < end; i++) {
		if (special_q_array->roots[i][which_special_q] != 0)
			trans_p_array->k[k++] = i;
	}
	if (k == 0)
		return 0;

	for (i = 0; i < orig_p_array->num_p; i++) {
		uint64 p2 = orig_p_array->p2[i];
		uint32 p2_w = orig_p_array->mont_w[i];
		uint64 p2_r = orig_p_array->mont_r[i];
		uint64 inv = montmul64(mp_modinv_2(sq2, p2),
					p2_r, p2, p2_w);

		trans_p_array->p[i] = orig_p_array->p[i];
		trans_p_array->p2[i] = orig_p_array->p2[i];
		for (j = 0; j < k; j++) {
			uint32 which_root = trans_p_array->k[j];
			uint64 proot = orig_p_array->roots[which_root][i];
			uint64 sqroot =
			    special_q_array->roots[which_root][which_special_q];
			uint64 res;

			if (proot == 0)
				res = 0;
			else
				res = montmul64(mp_modsub_2(proot, sqroot % p2,
							p2), inv, p2, p2_w);

			trans_p_array->roots[j][i] = res;
		}
	}
	trans_p_array->num_p = orig_p_array->num_p;

	return k;
}

/*------------------------------------------------------------------------*/
static uint32
trans_batch_sq_one_ad(p_soa_var_t *orig_p_array, p_soa_var_t *trans_p_array,
			p_soa_var_t *special_q_array, uint32 which_ad,
			uint32 start, uint32 len)
{
	uint32 i, j, k = 0, end;

	p_soa_var_reset(trans_p_array);
	end = start + len;
	for (i = start; i < end && k < POLY_BATCH_SIZE; i++) {
		if (special_q_array->roots[which_ad][i] != 0)
			trans_p_array->k[k++] = i;
	}
	if (k == 0)
		return 0;

	for (i = 0; i < orig_p_array->num_p; i++) {
		uint64 p2 = orig_p_array->p2[i];
		uint32 p2_w = orig_p_array->mont_w[i];
		uint64 p2_r = orig_p_array->mont_r[i];
		uint64 proot = orig_p_array->roots[which_ad][i];
		uint32 num_trans = trans_p_array->num_p;

		if (proot == 0)
			continue; /* skip this hole */

		trans_p_array->p[num_trans] = orig_p_array->p[i];
		trans_p_array->p2[num_trans] = orig_p_array->p2[i];
		for (j = 0; j < k; j++) {
			uint32 which_root = trans_p_array->k[j];
			uint64 sq2 = special_q_array->p2[which_root];
			uint64 sqroot =
			    special_q_array->roots[which_ad][which_root];
			uint64 inv = montmul64(mp_modinv_2(sq2, p2),
						p2_r, p2, p2_w);
			uint64 res = montmul64(mp_modsub_2(proot, sqroot % p2,
							p2), inv, p2, p2_w);

			trans_p_array->roots[j][num_trans] = res;
		}
		trans_p_array->num_p++;
	}

	return k;
}

/*------------------------------------------------------------------------*/
#define BATCH_AD_TRIV   0
#define BATCH_AD_ONE_SQ 1
#define BATCH_SQ_ONE_AD 2

static void
check_found(lattice_fb_t *L, found_t *found_array, uint32 found_array_size,
			uint32 batch_mode, uint32 which_root)
{
	uint32 i, k;
	p_soa_var_t *p_array = (p_soa_var_t *)L->trans_p_array;
	p_soa_var_t *special_q_array = (p_soa_var_t *)L->special_q_array;

	for (i = 0; i < found_array_size; i++) {
		found_t *f = found_array + i;
		uint128 proot, res;
		uint64 p2;

		if (f->p == 0)
			continue;

		p2 = (uint64)f->p * f->p;

		proot.w[0] = (uint32)f->proot;
		proot.w[1] = (uint32)(f->proot >> 32);
		proot.w[2] = 0;
		proot.w[3] = 0;

		res = add128(proot, mul64(f->offset, p2));

		switch (batch_mode) {

		  case BATCH_AD_TRIV:
			handle_collision(L->poly, f->k, f->p, f->q, 1, 0, res);
			break;

		  case BATCH_AD_ONE_SQ:
			k = p_array->k[f->k];
			handle_collision(L->poly,
				k, f->p, f->q,
				special_q_array->p[which_root],
				special_q_array->roots[k][which_root],
				res);
			break;

		  case BATCH_SQ_ONE_AD:
			k = p_array->k[f->k];
			handle_collision(L->poly,
				which_root, f->p, f->q,
				special_q_array->p[k],
				special_q_array->roots[which_root][k],
				res);
			break;
		}
	}
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 threads_per_block, gpu_info_t *gpu_info,
			CUfunction gpu_kernel, uint32 batch_mode,
			uint32 which_root, uint32 start, uint32 len,
			uint32 *used)
{
	uint32 i, j;

	p_soa_t *p_marshall;
	q_soa_t *q_marshall;
	found_t *found_array;
	uint32 found_array_size;
	uint32 num_q_done;

	p_soa_var_t *orig_p_array = (p_soa_var_t *)L->orig_p_array;
	p_soa_var_t *trans_p_array = (p_soa_var_t *)L->trans_p_array;
	p_soa_var_t *orig_q_array = (p_soa_var_t *)L->orig_q_array;
	p_soa_var_t *trans_q_array = (p_soa_var_t *)L->trans_q_array;
	p_soa_var_t *special_q_array = (p_soa_var_t *)L->special_q_array;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	void *gpu_ptr;
	uint32 batch_size;
	*used = len;

	switch (batch_mode) {

	  default:
	  case BATCH_AD_TRIV:
		batch_size = len;
		for (j = start; j < len; j++) {
			memcpy(trans_p_array->roots[j], orig_p_array->roots[j],
			       orig_p_array->num_p * sizeof(uint64));
			memcpy(trans_q_array->roots[j], orig_q_array->roots[j],
			       orig_q_array->num_p * sizeof(uint64));
		}
		memcpy(trans_p_array->p, orig_p_array->p,
		       orig_p_array->num_p * sizeof(uint32));
		memcpy(trans_q_array->p, orig_q_array->p,
		       orig_q_array->num_p * sizeof(uint32));
		trans_p_array->num_p = orig_p_array->num_p;
		trans_q_array->num_p = orig_q_array->num_p;
		for (i = 0; i < trans_p_array->num_p; i++)
			trans_p_array->lattice_size[i] = MIN((uint32)(-1),
				(2 * L->poly->batch[len/2].sieve_size /
					orig_p_array->p2[i]));
		break;

	  case BATCH_AD_ONE_SQ:
		batch_size = trans_batch_ad_one_sq(orig_p_array, trans_p_array,
					special_q_array,
					which_root, start, len);
		trans_batch_ad_one_sq(orig_q_array, trans_q_array,
					special_q_array,
					which_root, start, len);
		if (!batch_size)
			return 0;
		for (i = 0; i < trans_p_array->num_p; i++)
			trans_p_array->lattice_size[i] = MIN((uint32)(-1),
				(2 * L->poly->batch[len/2].sieve_size /
					(special_q_array->p2[which_root] *
						(double)trans_p_array->p2[i])));
		break;
	  case BATCH_SQ_ONE_AD:
		batch_size = trans_batch_sq_one_ad(orig_p_array, trans_p_array,
					special_q_array,
					which_root, start, len);
		trans_batch_sq_one_ad(orig_q_array, trans_q_array,
					special_q_array,
					which_root, start, len);
		if (!batch_size)
			return 0;
		for (i = 0; i < trans_p_array->num_p; i++)
			trans_p_array->lattice_size[i] = MIN((uint32)(-1),
				(2 * L->poly->batch[which_root].sieve_size /
					(special_q_array->p2[start] *
						(double)trans_p_array->p2[i])));
		*used = trans_p_array->k[batch_size-1] - start + 1;
		break;
	}

	p_marshall = (p_soa_t *)L->p_marshall;
	q_marshall = (q_soa_t *)L->q_marshall;
	found_array = (found_t *)L->found_array;
	found_array_size = L->found_array_size;
	num_q_done = 0;

	i = 0;
	gpu_ptr = (void *)(size_t)L->gpu_p_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_p_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_q_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	CUDA_TRY(cuParamSeti(gpu_kernel, i, batch_size))
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	while (num_q_done < trans_q_array->num_p) {

		uint32 num_p_done = 0;
		time_t curr_time;
		double elapsed;
		uint32 curr_num_q = MIN(3 * found_array_size,
					trans_q_array->num_p - num_q_done);

		curr_num_q = MIN(curr_num_q, Q_SOA_BATCH_SIZE);

		/* force to be a multiple of the block size */
		curr_num_q -= (curr_num_q % threads_per_block);
		if (curr_num_q == 0)
			break;

		memcpy(q_marshall->p, 
			trans_q_array->p + num_q_done,
			curr_num_q * sizeof(uint32));

		for (i = 0; i < batch_size; i++) {
			memcpy(q_marshall->roots[i],
				trans_q_array->roots[i] + num_q_done,
				curr_num_q * sizeof(uint64));
		}

		CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_marshall,
				Q_SOA_BATCH_SIZE * (sizeof(uint32) +
					batch_size * sizeof(uint64))))
		CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, curr_num_q))

		while (num_p_done < trans_p_array->num_p) {

			uint32 curr_num_p = MIN(found_array_size / 3,
						trans_p_array->num_p - num_p_done);

			curr_num_p = MIN(curr_num_p, P_SOA_BATCH_SIZE);
			memcpy(p_marshall->p, 
				trans_p_array->p + num_p_done,
				curr_num_p * sizeof(uint32));
			memcpy(p_marshall->lattice_size, 
				trans_p_array->lattice_size + num_p_done,
				curr_num_p * sizeof(uint32));

			for (i = 0; i < batch_size; i++) {
				memcpy(p_marshall->roots[i],
					trans_p_array->roots[i] + num_p_done,
					curr_num_p * sizeof(uint64));
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, p_marshall,
				P_SOA_BATCH_SIZE * (2 * sizeof(uint32) +
					batch_size * sizeof(uint64))))

			CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, 
						curr_num_p))
#if 0
			printf("qnum %u pnum %u\n", curr_num_q, curr_num_p);
#endif

			num_blocks = gpu_info->num_compute_units;
			if (curr_num_q < found_array_size)
				num_blocks = curr_num_q / threads_per_block;

			CUDA_TRY(cuLaunchGrid(gpu_kernel, 
						num_blocks, 1))

			CUDA_TRY(cuMemcpyDtoH(found_array, 
						L->gpu_found_array, 
						num_blocks * 
						threads_per_block *
							sizeof(found_t)))

			check_found(L, found_array,
				num_blocks * threads_per_block,
				batch_mode, which_root);

			num_p_done += curr_num_p;
		}

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			return 1;

		curr_time = time(NULL);
		elapsed = curr_time - L->start_time;
		if (elapsed > L->deadline)
			return 1;

		num_q_done += curr_num_q;
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max,
		sieve_fb_t *sieve_small_p,
		uint32 small_p_min, uint32 small_p_max,
		sieve_fb_t *sieve_large_p,
		uint32 large_p_min, uint32 large_p_max) 
{
	uint32 i;
	uint32 quit = 0;
	p_soa_var_t *orig_p_array, *trans_p_array;
	p_soa_var_t *orig_q_array, *trans_q_array;
	p_soa_var_t *special_q_array;
	uint32 num_poly = L->poly->num_poly;
	uint32 degree = L->poly->degree;
	uint32 max_p_roots = (degree != 5) ? MAX_ROOTS : 1;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;

	uint32 threads_per_block;
	gpu_info_t *gpu_info = L->poly->gpu_info;
	CUmodule gpu_module;
       	CUfunction gpu_kernel;

	if (large_p_max < ((uint32)1 << 24))
		gpu_module = L->poly->gpu_module48;
	else
		gpu_module = L->poly->gpu_module64;

	CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
			gpu_module, "sieve_kernel"))

	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	L->q_marshall = (q_soa_t *)xmalloc(sizeof(q_soa_t));
	orig_p_array = L->orig_p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	orig_q_array = L->orig_q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	trans_p_array = L->trans_p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	trans_q_array = L->trans_q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	special_q_array = L->special_q_array = (p_soa_var_t *)xmalloc(
							sizeof(p_soa_var_t));

	CUDA_TRY(cuMemAlloc(&L->gpu_p_array, sizeof(p_soa_t)))
	CUDA_TRY(cuMemAlloc(&L->gpu_q_array, sizeof(q_soa_t)))

	CUDA_TRY(cuFuncGetAttribute((int *)&threads_per_block, 
			CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			gpu_kernel))

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel, 
				threads_per_block, 1, 1))

	L->found_array_size = threads_per_block *
				gpu_info->num_compute_units;
	L->found_array = (found_t *)xmalloc(L->found_array_size *
					sizeof(found_t));
	CUDA_TRY(cuMemAlloc(&L->gpu_found_array, 
			L->found_array_size * sizeof(found_t)))

	host_p_batch_size = MAX(10000, L->found_array_size / 3);
	host_q_batch_size = MAX(50000, 12 * L->found_array_size);
	p_soa_var_init(orig_p_array, host_p_batch_size);
	p_soa_var_init(trans_p_array, host_p_batch_size);
	p_soa_var_init(orig_q_array, host_q_batch_size);
	p_soa_var_init(trans_q_array, host_q_batch_size);
	p_soa_var_init(special_q_array, host_p_batch_size);

	sieve_fb_reset(sieve_large_p, large_p_min, large_p_max, 1, max_p_roots);
	while (!quit) {
		p_soa_var_reset(orig_q_array);
		while (sieve_fb_next(sieve_large_p, L->poly, store_p_soa,
					orig_q_array) != P_SEARCH_DONE)
			if (orig_q_array->num_p == host_q_batch_size)
				break;

		if (orig_q_array->num_p == 0)
			break;

		sieve_fb_reset(sieve_small_p, small_p_min, small_p_max,
						1, max_p_roots);
		while (!quit) {
			p_soa_var_reset(orig_p_array);
			while (sieve_fb_next(sieve_small_p, L->poly, 
						store_p_soa,
						orig_p_array) != P_SEARCH_DONE)
				if (orig_p_array->num_p == host_p_batch_size)
					break;

			if (orig_p_array->num_p == 0)
				break;

			if (special_q_min == 1) {
				/* handle trivial lattice */

				uint32 trash;
				quit = sieve_lattice_batch(obj, L,
					threads_per_block, gpu_info,
					gpu_kernel, BATCH_AD_TRIV,
					0, 0, num_poly, &trash);

				if (special_q_max == 1)
					continue;
			}

			sieve_fb_reset(sieve_special_q, special_q_min,
						special_q_max, 1, MAX_ROOTS);
			while (!quit) {
				uint32 num_sq;
				p_soa_var_reset(special_q_array);
				while (sieve_fb_next(sieve_special_q,
							L->poly, store_p_soa,
							special_q_array) !=
								P_SEARCH_DONE)
					if (special_q_array->num_p ==
							host_p_batch_size)
						break;

				num_sq = special_q_array->num_p;

				if (num_sq == 0)
					break;
				else if (num_sq < 200 && num_poly > 1) {
					/* batch many a_d with
					 * one special_q at a time */

					for (i = 0; i < num_sq && !quit; i++) {
						uint32 trash;
						quit = sieve_lattice_batch(obj,
							L, threads_per_block,
							gpu_info, gpu_kernel,
							BATCH_AD_ONE_SQ,
							i, 0, num_poly, &trash);
					}
				}
				else {
					/* batch many special_q with
					 * one a_d at a time */

					for (i = 0; i < num_poly; i++) {
						uint32 num_sq_done = 0;
						while (!quit && num_sq_done < num_sq) {
							uint32 num_used;
							quit = sieve_lattice_batch(obj,
								L, threads_per_block,
								gpu_info, gpu_kernel,
								BATCH_SQ_ONE_AD,
								i, num_sq_done,
								num_sq - num_sq_done,
								&num_used);
							num_sq_done += num_used;
						}
					}
				}
			}
		}
	}

	CUDA_TRY(cuMemFree(L->gpu_p_array))
	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	p_soa_var_free(orig_p_array);
	p_soa_var_free(orig_q_array);
	p_soa_var_free(trans_p_array);
	p_soa_var_free(trans_q_array);
	p_soa_var_free(special_q_array);
	free(orig_p_array);
	free(orig_q_array);
	free(trans_p_array);
	free(trans_q_array);
	free(special_q_array);
	free(L->p_marshall);
	free(L->q_marshall);
	free(L->found_array);
	return quit;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max)
{
	uint32 quit;
	uint32 large_p_min, large_p_max;
	uint32 small_p_min, small_p_max;
	sieve_fb_t sieve_large_p, sieve_small_p;
	curr_poly_t *middle_poly = L->poly->batch + L->poly->num_poly / 2;
	curr_poly_t *last_poly = L->poly->batch + L->poly->num_poly - 1;
	uint32 max_roots = (L->poly->degree != 5) ? L->poly->degree : 1;

	sieve_fb_init(&sieve_large_p, L->poly,
			0, 0, /* prime large_p */
			1, max_roots,
			0);

	sieve_fb_init(&sieve_small_p, L->poly,
			0, 0, /* prime small_p */
			1, max_roots,
			0);

	large_p_min = sqrt(middle_poly->p_size_max / special_q_max);
	large_p_max = MIN(MAX_P, large_p_min * params->p_scale);

	small_p_max = large_p_min - 1;
	small_p_min = small_p_max / params->p_scale;

	while (1) {
		gmp_printf("coeff %Zd-%Zd specialq %u - %u "
			   "p1 %u - %u p2 %u - %u\n",
				L->poly->batch[0].high_coeff,
				last_poly->high_coeff,
				special_q_min, special_q_max,
				small_p_min, small_p_max,
				large_p_min, large_p_max);

		quit = sieve_specialq_64(obj, L,
				sieve_special_q,
				special_q_min, special_q_max,
				&sieve_small_p,
				small_p_min, small_p_max,
				&sieve_large_p,
				large_p_min, large_p_max);

		if (quit || large_p_max == MAX_P ||
		    large_p_max / small_p_min > params->max_diverge)
			break;

		large_p_min = large_p_max + 1;
		large_p_max = MIN(MAX_P, large_p_min * params->p_scale);

		small_p_max = small_p_min - 1;
		small_p_min = small_p_max / params->p_scale;
	}

	sieve_fb_free(&sieve_large_p);
	sieve_fb_free(&sieve_small_p);
	return quit;
}
