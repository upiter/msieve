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
#include <stage1_core_gpu/stage1_core_sq.h>

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 num_p_alloc;

	uint32 *p;
	uint64 *p2;
	uint64 *start_root;
	float *lattice_size[SPECIALQ_BATCH_SIZE];
	uint64 *roots[SPECIALQ_BATCH_SIZE];
} p_soa_var_t;

static void
p_soa_var_init(p_soa_var_t *soa, uint32 batch_size)
{
	uint32 i;

	memset(soa, 0, sizeof(soa));
	soa->num_p_alloc = batch_size;
	soa->p = (uint32 *)xmalloc(batch_size * sizeof(uint32));
	soa->p2 = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	soa->start_root = (uint64 *)xmalloc(batch_size * sizeof(uint64));
	for (i = 0; i < SPECIALQ_BATCH_SIZE; i++) {
		soa->roots[i] =
			(uint64 *)xmalloc(batch_size * sizeof(uint64));
		soa->lattice_size[i] =
			(float *)xmalloc(batch_size * sizeof(float));
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	free(soa->p2);
	free(soa->start_root);
	for (i = 0; i < SPECIALQ_BATCH_SIZE; i++) {
		free(soa->roots[i]);
		free(soa->lattice_size[i]);
	}
}

static void
p_soa_var_reset(p_soa_var_t *soa)
{
	soa->num_p = 0;
}

static void 
store_p_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	p_soa_var_t *soa = (p_soa_var_t *)extra;
	uint32 i;
	uint64 p2 = (uint64)p * p;

	for (i = 0; i < num_roots && soa->num_p < soa->num_p_alloc; i++) {
		soa->p[soa->num_p] = p;
		soa->p2[soa->num_p] = p2;
		soa->start_root[soa->num_p] = roots[i];

		soa->num_p++;
	}
}

/*------------------------------------------------------------------------*/
static void
check_found(lattice_fb_t *L, found_t *found_array, uint32 found_array_size)
{
	uint32 i;
	p_soa_var_t *sq_array = (p_soa_var_t *)L->sq_array;

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

		handle_collision(L->poly, f->p, f->q,
				sq_array->p[f->k],
				sq_array->start_root[f->k],
				res);
	}
}

/*------------------------------------------------------------------------*/
static void
trans_batch_sq(p_soa_var_t *q_array, lattice_fb_t *L,
			uint32 threads_per_block, gpu_info_t *gpu_info,
			CUfunction gpu_kernel)
{
	uint32 i;

	p_soa_t *sq_marshall;
	q_soa_t *q_marshall;
	uint32 trans_size;
	uint32 num_q_done;
	p_soa_var_t *sq_array = (p_soa_var_t *)L->sq_array;
	uint32 num_blocks;
	uint32 num_q_offset;
	void *gpu_ptr;

	sq_marshall = (p_soa_t *)L->p_marshall;
	q_marshall = (q_soa_t *)L->q_marshall;
	num_q_done = 0;

	trans_size = threads_per_block * gpu_info->num_compute_units;

	i = 0;
	gpu_ptr = (void *)(size_t)L->gpu_q_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_p_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	CUDA_TRY(cuParamSeti(gpu_kernel, (int)i,
			sq_array->num_p))
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(float));
	CUDA_TRY(cuParamSetf(gpu_kernel, (int)i,
			2 * L->poly->sieve_size))
	i += sizeof(float);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	memcpy(sq_marshall->p, sq_array->p,
		sq_array->num_p * sizeof(uint32));

	memcpy(sq_marshall->start_root, sq_array->start_root,
		sq_array->num_p * sizeof(uint64));

	CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, sq_marshall,
			sizeof(p_soa_t)))

	while (num_q_done < q_array->num_p) {

		uint32 curr_num_q = MIN(3 * trans_size,
					q_array->num_p - num_q_done);

		curr_num_q = MIN(curr_num_q, Q_SOA_BATCH_SIZE);

		/* force to be a multiple of the block size */
		curr_num_q -= (curr_num_q % threads_per_block);
		if (curr_num_q == 0)
			break;

		memcpy(q_marshall->p, 
			q_array->p + num_q_done,
			curr_num_q * sizeof(uint32));

		memcpy(q_marshall->start_root, 
			q_array->start_root + num_q_done,
			curr_num_q * sizeof(uint64));

		CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_marshall,
				Q_SOA_BATCH_SIZE * (sizeof(uint32) + 
					sizeof(uint64))))
		CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, curr_num_q))

		num_blocks = gpu_info->num_compute_units;
		if (curr_num_q < trans_size)
			num_blocks = curr_num_q /
				threads_per_block;

		CUDA_TRY(cuLaunchGrid(gpu_kernel, 
				num_blocks, 1))

		CUDA_TRY(cuMemcpyDtoH(q_marshall, L->gpu_q_array,
				sizeof(q_soa_t)))

		for (i = 0; i < sq_array->num_p; i++) {
			memcpy(q_array->lattice_size[i] + num_q_done,
				q_marshall->lsize[i],
				curr_num_q * sizeof(float));

			memcpy(q_array->roots[i] + num_q_done,
				q_marshall->roots[i],
				curr_num_q * sizeof(uint64));
		}

		num_q_done += curr_num_q;
	}

	q_array->num_p = num_q_done;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 threads_per_block, gpu_info_t *gpu_info,
			CUfunction gpu_kernel)
{
	uint32 i;

	p_soa_t *p_marshall;
	q_soa_t *q_marshall;
	found_t *found_array;
	uint32 found_array_size;
	uint32 num_q_done;
	p_soa_var_t *p_array = (p_soa_var_t *)L->p_array;
	p_soa_var_t *q_array = (p_soa_var_t *)L->q_array;
	p_soa_var_t *sq_array = (p_soa_var_t *)L->sq_array;
	uint32 num_roots = sq_array->num_p;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	void *gpu_ptr;

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
	CUDA_TRY(cuParamSeti(gpu_kernel, (int)i, num_roots))
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))

	while (num_q_done < q_array->num_p) {

		uint32 num_p_done = 0;
		time_t curr_time;
		double elapsed;
		uint32 curr_num_q = MIN(3 * found_array_size,
					q_array->num_p - num_q_done);

		curr_num_q = MIN(curr_num_q, Q_SOA_BATCH_SIZE);

		/* force to be a multiple of the block size */
		curr_num_q -= (curr_num_q % threads_per_block);
		if (curr_num_q == 0)
			break;

		memcpy(q_marshall->p, 
			q_array->p + num_q_done,
			curr_num_q * sizeof(uint32));

		for (i = 0; i < num_roots; i++) {
			memcpy(q_marshall->roots[i],
				q_array->roots[i] + num_q_done,
				curr_num_q * sizeof(uint64));
		}

		CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_marshall,
				Q_SOA_BATCH_SIZE * (sizeof(uint32) +
					sizeof(uint64) +
					num_roots * sizeof(uint64))))
		CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, curr_num_q))

		num_blocks = gpu_info->num_compute_units;
		if (curr_num_q < found_array_size)
			num_blocks = curr_num_q /
				threads_per_block;

		while (num_p_done < p_array->num_p) {

			uint32 curr_num_p = MIN(found_array_size / 3,
						p_array->num_p - num_p_done);

			curr_num_p = MIN(curr_num_p, P_SOA_BATCH_SIZE);

			memcpy(p_marshall->p, 
				p_array->p + num_p_done,
				curr_num_p * sizeof(uint32));

			for (i = 0; i < num_roots; i++) {
				memcpy(p_marshall->lsize[i],
					p_array->lattice_size[i] + num_p_done,
					curr_num_p * sizeof(float));

				memcpy(p_marshall->roots[i],
					p_array->roots[i] + num_p_done,
					curr_num_p * sizeof(uint64));
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, p_marshall,
					sizeof(p_soa_t)))

			CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, 
						curr_num_p))

#if 0
			printf("qnum %u pnum %u sqnum %u\n",
				curr_num_q, curr_num_p, num_roots);
#endif

			CUDA_TRY(cuLaunchGrid(gpu_kernel, 
					num_blocks, 1))

			CUDA_TRY(cuMemcpyDtoH(found_array, 
					L->gpu_found_array, 
					num_blocks * 
					threads_per_block *
						sizeof(found_t)))

			check_found(L, found_array,
				num_blocks * threads_per_block);

			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				return 1;

			num_p_done += curr_num_p;

			curr_time = time(NULL);
			elapsed = curr_time - L->start_time;
			if (elapsed > L->deadline)
				return 1;
		}

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
	uint32 quit = 0;
	uint32 threads_per_block_sieve;
	uint32 threads_per_block_trans;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;
	p_soa_var_t *p_array;
	p_soa_var_t *q_array;
	p_soa_var_t *sq_array;
	gpu_info_t *gpu_info = L->poly->gpu_info;
	CUmodule gpu_module = L->poly->gpu_module_sq;
	CUfunction gpu_kernel_sieve;
	CUfunction gpu_kernel_trans;

	if (large_p_max < ((uint32)1 << 24))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel_sieve, 
				gpu_module, "sieve_kernel_48"))
	else
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel_sieve, 
				gpu_module, "sieve_kernel_64"))

	CUDA_TRY(cuModuleGetFunction(&gpu_kernel_trans, 
			gpu_module, "trans_batch_sq"))

	L->p_marshall = (p_soa_t *)xmalloc(sizeof(p_soa_t));
	L->q_marshall = (q_soa_t *)xmalloc(sizeof(q_soa_t));
	p_array = L->p_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	q_array = L->q_array = (p_soa_var_t *)xmalloc(
					sizeof(p_soa_var_t));
	sq_array = L->sq_array = (p_soa_var_t *)xmalloc(sizeof(p_soa_var_t));

	CUDA_TRY(cuMemAlloc(&L->gpu_p_array, sizeof(p_soa_t)))
	CUDA_TRY(cuMemAlloc(&L->gpu_q_array, sizeof(q_soa_t)))

	CUDA_TRY(cuFuncGetAttribute((int *)&threads_per_block_sieve, 
			CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			gpu_kernel_sieve))

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel_sieve, 
				threads_per_block_sieve, 1, 1))

	CUDA_TRY(cuFuncGetAttribute((int *)&threads_per_block_trans, 
			CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
			gpu_kernel_trans))

	CUDA_TRY(cuFuncSetBlockShape(gpu_kernel_trans, 
				threads_per_block_trans, 1, 1))

	L->found_array_size = threads_per_block_sieve *
				gpu_info->num_compute_units;
	L->found_array = (found_t *)xmalloc(L->found_array_size *
					sizeof(found_t));
	CUDA_TRY(cuMemAlloc(&L->gpu_found_array, 
			L->found_array_size * sizeof(found_t)))

	host_q_batch_size = MAX(50000, 12 * L->found_array_size);
	host_p_batch_size = host_q_batch_size;

	p_soa_var_init(q_array, host_q_batch_size);
	p_soa_var_init(p_array, host_p_batch_size);
	p_soa_var_init(sq_array, SPECIALQ_BATCH_SIZE);

	sieve_fb_reset(sieve_large_p, large_p_min, large_p_max, 1, MAX_ROOTS);
	while (!quit) {

		p_soa_var_reset(q_array);
		while (sieve_fb_next(sieve_large_p, L->poly, store_p_soa,
					q_array) != P_SEARCH_DONE)
			if (q_array->num_p == host_q_batch_size)
				break;

		if (q_array->num_p == 0)
			break;

		sieve_fb_reset(sieve_small_p, small_p_min, small_p_max,
						1, MAX_ROOTS);
		while (!quit) {

			p_soa_var_reset(p_array);
			while (sieve_fb_next(sieve_small_p, L->poly, 
						store_p_soa,
						p_array) != P_SEARCH_DONE)
				if (p_array->num_p == host_p_batch_size)
					break;

			if (p_array->num_p == 0)
				break;

			sieve_fb_reset(sieve_special_q, special_q_min,
						special_q_max, 1, MAX_ROOTS);
			while (!quit) {

				p_soa_var_reset(sq_array);
				while (sieve_fb_next(sieve_special_q,
							L->poly, store_p_soa,
							sq_array) !=
								P_SEARCH_DONE)
					if (sq_array->num_p ==
							SPECIALQ_BATCH_SIZE)
						break;

				if (sq_array->num_p == 0)
					break;

				trans_batch_sq(q_array, L,
						threads_per_block_trans,
						gpu_info, gpu_kernel_trans);
				trans_batch_sq(p_array, L,
						threads_per_block_trans,
						gpu_info, gpu_kernel_trans);

				quit = sieve_lattice_batch(obj,
						L, threads_per_block_sieve,
						gpu_info, gpu_kernel_sieve);
			}
		}
	}

	CUDA_TRY(cuMemFree(L->gpu_p_array))
	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	p_soa_var_free(p_array);
	p_soa_var_free(q_array);
	p_soa_var_free(sq_array);
	free(p_array);
	free(q_array);
	free(sq_array);
	free(L->p_marshall);
	free(L->q_marshall);
	free(L->found_array);
	return quit;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu_sq(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max)
{
	uint32 i, quit;
	uint32 large_p_min, large_p_max;
	uint32 small_p_min, small_p_max;
	uint32 degree = L->poly->degree;
	double p_size_max = L->poly->p_size_max;
	sieve_fb_t sieve_large_p, sieve_small_p;

	p_size_max /= special_q_max;
	if ((uint32)sqrt(p_size_max) * P_SCALE > (uint32)(-1)) {
		printf("error: invalid parameters for rational coefficient "
			"in sieve_lattice_gpu_sq()\n");
		return 0;
	}

	large_p_min = sqrt(p_size_max);
	large_p_max = large_p_min * P_SCALE;
	small_p_max = large_p_min - 1;
	small_p_min = small_p_max / P_SCALE;

	sieve_fb_init(&sieve_large_p, L->poly,
			0, 0, /* prime large_p */
			1, degree,
			0);

	sieve_fb_init(&sieve_small_p, L->poly,
			0, 0, /* prime small_p */
			1, degree,
			0);

	for (i = 0; i < 3; i++) {
		gmp_printf("coeff %Zd specialq %u - %u "
			   "p1 %u - %u p2 %u - %u\n",
				L->poly->high_coeff,
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

		if (quit || large_p_max > (uint32)(-1) / P_SCALE)
			break;

		large_p_min = large_p_max;
		large_p_max = large_p_min * P_SCALE;
		small_p_max = small_p_min;
		small_p_min = small_p_max / P_SCALE;
	}

	sieve_fb_free(&sieve_large_p);
	sieve_fb_free(&sieve_small_p);
	return quit;
}
