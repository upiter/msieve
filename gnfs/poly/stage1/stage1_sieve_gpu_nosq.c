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
#include <stage1_core_gpu/stage1_core_nosq.h>

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	uint32 *p;
	uint64 *roots[MAX_ROOTS];
} q_soa_var_t;

#define MAX_P_SOA_ARRAYS 10

typedef struct {
	uint32 num_arrays;
	uint32 num_p;
	q_soa_var_t soa[MAX_P_SOA_ARRAYS];
} q_soa_array_t;

static void
q_soa_array_init(q_soa_array_t *s, uint32 degree)
{
	uint32 i, j;
	memset(s, 0, sizeof(q_soa_array_t));

	switch (degree) {
	case 4:
		s->num_arrays = 3;
		s->soa[0].num_roots = 16;
		s->soa[1].num_roots = 8;
		s->soa[2].num_roots = 4;
		break;
	case 5:
		s->num_arrays = 2;
		s->soa[0].num_roots = 25;
		s->soa[1].num_roots = 5;
		break;
	case 6:
		s->num_arrays = 10;
		s->soa[9].num_roots = 6;
		s->soa[8].num_roots = 8;
		s->soa[7].num_roots = 12;
		s->soa[6].num_roots = 16;
		s->soa[5].num_roots = 24;
		s->soa[4].num_roots = 32;
		s->soa[3].num_roots = 36;
		s->soa[2].num_roots = 48;
		s->soa[1].num_roots = 64;
		s->soa[0].num_roots = 72;
		break;
	}

	for (i = 0; i < s->num_arrays; i++) {
		q_soa_var_t *soa = s->soa + i;

		soa->num_p_alloc = 1000;
		soa->p = (uint32 *)xmalloc(soa->num_p_alloc * 
					sizeof(uint32));
		for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xmalloc(
						soa->num_p_alloc * 
						sizeof(uint64));
		}
	}
}

static void
q_soa_array_free(q_soa_array_t *s)
{
	uint32 i, j;

	for (i = 0; i < s->num_arrays; i++) {
		q_soa_var_t *soa = s->soa + i;

		free(soa->p);
		for (j = 0; j < soa->num_roots; j++)
			free(soa->roots[j]);
	}
}

static void
q_soa_array_reset(q_soa_array_t *s)
{
	uint32 i;

	s->num_p = 0;
	for (i = 0; i < s->num_arrays; i++)
		s->soa[i].num_p = 0;
}

static void
q_soa_var_grow(q_soa_var_t *soa)
{
	uint32 i;

	soa->num_p_alloc *= 2;
	soa->p = (uint32 *)xrealloc(soa->p, 
				soa->num_p_alloc * 
				sizeof(uint32));
	for (i = 0; i < soa->num_roots; i++) {
		soa->roots[i] = (uint64 *)xrealloc(soa->roots[i], 
					soa->num_p_alloc * 
					sizeof(uint64));
	}
}

static void 
store_q_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i, j;
	lattice_fb_t *L = (lattice_fb_t *)extra;
	q_soa_array_t *s = (q_soa_array_t *)(L->q_array);

	for (i = 0; i < s->num_arrays; i++) {
		uint32 num;
		q_soa_var_t *soa = s->soa + i;

		if (soa->num_roots != num_roots)
			continue;

		num = soa->num_p;
		if (soa->num_p_alloc == num)
			q_soa_var_grow(soa);

		soa->p[num] = p;
		for (j = 0; j < num_roots; j++)
			soa->roots[j][num] = roots[j];

		soa->num_p++;
		s->num_p++;
		break;
	}
}

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 num_p;
	uint32 p_size;
	uint32 p_size_alloc;
	p_packed_t *curr;
	p_packed_t *packed_array;
} p_packed_var_t;

static void 
p_packed_init(p_packed_var_t *s)
{
	memset(s, 0, sizeof(p_packed_var_t));

	s->p_size_alloc = 1000;
	s->packed_array = s->curr = (p_packed_t *)xmalloc(s->p_size_alloc *
						sizeof(p_packed_t));
}

static void 
p_packed_free(p_packed_var_t *s)
{
	free(s->packed_array);
}

static void 
p_packed_reset(p_packed_var_t *s)
{
	s->num_p = s->p_size = 0;
	s->curr = s->packed_array;
}

static p_packed_t * 
p_packed_next(p_packed_t *curr)
{
	return (p_packed_t *)((uint64 *)curr + 
			P_PACKED_HEADER_WORDS + curr->num_roots);
}

static void 
store_p_packed(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i;
	lattice_fb_t *L = (lattice_fb_t *)extra;
	p_packed_var_t *s = (p_packed_var_t *)(L->p_array);
	p_packed_t *curr;

	if ((p_packed_t *)((uint64 *)s->curr + s->p_size) + 1 >=
			s->packed_array + s->p_size_alloc ) {

		s->p_size_alloc *= 2;
		s->packed_array = (p_packed_t *)xrealloc(
						s->packed_array,
						s->p_size_alloc *
						sizeof(p_packed_t));
		s->curr = (p_packed_t *)((uint64 *)s->packed_array + s->p_size);
	}

	curr = s->curr;
	curr->p = p;
	curr->lattice_size = 2 * L->poly->sieve_size / 
				((double)p * p);
	curr->num_roots = num_roots;
	curr->pad = 0;
	for (i = 0; i < num_roots; i++)
		curr->roots[i] = roots[i];

	s->num_p++;
	s->curr = p_packed_next(s->curr);
	s->p_size = ((uint8 *)s->curr - 
			(uint8 *)s->packed_array) / sizeof(uint64);
}

/*------------------------------------------------------------------------*/
static uint32
sieve_lattice_batch(msieve_obj *obj, lattice_fb_t *L,
			uint32 threads_per_block,
			p_packed_var_t *p_array,
			q_soa_array_t *q_array,
			gpu_info_t *gpu_info, 
			CUfunction gpu_kernel)
{
	uint32 i, j;
	p_packed_t *packed_array = p_array->packed_array;
	q_soa_t *q_marshall = (q_soa_t *)L->q_marshall;
	uint32 num_blocks;
	uint32 num_p_offset;
	uint32 num_q_offset;
	uint32 num_qroots_offset;
	uint32 found_array_size = L->found_array_size;
	found_t *found_array = (found_t *)L->found_array;
	void *gpu_ptr;

	i = 0;
	gpu_ptr = (void *)(size_t)L->gpu_q_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_q_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_qroots_offset = i;
	i += sizeof(uint32);

	CUDA_ALIGN_PARAM(i, __alignof(uint32));
	num_p_offset = i;
	i += sizeof(uint32);

	gpu_ptr = (void *)(size_t)L->gpu_found_array;
	CUDA_ALIGN_PARAM(i, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(gpu_kernel, (int)i, 
			&gpu_ptr, sizeof(gpu_ptr)))
	i += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(gpu_kernel, i))


	for (i = 0; i < q_array->num_arrays; i++) {

		q_soa_var_t *soa = q_array->soa + i;
		uint32 num_qroots = soa->num_roots;
		uint32 num_q_done = 0;
		float elapsed_ms;

		if (soa->num_p < threads_per_block)
			continue;

		CUDA_TRY(cuParamSeti(gpu_kernel, 
				num_qroots_offset, num_qroots))

		while (num_q_done < soa->num_p) {

			uint32 num_p_done = 0;
			uint32 packed_words = 0;
			uint32 curr_num_p = 0;
			p_packed_t *packed_start = packed_array;

			uint32 curr_num_q = MIN(3 * found_array_size,
						soa->num_p - num_q_done);

			curr_num_q = MIN(curr_num_q, Q_SOA_BATCH_SIZE);

			memcpy(q_marshall->p, 
				soa->p + num_q_done,
				curr_num_q * sizeof(uint32));
			for (j = 0; j < num_qroots; j++) {
				memcpy(q_marshall->roots[j],
					soa->roots[j] + num_q_done,
					curr_num_q * sizeof(uint64));
			}

			CUDA_TRY(cuMemcpyHtoD(L->gpu_q_array, q_marshall,
					Q_SOA_BATCH_SIZE * (sizeof(uint32) +
						num_qroots * sizeof(uint64))))

			CUDA_TRY(cuParamSeti(gpu_kernel, num_q_offset, 
						curr_num_q))

			while (num_p_done < p_array->num_p) {
				p_packed_t *curr_packed = packed_start;

				do {
					uint32 next_words = packed_words +
							P_PACKED_HEADER_WORDS +
							curr_packed->num_roots;

					if (next_words >= P_ARRAY_WORDS)
						break;

					curr_num_p++;
					packed_words = next_words;
					curr_packed = p_packed_next(
								curr_packed);
				} while (++num_p_done < p_array->num_p);

#if 0
				printf("qroots %u qnum %u pnum %u pwords %u\n",
						num_qroots, curr_num_q,
						curr_num_p, packed_words);
#endif
				CUDA_TRY(cuMemcpyHtoD(L->gpu_p_array, 
							packed_start,
							packed_words *
							sizeof(uint64)))

				CUDA_TRY(cuParamSeti(gpu_kernel, num_p_offset, 
							curr_num_p))

				num_blocks = gpu_info->num_compute_units;
				if (curr_num_q < found_array_size) {
					num_blocks = (curr_num_q + 
						threads_per_block - 1) /
						threads_per_block;
				}

				CUDA_TRY(cuEventRecord(L->start, 0))
				CUDA_TRY(cuLaunchGrid(gpu_kernel, 
							num_blocks, 1))
				CUDA_TRY(cuEventRecord(L->end, 0))
				CUDA_TRY(cuEventSynchronize(L->end))

				CUDA_TRY(cuMemcpyDtoH(found_array, 
							L->gpu_found_array, 
							threads_per_block * 
							num_blocks *
							sizeof(found_t)))

				for (j = 0; j < threads_per_block *
						num_blocks; j++) {
					found_t *f = found_array + j;

					if (f->p > 0) {
						uint128 proot = {{0}};
						uint128 res;
						uint64 p2 = (uint64)f->p * f->p;

						proot.w[0] = (uint32)f->proot;
						proot.w[1] = 
						      (uint32)(f->proot >> 32);

						res = add128(proot, mul64(
								f->offset, p2));

						handle_collision(L->poly,
								f->p, f->q,
								1, 0, res);
					}
				}

				CUDA_TRY(cuEventElapsedTime(&elapsed_ms,
							L->start, L->end))
				L->deadline -= elapsed_ms / 1000;

				if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
					return 1;

				if (L->deadline < 0)
					return 1;

				packed_start = curr_packed;
				packed_words = 0;
				curr_num_p = 0;
			}

			num_q_done += curr_num_q;
		}
	}

	return 0;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_nospecialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_small_p,
		uint32 small_p_min, uint32 small_p_max,  
		sieve_fb_t *sieve_large_p, 
		uint32 large_p_min, uint32 large_p_max)
{
	uint32 quit = 0;
	double cpu_start_time = get_cpu_time();
	p_packed_var_t * p_array;
	q_soa_array_t * q_array;
	uint32 degree = L->poly->degree;
	uint32 p_min_roots, p_max_roots;
	uint32 q_min_roots, q_max_roots;
	uint32 threads_per_block;
	gpu_info_t *gpu_info = L->poly->gpu_info;
	CUmodule gpu_module = L->poly->gpu_module_nosq;
       	CUfunction gpu_kernel;
	uint32 host_p_batch_size;
	uint32 host_q_batch_size;

	if (large_p_max < ((uint32)1 << 24))
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
				gpu_module, "sieve_kernel_48"))
	else
		CUDA_TRY(cuModuleGetFunction(&gpu_kernel, 
				gpu_module, "sieve_kernel_64"))


	L->q_marshall = (q_soa_t *)xmalloc(sizeof(q_soa_t));
	q_array = L->q_array = (q_soa_array_t *)xmalloc(
					sizeof(q_soa_array_t));
	p_array = L->p_array = (p_packed_var_t *)xmalloc(
					sizeof(p_packed_var_t));
	p_packed_init(p_array);
	q_soa_array_init(q_array, degree);

	CUDA_TRY(cuMemAlloc(&L->gpu_q_array, sizeof(q_soa_t)))
	CUDA_TRY(cuModuleGetGlobal(&L->gpu_p_array, 
				NULL, gpu_module, "pbatch"))

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

	CUDA_TRY(cuEventCreate(&L->start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&L->end, CU_EVENT_BLOCKING_SYNC))

	host_p_batch_size = MAX(10000, L->found_array_size / 3);
	host_q_batch_size = MAX(50000, 12 * L->found_array_size);

	p_min_roots = 1;
	p_max_roots = degree * degree;

	q_min_roots = degree;
	q_max_roots = degree * degree;
	if (degree == 6)
		q_max_roots *= 2;

	sieve_fb_reset(sieve_large_p, large_p_min, large_p_max,
			q_min_roots, q_max_roots);
	while (!quit) {

		double curr_time;

		q_soa_array_reset(q_array);
		while (sieve_fb_next(sieve_large_p, L->poly, store_q_soa,
					L) != P_SEARCH_DONE)
			if (q_array->num_p == host_q_batch_size)
				break;

		if (q_array->num_p == 0)
			break;

		sieve_fb_reset(sieve_small_p, small_p_min, small_p_max,
				p_min_roots, p_max_roots);
		while (!quit) {

			p_packed_reset(p_array);
			while (sieve_fb_next(sieve_small_p, L->poly, 
						store_p_packed,
						L) != P_SEARCH_DONE)
				if (p_array->num_p == host_p_batch_size)
					break;

			if (p_array->num_p == 0)
				break;

			quit = sieve_lattice_batch(obj, L,
					threads_per_block,
					p_array, q_array,
					gpu_info, gpu_kernel);
		}

		curr_time = get_cpu_time();
		L->deadline -= curr_time - cpu_start_time;
		cpu_start_time = curr_time;
	}

	CUDA_TRY(cuMemFree(L->gpu_q_array))
	CUDA_TRY(cuMemFree(L->gpu_found_array))
	CUDA_TRY(cuEventDestroy(L->start))
	CUDA_TRY(cuEventDestroy(L->end))
	p_packed_free(p_array);
	q_soa_array_free(q_array);
	free(p_array);
	free(q_array);
	free(L->found_array);
	free(L->q_marshall);
	return quit;
}

/*------------------------------------------------------------------------*/
uint32
sieve_lattice_gpu_nosq(msieve_obj *obj, lattice_fb_t *L)
{
	uint32 quit;
	uint32 large_p_min, large_p_max;
	uint32 small_p_min, small_p_max;
	uint32 large_p_fb_max;
	uint32 degree = L->poly->degree;
	double p_size_max = L->poly->p_size_max;
	sieve_fb_t sieve_large_p, sieve_small_p;

	if (sqrt(p_size_max) > (uint32)(-1) / P_SCALE) {
		printf("error: invalid parameters for rational coefficient "
			"in sieve_lattice_gpu_nosq()\n");
		return 0;
	}

	large_p_min = sqrt(p_size_max);
	large_p_max = large_p_min * P_SCALE;
	small_p_max = large_p_min - 1;
	small_p_min = small_p_max / P_SCALE;

	large_p_fb_max = sqrt(small_p_min) / 2;

	sieve_fb_init(&sieve_small_p, L->poly,
			large_p_fb_max, 100000, 1, degree, 0);

	sieve_fb_init(&sieve_large_p, L->poly,
			5, large_p_fb_max, 1, degree, 0);

	while (1) {
		gmp_printf("coeff %Zd p1 %u - %u p2 %u - %u\n",
				L->poly->high_coeff,
				small_p_min, small_p_max,
				large_p_min, large_p_max);

		quit = sieve_nospecialq_64(obj, L,
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

		if (small_p_min < large_p_fb_max)
			break;
	}

	sieve_fb_free(&sieve_large_p);
	sieve_fb_free(&sieve_small_p);
	return quit;
}
