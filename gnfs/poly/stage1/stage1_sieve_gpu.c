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
#include <stage1_core_gpu/stage1_core.h>

/* GPU collision search; this code looks for self-collisions
   among arithmetic progressions, by finding k1 and k2 such that
   for two arithmetic progressions r1+k*p1^2 and r2+k*p2^2 we
   have

      r1 + k1*p1^2 = r2 + k2*p2^2

   such that
      - p1 and p2 are coprime and < 2^32
      - the value where they coincide is of size smaller
        than a fixed bound

   This code uses a sort routine to find collisions across all the
   p1 and p2 in the set simultaneously. We further use a 'special-q'
   formulation where all the inputs to the sort routine are
   constrained to fall on a third arithmetic progression r3 + k*q^2
   for some k. We choose a given q and for each of its roots run the
   complete sort. This is analogous to lattice sieving across the
   interval.
   
   This allows us to choose q so that the sort problem is of
   reasonable size but the collisions found are still over the
   original, impractically large range. */

/*------------------------------------------------------------------------*/

typedef struct {
	uint32 num_roots;
	uint32 num_p;
	uint32 num_p_alloc;

	CUdeviceptr dev_p;
	CUdeviceptr dev_start_roots;
	CUdeviceptr dev_p_entry;
	CUdeviceptr dev_root_entry;
	CUstream stream;

	uint32 *p;
	uint64 *roots[MAX_ROOTS];

} p_soa_var_t;

#define MAX_P_SOA_ARRAYS 16

typedef struct {
	uint32 num_arrays;
	uint32 num_p;

	uint32 max_p_roots;
	p_soa_var_t *soa;

} p_soa_array_t;

static void
p_soa_array_init(p_soa_array_t *s, uint32 degree)
{
	uint32 i, j;
	memset(s, 0, sizeof(p_soa_array_t));

	s->soa = (p_soa_var_t *)xmalloc(MAX_P_SOA_ARRAYS *
					sizeof(p_soa_var_t));

	switch (degree) {
	case 4:
		s->num_arrays = 3;
		s->soa[0].num_roots = 2;
		s->soa[1].num_roots = 4;
		s->soa[2].num_roots = 8;
		s->max_p_roots = 8;
		break;

	case 5:
		s->num_arrays = 3;
		s->soa[0].num_roots = 1;
		s->soa[1].num_roots = 5;
		s->soa[2].num_roots = 25;
		s->max_p_roots = 25;
		break;

	case 6:
		s->num_arrays = 5;
		s->soa[0].num_roots = 2;
		s->soa[1].num_roots = 4;
		s->soa[2].num_roots = 6;
		s->soa[3].num_roots = 12;
		s->soa[4].num_roots = 36;
		s->max_p_roots = 36;
		break;

	case 7: /* ;) */
		s->num_arrays = 3;
		s->soa[0].num_roots = 1;
		s->soa[1].num_roots = 7;
		s->soa[2].num_roots = 49;
		s->max_p_roots = 49;
		break;
	}

	for (i = 0; i < s->num_arrays; i++) {
		p_soa_var_t *soa = s->soa + i;

		soa->num_p = 0;
		soa->num_p_alloc = 256;
		soa->p = (uint32 *)xmalloc(soa->num_p_alloc * sizeof(uint32));
		for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xmalloc(soa->num_p_alloc *
								sizeof(uint64));
		}
	}
}

static void
p_soa_var_free(p_soa_var_t *soa)
{
	uint32 i;

	free(soa->p);
	for (i = 0; i < soa->num_roots; i++)
		free(soa->roots[i]);
}

static void
p_soa_array_free(p_soa_array_t *s)
{
	uint32 i;

	for (i = 0; i < s->num_arrays; i++)
		p_soa_var_free(s->soa + i);
}

static void
p_soa_array_compact(p_soa_array_t *s)
{
	uint32 i, j;

	i = 0;
	while (i < s->num_arrays) {
		if (s->soa[i].num_p == 0) {

			s->num_arrays--;
			p_soa_var_free(s->soa + i);
			for (j = i; j < s->num_arrays; j++)
				s->soa[j] = s->soa[j + 1];
		}
		else {
			i++;
		}
	}
}

static void
store_p_soa(uint32 p, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i, j;
	p_soa_array_t *s = (p_soa_array_t *)extra;

	j = (uint32)(-1);
	for (i = 0; i < s->num_arrays; i++) {

		if (s->soa[i].num_roots > num_roots)
			continue;

		if (j > i || s->soa[i].num_roots > s->soa[j].num_roots)
			j = i;
	}

	if (j < s->num_arrays) {
		p_soa_var_t *soa = s->soa + j;

		if (soa->num_p_alloc == soa->num_p) {
			soa->num_p_alloc *= 2;
			soa->p = (uint32 *)xrealloc(soa->p, soa->num_p_alloc *
							sizeof(uint32));
			for (j = 0; j < soa->num_roots; j++) {
			soa->roots[j] = (uint64 *)xrealloc(soa->roots[j],
					soa->num_p_alloc * sizeof(uint64));
			}
		}

		soa->p[soa->num_p] = p;
		for (j = 0; j < soa->num_roots; j++)
			soa->roots[j][soa->num_p] = roots[j];

		soa->num_p++;
		s->num_p++;
	}
}

#define NUM_SPECIALQ_ALLOC (BATCH_SPECIALQ_MAX * 5)

typedef struct {
	uint32 num_specialq;
	specialq_t specialq[NUM_SPECIALQ_ALLOC];
} specialq_array_t;

static void
store_specialq(uint32 q, uint32 num_roots, uint64 *roots, void *extra)
{
	uint32 i;
	specialq_array_t *q_array = (specialq_array_t *)extra;

	for (i = 0; i < num_roots &&
			q_array->num_specialq < NUM_SPECIALQ_ALLOC; i++) {

		q_array->specialq[q_array->num_specialq].p = q;
		q_array->specialq[q_array->num_specialq].root = roots[i];

		q_array->num_specialq++;
	}
}

typedef struct {

	p_soa_array_t *p_array;

	uint32 num_entries;

	CUfunction *gpu_kernel;
	uint32 *threads_per_block;

	CUdeviceptr gpu_p_array;
	CUdeviceptr gpu_q_array;
	CUdeviceptr gpu_found_array;
	CUdeviceptr gpu_root_array;
	found_t *found_array;
	uint32 found_array_size;

	CUevent start;
	CUevent end;

	double gpu_elapsed;

} device_data_t;

/*------------------------------------------------------------------------*/
static void
check_found_array(poly_search_t *poly, device_data_t *d)
{
	uint32 i;
	uint32 found_array_size = d->found_array_size;
	float elapsed_ms;
	found_t *found_array = d->found_array;

	CUDA_TRY(cuEventRecord(d->start, 0))

	CUDA_TRY(cuMemcpyDtoH(found_array, d->gpu_found_array,
			d->found_array_size * sizeof(found_t)))
	CUDA_TRY(cuMemsetD8(d->gpu_found_array, 0,
			d->found_array_size * sizeof(found_t)))

	for (i = 0; i < found_array_size; i++) {
		found_t *found = found_array + i;
		uint32 p1 = found->p1;
		uint32 p2 = found->p2;
		uint32 q = found->q;
		uint64 qroot = found->qroot;
		int64 offset = found->offset;

		if (p1 != 0) {
			double check = (double)p1 * p2;

			check = check * check * poly->coeff_max
					/ poly->m0 / poly->degree;

			if (fabs((double)offset) < check) {

				handle_collision(poly, (uint64)p1 * p2, q,
						qroot, offset);
			}
		}
	}

	CUDA_TRY(cuEventRecord(d->end, 0))
	CUDA_TRY(cuEventSynchronize(d->end))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms, d->start, d->end))
	d->gpu_elapsed += elapsed_ms / 1000;
}

#define MAX_SPECIAL_Q ((uint32)(-1))
#define MAX_OTHER ((uint32)1 << 27)

#define NUM_GPU_FUNCTIONS 5
#define GPU_TRANS 0
#define GPU_SORT 1
#define GPU_MERGE 2
#define GPU_MERGE1 3
#define GPU_FINAL 4

#define SHARED_ELEM_SIZE (sizeof(uint32) + sizeof(uint64))

/*------------------------------------------------------------------------*/
static uint32
handle_special_q_batch(msieve_obj *obj, poly_search_t *poly,
			device_data_t *d, specialq_t *specialq_batch,
			uint32 num_specialq)
{
	uint32 i, j, k;
	uint32 j_offset, k_offset;
	uint32 quit = 0;
	p_soa_array_t *p_array = d->p_array;
	uint32 num_blocks;
	float elapsed_ms;
	void *gpu_ptr;

	CUDA_TRY(cuEventRecord(d->start, 0))

	CUDA_TRY(cuMemcpyHtoD(d->gpu_q_array, specialq_batch,
			sizeof(specialq_t) * num_specialq))

	CUDA_TRY(cuMemsetD32(d->gpu_p_array, 0,
			num_specialq * d->num_entries))
	CUDA_TRY(cuMemsetD32(d->gpu_root_array, 0,
			2 * num_specialq * d->num_entries))

	for (i = 0; i < p_array->num_arrays; i++) {
		p_soa_var_t *soa = p_array->soa + i;
		uint32 num = soa->num_p;

		j = 0;
		gpu_ptr = (void *)(size_t)soa->dev_p;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_TRANS], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_TRANS], (int)j,
				(int)num))
		j += sizeof(uint32);

		gpu_ptr = (void *)(size_t)soa->dev_start_roots;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_TRANS], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_TRANS], (int)j,
				(int)soa->num_roots))
		j += sizeof(uint32);

		gpu_ptr = (void *)(size_t)soa->dev_p_entry;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_TRANS], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		gpu_ptr = (void *)(size_t)soa->dev_root_entry;
		CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
		CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_TRANS], (int)j,
				&gpu_ptr, sizeof(gpu_ptr)))
		j += sizeof(gpu_ptr);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_TRANS], (int)j,
				(int)num_specialq))
		j += sizeof(uint32);

		CUDA_ALIGN_PARAM(j, __alignof(uint32));
		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_TRANS], (int)j,
				(int)d->num_entries))
		j += sizeof(uint32);

		CUDA_TRY(cuParamSetSize(d->gpu_kernel[GPU_TRANS], j))

		num_blocks = (num - 1) /
				d->threads_per_block[GPU_TRANS] + 1;
		CUDA_TRY(cuLaunchGridAsync(d->gpu_kernel[GPU_TRANS],
				num_blocks, 1, soa->stream))
	}

	j = 0;
	gpu_ptr = (void *)(size_t)d->gpu_p_array;
	CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_SORT], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_MERGE], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_MERGE1], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	j += sizeof(gpu_ptr);

	gpu_ptr = (void *)(size_t)d->gpu_root_array;
	CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_SORT], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_MERGE], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_MERGE1], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	j += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(d->gpu_kernel[GPU_SORT], j))

	CUDA_ALIGN_PARAM(j, __alignof(uint32));
	j_offset = j;
	j += sizeof(uint32);

	CUDA_TRY(cuParamSetSize(d->gpu_kernel[GPU_MERGE], j))

	CUDA_ALIGN_PARAM(j, __alignof(uint32));
	k_offset = j;
	j += sizeof(uint32);

	CUDA_TRY(cuParamSetSize(d->gpu_kernel[GPU_MERGE1], j))

	num_blocks = d->num_entries / d->threads_per_block[GPU_SORT] / 2;

	CUDA_TRY(cuLaunchGrid(d->gpu_kernel[GPU_SORT],
			num_blocks, num_specialq))

	j = 2 * d->threads_per_block[GPU_SORT];
	for (; j < d->num_entries; j *= 2) {

		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_MERGE],
				(int)j_offset, j))
		CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_MERGE1],
				(int)j_offset, j))

		num_blocks = d->num_entries /
				d->threads_per_block[GPU_MERGE1] / 2;

		for (k = j; k > d->threads_per_block[GPU_MERGE];
							k /= 2) {

			CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_MERGE1],
					(int)k_offset, k))

			CUDA_TRY(cuLaunchGrid(d->gpu_kernel[GPU_MERGE1],
					num_blocks, num_specialq))
		}

		num_blocks = d->num_entries /
				d->threads_per_block[GPU_MERGE] / 2;

		CUDA_TRY(cuLaunchGrid(d->gpu_kernel[GPU_MERGE],
				num_blocks, num_specialq))
	}

	j = 0;
	gpu_ptr = (void *)(size_t)d->gpu_p_array;
	CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_FINAL], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	j += sizeof(gpu_ptr);

	gpu_ptr = (void *)(size_t)d->gpu_root_array;
	CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_FINAL], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	j += sizeof(gpu_ptr);

	CUDA_ALIGN_PARAM(j, __alignof(uint32));
	CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_FINAL], (int)j,
			d->num_entries))
	j += sizeof(uint32);

	CUDA_ALIGN_PARAM(j, __alignof(uint32));
	CUDA_TRY(cuParamSeti(d->gpu_kernel[GPU_FINAL], (int)j,
			num_specialq))
	j += sizeof(uint32);

	gpu_ptr = (void *)(size_t)d->gpu_found_array;
	CUDA_ALIGN_PARAM(j, __alignof(gpu_ptr));
	CUDA_TRY(cuParamSetv(d->gpu_kernel[GPU_FINAL], (int)j,
			&gpu_ptr, sizeof(gpu_ptr)))
	j += sizeof(gpu_ptr);

	CUDA_TRY(cuParamSetSize(d->gpu_kernel[GPU_FINAL], j))

	num_blocks = (d->num_entries * num_specialq - 1) /
			d->threads_per_block[GPU_FINAL] + 1;
	if (num_blocks > (uint32)poly->gpu_info->num_compute_units)
		num_blocks = poly->gpu_info->num_compute_units;

	CUDA_TRY(cuLaunchGrid(d->gpu_kernel[GPU_FINAL], num_blocks, 1))

	CUDA_TRY(cuEventRecord(d->end, 0))
	CUDA_TRY(cuEventSynchronize(d->end))
	CUDA_TRY(cuEventElapsedTime(&elapsed_ms, d->start, d->end))
	d->gpu_elapsed += elapsed_ms / 1000;

	if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
		quit = 1;

	return quit;
}

/*------------------------------------------------------------------------*/
static INLINE uint32
next_2pow(uint32 n)
{
	uint32 i;

	for (i = 1; i > 0 && i < n; i *= 2);

	return i;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, poly_search_t *poly,
		sieve_fb_t *sieve_special_q, sieve_fb_t *sieve_p,
		uint32 special_q_min, uint32 special_q_max,
		uint32 p_min, uint32 p_max, double deadline, double *elapsed)
{
	uint32 i, j, k;
	uint32 quit = 0;
	uint32 all_q_done = 0;
	uint32 degree = poly->degree;
	uint32 num_batch_specialq;
	uint32 pass_cnt = 0;
	specialq_array_t q_array;
	p_soa_array_t p_array;
	device_data_t data;
	double cpu_start_time = get_cpu_time();
	CUmodule gpu_module = poly->gpu_module;

	*elapsed = 0;

	memset(&data, 0, sizeof(device_data_t));

	data.p_array = &p_array;
	p_soa_array_init(&p_array, degree);

	/* build all the arithmetic progressions */

	sieve_fb_reset(sieve_p, p_min, p_max, 1, p_array.max_p_roots);
	while (sieve_fb_next(sieve_p, poly, store_p_soa,
			&p_array) != P_SEARCH_DONE) {
		;
	}
	p_soa_array_compact(&p_array);

	for (i = j = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;

		j += soa->num_p * soa->num_roots;
	}
	data.num_entries = next_2pow(j);

#if 1
	printf("aprogs: %u entries, %u roots\n", p_array.num_p, j);
#endif

	num_batch_specialq = MIN(BATCH_SPECIALQ_MAX,
				poly->gpu_info->num_compute_units * 3);
	num_batch_specialq = MIN(num_batch_specialq,
					poly->gpu_info->global_mem_size /
					data.num_entries / 3 /
					SHARED_ELEM_SIZE - 1);
	num_batch_specialq = MAX(1, num_batch_specialq);
	printf("batch size %u\n", num_batch_specialq);

	CUDA_TRY(cuMemAlloc(&data.gpu_p_array,
			num_batch_specialq *
			data.num_entries * sizeof(uint32)))
	CUDA_TRY(cuMemAlloc(&data.gpu_root_array,
			num_batch_specialq *
			data.num_entries * sizeof(uint64)))
	CUDA_TRY(cuModuleGetGlobal(&data.gpu_q_array, 
			NULL, gpu_module, "q_batch"))

	for (i = j = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;
		uint32 num = soa->num_p;

		soa->dev_p_entry = data.gpu_p_array + j * sizeof(uint32);
		soa->dev_root_entry = data.gpu_root_array + j * sizeof(uint64);

		j += num * soa->num_roots;

		CUDA_TRY(cuStreamCreate(&soa->stream, 0))

		CUDA_TRY(cuMemAlloc(&soa->dev_p, num * sizeof(uint32)))
		CUDA_TRY(cuMemAlloc(&soa->dev_start_roots,
				soa->num_roots * num * sizeof(uint64)))

		CUDA_TRY(cuMemcpyHtoD(soa->dev_p, soa->p,
				num * sizeof(uint32)))

		for (k = 0; k < soa->num_roots; k++) {
			CUDA_TRY(cuMemcpyHtoD(soa->dev_start_roots +
					k * num * sizeof(uint64),
					soa->roots[k], num * sizeof(uint64)))
		}
	}

	CUDA_TRY(cuEventCreate(&data.start, CU_EVENT_BLOCKING_SYNC))
	CUDA_TRY(cuEventCreate(&data.end, CU_EVENT_BLOCKING_SYNC))

	data.gpu_kernel = (CUfunction *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(CUfunction));
	data.threads_per_block = (uint32 *)xmalloc(NUM_GPU_FUNCTIONS *
				sizeof(uint32));

	CUDA_TRY(cuModuleGetFunction(&data.gpu_kernel[GPU_TRANS],
				gpu_module, "sieve_kernel_trans"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				data.gpu_kernel[GPU_TRANS]))
	data.threads_per_block[GPU_TRANS] = MIN(i, 256);
	CUDA_TRY(cuFuncSetBlockShape(data.gpu_kernel[GPU_TRANS],
				data.threads_per_block[GPU_TRANS], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&data.gpu_kernel[GPU_FINAL],
				gpu_module, "sieve_kernel_final"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				data.gpu_kernel[GPU_FINAL]))
	data.threads_per_block[GPU_FINAL] = MIN(i, 256);
	CUDA_TRY(cuFuncSetBlockShape(data.gpu_kernel[GPU_FINAL],
				data.threads_per_block[GPU_FINAL], 1, 1))

	CUDA_TRY(cuModuleGetFunction(&data.gpu_kernel[GPU_SORT],
				gpu_module, "sieve_kernel_sort"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				data.gpu_kernel[GPU_SORT]))
	i = MIN(i, data.num_entries / 2);
	i = MIN(i, (poly->gpu_info->shared_mem_size - 4096) /
			SHARED_ELEM_SIZE / 2);
	data.threads_per_block[GPU_SORT] = next_2pow(i + 1) / 2;
	CUDA_TRY(cuFuncSetBlockShape(data.gpu_kernel[GPU_SORT],
				data.threads_per_block[GPU_SORT], 1, 1))
	CUDA_TRY(cuFuncSetSharedSize(data.gpu_kernel[GPU_SORT],
				2 * data.threads_per_block[GPU_SORT] *
				SHARED_ELEM_SIZE))

	CUDA_TRY(cuModuleGetFunction(&data.gpu_kernel[GPU_MERGE],
				gpu_module, "sieve_kernel_merge"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				data.gpu_kernel[GPU_MERGE]))
	i = MIN(i, 2 * data.threads_per_block[GPU_SORT]);
	i = MIN(i, (poly->gpu_info->shared_mem_size - 4096) /
			SHARED_ELEM_SIZE / 2);
	data.threads_per_block[GPU_MERGE] = next_2pow(i + 1) / 2;
	CUDA_TRY(cuFuncSetBlockShape(data.gpu_kernel[GPU_MERGE],
				data.threads_per_block[GPU_MERGE], 1, 1))
	CUDA_TRY(cuFuncSetSharedSize(data.gpu_kernel[GPU_MERGE],
				2 * data.threads_per_block[GPU_MERGE] *
				SHARED_ELEM_SIZE))

	CUDA_TRY(cuModuleGetFunction(&data.gpu_kernel[GPU_MERGE1],
				gpu_module, "sieve_kernel_merge1"))
	CUDA_TRY(cuFuncGetAttribute((int *)&i,
				CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
				data.gpu_kernel[GPU_MERGE1]))
	i = MIN(i, 2 * data.threads_per_block[GPU_SORT]);
	data.threads_per_block[GPU_MERGE1] = MIN(next_2pow(i + 1) / 2, 256);
	CUDA_TRY(cuFuncSetBlockShape(data.gpu_kernel[GPU_MERGE1],
				data.threads_per_block[GPU_MERGE1], 1, 1))

	data.found_array_size = data.threads_per_block[GPU_FINAL] *
			poly->gpu_info->num_compute_units;
	CUDA_TRY(cuMemAlloc(&data.gpu_found_array, sizeof(found_t) *
			data.found_array_size))
	data.found_array = (found_t *)xmalloc(sizeof(found_t) *
			data.found_array_size);

	CUDA_TRY(cuMemsetD8(data.gpu_found_array, 0,
		data.found_array_size * sizeof(found_t)))

	q_array.num_specialq = 0;
	if (special_q_min == 1) {

		q_array.specialq[0].p = 1;
		q_array.specialq[0].root = 0;
		q_array.num_specialq++;
	}

	sieve_fb_reset(sieve_special_q, special_q_min, special_q_max, 
			degree, MAX_ROOTS);
	while (!quit && !all_q_done) {

		all_q_done = sieve_fb_next(sieve_special_q, poly,
				store_specialq, &q_array) == P_SEARCH_DONE;

		if (all_q_done || q_array.num_specialq == NUM_SPECIALQ_ALLOC) {

			i = 0;
			while (i < q_array.num_specialq) {

				uint32 curr_num_specialq =
						MIN(q_array.num_specialq - i,
							num_batch_specialq);

				quit = handle_special_q_batch(obj, poly, &data,
						q_array.specialq + i,
						curr_num_specialq);

				if (quit)
					break;

				if (++pass_cnt % 16 == 0)
					check_found_array(poly, &data);

				i += curr_num_specialq;
			}

			q_array.num_specialq = 0;

			*elapsed = get_cpu_time() - cpu_start_time +
					data.gpu_elapsed;

			if (*elapsed > deadline)
				quit = 1;
		}
	}

	check_found_array(poly, &data);
	CUDA_TRY(cuMemFree(data.gpu_p_array))
	CUDA_TRY(cuMemFree(data.gpu_root_array))
	CUDA_TRY(cuMemFree(data.gpu_found_array))
	for (i = 0; i < p_array.num_arrays; i++) {
		p_soa_var_t *soa = p_array.soa + i;

		CUDA_TRY(cuMemFree(soa->dev_p));
		CUDA_TRY(cuMemFree(soa->dev_start_roots));
		CUDA_TRY(cuStreamDestroy(soa->stream));
	}
	free(data.found_array);
	free(data.gpu_kernel);
	free(data.threads_per_block);
	p_soa_array_free(&p_array);
	CUDA_TRY(cuEventDestroy(data.start))
	CUDA_TRY(cuEventDestroy(data.end))
	return quit;
}

/*------------------------------------------------------------------------*/
double
sieve_lattice_gpu(msieve_obj *obj, poly_search_t *poly, double deadline)
{
	uint32 degree = poly->degree;
	uint32 num_pieces;
	uint32 p_min, p_max;
	uint32 special_q_min, special_q_max;
	uint32 special_q_min2, special_q_max2;
	uint32 special_q_fb_max;
	double p_size_max = poly->p_size_max;
	double sieve_bound = poly->coeff_max / poly->m0 / degree;
	double elapsed = 0;
	sieve_fb_t sieve_p, sieve_special_q;

	/* size the problem; we choose p_min so that we can use
	   exactly one offset from each progression (the one
	   nearest to m0) in the search. Choosing larger p
	   implies that we could use more of their offsets, but
	   it appears not to be optimal to do so since the
	   biggest part of the search difficulty is the sorting
	   phase, and larger p implies that we need to sort more
	   of them to find each collision */

	p_min = MIN(MAX_OTHER / P_SCALE, sqrt(0.5 / sieve_bound));
	p_min = MIN(p_min, sqrt(p_size_max) / P_SCALE);

	p_max = p_min * P_SCALE;

	special_q_min = 1;
	special_q_max = MIN(MAX_SPECIAL_Q, p_size_max / p_max / p_max);

	/* set up the special q factory; special-q may have 
	   arbitrary factors, but many small factors are 
	   preferred since that will allow for many more roots
	   per special q, so we choose the factors to be as 
	   small as possible */

	special_q_fb_max = MIN(200000, special_q_max);
	sieve_fb_init(&sieve_special_q, poly,
			5, special_q_fb_max,
			1, degree,
			1);

	/* because special-q can have any factors, we require that
	   the progressions we generate use p that have somewhat
	   large factors. This minimizes the chance that a given
	   special-q has factors in common with many progressions
	   in the set */

	sieve_fb_init(&sieve_p, poly, 
			100, 5000,
			1, degree,
		       	0);

	num_pieces = (double)special_q_max * p_max
			/ log(special_q_max) / log(p_max)
			/ 3e9;
	num_pieces = MIN(num_pieces, 450);

	/* large search problems can be randomized so that
	   multiple runs over the same range of leading
	   a_d will likely generate different results */

	if (num_pieces > 1) { /* randomize the special_q range */
		uint32 piece_length = (special_q_max - special_q_min)
				/ num_pieces;
		uint32 piece = get_rand(&obj->seed1, &obj->seed2)
				% num_pieces;

		printf("randomizing rational coefficient: "
			"using piece #%u of %u\n",
			piece + 1, num_pieces);

		special_q_min2 = special_q_min + piece * piece_length;
		special_q_max2 = special_q_min2 + piece_length;
	}
	else {
		special_q_min2 = special_q_min;
		special_q_max2 = special_q_max;
	}

	gmp_printf("coeff %Zd specialq %u - %u other %u - %u\n",
			poly->high_coeff,
			special_q_min2, special_q_max2,
			p_min, p_max);

	sieve_specialq_64(obj, poly, &sieve_special_q, &sieve_p,
			special_q_min2, special_q_max2, p_min, p_max,
			deadline, &elapsed);

	sieve_fb_free(&sieve_special_q);
	sieve_fb_free(&sieve_p);
	return elapsed;
}
