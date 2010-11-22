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

#include "lanczos.h"

/*-------------------------------------------------------------------*/
static void mul_unpacked(packed_matrix_t *matrix,
			  uint64 *x, uint64 *b) {

	uint32 ncols = matrix->ncols;
	uint32 num_dense_rows = matrix->num_dense_rows;
	la_col_t *A = matrix->unpacked_cols;
	uint32 i, j;

	memset(b, 0, ncols * sizeof(uint64));
	
	for (i = 0; i < ncols; i++) {
		la_col_t *col = A + i;
		uint32 *row_entries = col->data;
		uint64 tmp = x[i];

		for (j = 0; j < col->weight; j++) {
			b[row_entries[j]] ^= tmp;
		}
	}

	if (num_dense_rows) {
		for (i = 0; i < ncols; i++) {
			la_col_t *col = A + i;
			uint32 *row_entries = col->data + col->weight;
			uint64 tmp = x[i];
	
			for (j = 0; j < num_dense_rows; j++) {
				if (row_entries[j / 32] & 
						((uint32)1 << (j % 32))) {
					b[j] ^= tmp;
				}
			}
		}
	}
}

/*-------------------------------------------------------------------*/
static void mul_trans_unpacked(packed_matrix_t *matrix,
				uint64 *x, uint64 *b) {

	uint32 ncols = matrix->ncols;
	uint32 num_dense_rows = matrix->num_dense_rows;
	la_col_t *A = matrix->unpacked_cols;
	uint32 i, j;

	for (i = 0; i < ncols; i++) {
		la_col_t *col = A + i;
		uint32 *row_entries = col->data;
		uint64 accum = 0;

		for (j = 0; j < col->weight; j++) {
			accum ^= x[row_entries[j]];
		}
		b[i] = accum;
	}

	if (num_dense_rows) {
		for (i = 0; i < ncols; i++) {
			la_col_t *col = A + i;
			uint32 *row_entries = col->data + col->weight;
			uint64 accum = b[i];
	
			for (j = 0; j < num_dense_rows; j++) {
				if (row_entries[j / 32] &
						((uint32)1 << (j % 32))) {
					accum ^= x[j];
				}
			}
			b[i] = accum;
		}
	}
}

/*-------------------------------------------------------------------*/
static void mul_packed(packed_matrix_t *matrix, uint64 *x, uint64 *b) {

	uint32 i;
	uint32 nrows = matrix->nrows;

	for (i = 0; i < matrix->num_threads; i++) {
		thread_data_t *t = matrix->thread_data + i;

		/* use each thread's scratch vector, except the
		   first thead, which has no scratch vector but
		   uses b instead */

		t->x = x;
		if (i == 0)
			t->b = b;
		memset(t->b, 0, nrows * sizeof(uint64));

		/* fire off each part of the matrix multiply
		   in a separate thread from the thread pool, 
		   except the last part. The current thread 
		   does the last partial multiply, and this 
		   saves one synchronize operation */

		if (i == matrix->num_threads - 1) {
			mul_packed_core(t);
		}
		else {
			t->command = COMMAND_MATMUL;
#if defined(WIN32) || defined(_WIN64)
			SetEvent(t->run_event);
#else
			pthread_cond_signal(&t->run_cond);
			pthread_mutex_unlock(&t->run_lock);
#endif
		}
	}

	/* wait for each thread to finish. All the scratch
	   vectors used by threads get xor-ed into the final b
	   vector */

	for (i = 0; i < matrix->num_threads; i++) {
		thread_data_t *t = matrix->thread_data + i;

		if (i < matrix->num_threads - 1) {
#if defined(WIN32) || defined(_WIN64)
			WaitForSingleObject(t->finish_event, INFINITE);
#else
			pthread_mutex_lock(&t->run_lock);
			while (t->command != COMMAND_WAIT)
				pthread_cond_wait(&t->run_cond, &t->run_lock);
#endif
		}

		if (i > 0)
			accum_xor(b, t->b, nrows);
	}

#if defined(GCC_ASM32A) && defined(HAS_MMX)
	ASM_G volatile ("emms");
#elif defined(MSC_ASM32A) && defined(HAS_MMX)
	ASM_M emms
#endif
}

/*-------------------------------------------------------------------*/
void mul_trans_packed(packed_matrix_t *matrix, uint64 *x, uint64 *b) {

	uint32 i;
	uint32 ncols = matrix->ncols;
	uint64 *tmp_b[MAX_THREADS];

	memset(b, 0, ncols * sizeof(uint64));

	for (i = 0; i < matrix->num_threads; i++) {
		thread_data_t *t = matrix->thread_data + i;

		/* separate threads fill up disjoint portions
		   of a single b vector, and do not need 
		   per-thread scratch space */

		tmp_b[i] = t->b;
		t->x = x;
		t->b = b;

		/* fire off each part of the matrix multiply
		   in a separate thread from the thread pool, 
		   except the last part. The current thread 
		   does the last partial multiply, and this 
		   saves one synchronize operation */

		if (i == matrix->num_threads - 1) {
			mul_trans_packed_core(t);
		}
		else {
			t->command = COMMAND_MATMUL_TRANS;
#if defined(WIN32) || defined(_WIN64)
			SetEvent(t->run_event);
#else
			pthread_cond_signal(&t->run_cond);
			pthread_mutex_unlock(&t->run_lock);
#endif
		}
	}

	/* wait for each thread to finish */

	for (i = 0; i < matrix->num_threads; i++) {
		thread_data_t *t = matrix->thread_data + i;

		if (i < matrix->num_threads - 1) {
#if defined(WIN32) || defined(_WIN64)
			WaitForSingleObject(t->finish_event, INFINITE);
#else
			pthread_mutex_lock(&t->run_lock);
			while (t->command != COMMAND_WAIT)
				pthread_cond_wait(&t->run_cond, &t->run_lock);
#endif
		}
		t->b = tmp_b[i];
	}

#if defined(GCC_ASM32A) && defined(HAS_MMX)
	ASM_G volatile ("emms");
#elif defined(MSC_ASM32A) && defined(HAS_MMX)
	ASM_M emms
#endif
}

/*-------------------------------------------------------------------*/
int compare_row_off(const void *x, const void *y) {
	entry_idx_t *xx = (entry_idx_t *)x;
	entry_idx_t *yy = (entry_idx_t *)y;

	if (xx->row_off > yy->row_off)
		return 1;
	if (xx->row_off < yy->row_off)
		return -1;

	return (int)xx->col_off - (int)yy->col_off;
}

/*--------------------------------------------------------------------*/
static int compare_uint16(const void *x, const void *y) {
        uint16 *xx = (uint16 *)x;
        uint16 *yy = (uint16 *)y;
        return (int)*xx - (int)*yy;
}

static void matrix_thread_init(thread_data_t *t) {

	uint32 i, j, k, m;
	uint32 num_row_blocks;
	uint32 num_col_blocks;
	uint32 dense_row_blocks;
	packed_block_t *curr_stripe;
#ifdef LARGEBLOCKS
	uint32 *n_in_a_row;
#else
	entry_idx_t *e;
#endif

	la_col_t *A = t->initial_cols;
	uint32 col_min = t->col_min;
	uint32 col_max = t->col_max;
	uint32 nrows = t->nrows_in;
	uint32 ncols = col_max - col_min + 1;
	uint32 block_size = t->block_size;
	uint32 block_col_size = MIN(block_size, 0xFFFF);
	uint32 num_dense_rows = t->num_dense_rows;
	uint32 first_block_size = t->first_block_size;

	t->nrows = nrows;
	t->ncols = ncols;

	/* each thread needs scratch space to store
	   matrix products. The first thread doesn't need
	   scratch space, it's provided by calling code */

	if (t->my_oid > 0)
		t->b = (uint64 *)xmalloc(MAX(nrows, ncols) * 
					sizeof(uint64));

	/* pack the dense rows 64 at a time */

	dense_row_blocks = (num_dense_rows + 63) / 64;
	if (dense_row_blocks) {
		t->dense_blocks = (uint64 **)xmalloc(dense_row_blocks *
						sizeof(uint64 *));
		for (i = 0; i < dense_row_blocks; i++) {
			t->dense_blocks[i] = (uint64 *)xmalloc(ncols *
							sizeof(uint64));
		}

		for (i = 0; i < ncols; i++) {
			la_col_t *c = A + col_min + i;
			uint32 *src = c->data + c->weight;
			for (j = 0; j < dense_row_blocks; j++) {
				t->dense_blocks[j][i] = 
						(uint64)src[2 * j + 1] << 32 |
						(uint64)src[2 * j];
			}
		}
	}

	/* allocate blocks in row-major order; a 'stripe' is
	   a vertical column of blocks. If packing the lowest
	   row indices, the first block has NUM_MEDIUM_ROWS rows
	   instead of block_size */

	/* with LARGEBLOCKS: blocks are rectangular, up to 2^32 rows, but <2^16 block_col_size */

	num_col_blocks = (ncols + (block_col_size-1)) / block_col_size;
	num_row_blocks = (nrows - first_block_size +
				(block_size-1)) / block_size + 1;

	t->num_blocks = num_row_blocks * num_col_blocks;
	t->blocks = curr_stripe = (packed_block_t *)xcalloc(
						(size_t)t->num_blocks,
						sizeof(packed_block_t));

#ifdef LARGEBLOCKS
	n_in_a_row = (uint32 *)xmalloc(block_size*sizeof(uint32));
#endif

	/* we convert the sparse part of the matrix to packed
	   format one stripe at a time. This limits the worst-
	   case memory use of the packing process */

	for (i = 0; i < num_col_blocks; i++, curr_stripe++) {

		uint32 curr_cols = MIN(block_col_size, ncols - i * block_col_size);
		packed_block_t *b;

		/* initialize the blocks in stripe i */

		for (j = 0, b = curr_stripe; j < num_row_blocks; j++) {

			if (j == 0) {
				b->start_row = 0;
				b->num_rows = first_block_size;
			}
			else {
				b->start_row = first_block_size +
						(j - 1) * block_size;
				b->num_rows = block_size;
			}

			b->start_col = col_min + i * block_col_size;
			b += num_col_blocks;
		}

		/* count the number of nonzero entries in each block */

		for (j = 0; j < curr_cols; j++) {
			la_col_t *c = A + col_min + i * block_col_size + j;

			for (k = 0, b = curr_stripe; k < c->weight; k++) {
				uint32 index = c->data[k];

				while (index >= b->start_row + b->num_rows)
					b += num_col_blocks;
				b->num_entries_alloc++;
			}
		}

		/* concatenate the nonzero elements of the matrix
		   columns corresponding to this stripe. Note that
		   the ordering of elements within a single block
		   is arbitrary, and it's possible that some non-
		   default ordering can enhance performance. However,
		   I've tried a few different ideas and they're much
		   slower than just storing the nonzeros in the order
		   in which they occur in the unpacked matrix 
		   
		   We technically can combine the previous pass through
		   the columns with this pass, but on some versions of
		   libc the number of reallocations causes an incredible
		   slowdown */

		for (j = 0, b = curr_stripe; j < num_row_blocks; j++) {
#ifdef LARGEBLOCKS
			b->row_off = (uint32 *)xmalloc(
						b->num_entries_alloc *
						sizeof(uint32));
			b->col_off = (uint16 *)xmalloc(
						b->num_entries_alloc *
						sizeof(uint16));
#else
			b->entries = (entry_idx_t *)xmalloc(
						b->num_entries_alloc *
						sizeof(entry_idx_t));
#endif
			b += num_col_blocks;
		}

		for (j = 0; j < curr_cols; j++) {
			la_col_t *c = A + col_min + i * block_col_size + j;

			for (k = 0, b = curr_stripe; k < c->weight; k++) {
				uint32 index = c->data[k];

				while (index >= b->start_row + b->num_rows)
					b += num_col_blocks;

#ifdef LARGEBLOCKS
				b->row_off[b->num_entries] = index - b->start_row;
				b->col_off[b->num_entries++] = j;
#else
				e = b->entries + b->num_entries++;
				e->row_off = index - b->start_row;
				e->col_off = j;
#endif
			}

			free(c->data);
			c->data = NULL;
		}

		/* convert the first block in the stripe to
		   a somewhat-compressed format. Entries in this
		   first block are stored by row, and all rows
		   are concatenated into a single 16-bit array */

#ifdef LARGEBLOCKS
		/* SB: instead of qsort, we will count all entries for each row; 
		   then rolling-sum the offsets,
		   then put them in place, and finally qsort each chunk */

		b = curr_stripe;
		memset(n_in_a_row, 0, block_size*sizeof(uint32));

		for (j = 0; j < b->num_entries; j++)
			n_in_a_row[b->row_off[j]]++;
                for (j = k = 0; j < block_size; j++)
                        if(n_in_a_row[j]) k++;

		/* we need a 16-bit word for each element and two more
		   16-bit words at the start of each of the k packed
		   arrays making up med_entries. The first extra word
		   gives the row number and the second gives the number
		   of entries in that row. We also need a few extra words 
		   at the array end because the multiply code uses a 
		   software pipeline and would fetch off the end of 
		   med_entries otherwise */

		/* SB: +1 more 16-bit word for high address of row_off */

		b->med_entries = (uint16 *)xmalloc((b->num_entries + 
					3 * k + 12) * sizeof(uint16));
		for (j = k = 0; j < block_size; j++) if((m=n_in_a_row[j])) {
			b->med_entries[k++] = j >> 16;
			b->med_entries[k++] = j & 0xFFFF; /* row_off in two */
			b->med_entries[k++] = m;
			n_in_a_row[j] = k; /* now reused and keeps offsets for writing */
			k += m;
		}
		b->med_entries[k] = b->med_entries[k+1] = b->med_entries[k+2] = 0;

		/* put all elements in their places, using the prepared offsets */
		for (j = 0; j < b->num_entries; j++) {
                        b->med_entries[n_in_a_row[b->row_off[j]]++] = b->col_off[j];
		}

		free(b->row_off);
		b->row_off = NULL;
		free(b->col_off);
		b->col_off = NULL;

		/* ok, let's also qsort each chunk; it helps the cache */
		for (k = 3; (m = b->med_entries[k-1]); k += m+3) {
			qsort(b->med_entries+k, m, sizeof(uint16), compare_uint16);
		}
#else
		b = curr_stripe;
		e = b->entries;
		qsort(e, (size_t)b->num_entries, 
				sizeof(entry_idx_t), compare_row_off);
		for (j = k = 1; j < b->num_entries; j++) {
			if (e[j].row_off != e[j-1].row_off)
				k++;
		}

		/* we need a 16-bit word for each element and two more
		   16-bit words at the start of each of the k packed
		   arrays making up med_entries. The first extra word
		   gives the row number and the second gives the number
		   of entries in that row. We also need a few extra words 
		   at the array end because the multiply code uses a 
		   software pipeline and would fetch off the end of 
		   med_entries otherwise */

		b->med_entries = (med_off_t *)xmalloc((b->num_entries + 
						2 * k + 8) * sizeof(med_off_t));
		j = k = 0;
		while (j < b->num_entries) {
			for (m = 0; j + m < b->num_entries; m++) {
				if (m > 0 && 
				    e[j+m].row_off != e[j+m-1].row_off)
					break;
				b->med_entries[k+m+2] = e[j+m].col_off;
			}
			b->med_entries[k] = e[j].row_off;
			b->med_entries[k+1] = m;
			j += m;
			k += m + 2;
		}
		b->med_entries[k] = b->med_entries[k+1] = 0;
		free(b->entries);
		b->entries = NULL;
#endif
	}
#ifdef LARGEBLOCKS
	free(n_in_a_row);
#endif
}

/*-------------------------------------------------------------------*/
static void matrix_thread_free(thread_data_t *t) {

	uint32 i;

	for (i = 0; i < (t->num_dense_rows + 63) / 64; i++)
		free(t->dense_blocks[i]);
	free(t->dense_blocks);

	for (i = 0; i < t->num_blocks; i++) {
#ifdef LARGEBLOCKS
		free(t->blocks[i].row_off);
		free(t->blocks[i].col_off);
#else
		free(t->blocks[i].entries);
#endif
		free(t->blocks[i].med_entries);
	}
	free(t->blocks);
	if (t->my_oid > 0)
		free(t->b);
}

/*-------------------------------------------------------------------*/
#if defined(WIN32) || defined(_WIN64)
static DWORD WINAPI worker_thread_main(LPVOID thread_data) {
#else
static void *worker_thread_main(void *thread_data) {
#endif
	thread_data_t *t = (thread_data_t *)thread_data;

	while(1) {

		/* wait forever for work to do */
#if defined(WIN32) || defined(_WIN64)
		WaitForSingleObject(t->run_event, INFINITE);
#else
		pthread_mutex_lock(&t->run_lock);
		while (t->command == COMMAND_WAIT) {
			pthread_cond_wait(&t->run_cond, &t->run_lock);
		}
#endif
		/* do work */

		switch (t->command) {
		case COMMAND_INIT:
			matrix_thread_init(t);
			break;
		case COMMAND_MATMUL:
			mul_packed_core(t);
			break;
		case COMMAND_MATMUL_TRANS:
			mul_trans_packed_core(t);
			break;
		case COMMAND_INNER_PRODUCT:
			core_64xN_Nx64(t->x, t->b, t->y, t->vsize);
			break;
		case COMMAND_OUTER_PRODUCT:
			core_Nx64_64x64_acc(t->x, t->b, t->y, t->vsize);
			break;
		default:
			goto thread_done;
		}

		/* signal completion */

		t->command = COMMAND_WAIT;
#if defined(WIN32) || defined(_WIN64)
		SetEvent(t->finish_event);
#else
		pthread_cond_signal(&t->run_cond);
		pthread_mutex_unlock(&t->run_lock);
#endif
	}

thread_done:
	matrix_thread_free(t);

#if defined(WIN32) || defined(_WIN64)
	return 0;
#else
	return NULL;
#endif
}

/*-------------------------------------------------------------------*/
static void start_worker_thread(thread_data_t *t, 
				uint32 is_master_thread) {

	/* create a thread that will handle matrix multiplies 
	   for block k of the matrix. The last block does 
	   not get its own thread (the current thread handles it) */

	if (is_master_thread) {
		matrix_thread_init(t);
		return;
	}

	t->command = COMMAND_INIT;
#if defined(WIN32) || defined(_WIN64)
	t->run_event = CreateEvent(NULL, FALSE, TRUE, NULL);
	t->finish_event = CreateEvent(NULL, FALSE, FALSE, NULL);
	t->thread_id = CreateThread(NULL, 0, worker_thread_main, t, 0, NULL);

	WaitForSingleObject(t->finish_event, INFINITE); /* wait for ready */
#else
	pthread_mutex_init(&t->run_lock, NULL);
	pthread_cond_init(&t->run_cond, NULL);

	pthread_cond_signal(&t->run_cond);
	pthread_mutex_unlock(&t->run_lock);
	pthread_create(&t->thread_id, NULL, worker_thread_main, t);

	pthread_mutex_lock(&t->run_lock); /* wait for ready */
	while (t->command != COMMAND_WAIT)
		pthread_cond_wait(&t->run_cond, &t->run_lock);
#endif
}

/*-------------------------------------------------------------------*/
static void stop_worker_thread(thread_data_t *t,
				uint32 is_master_thread)
{
	if (is_master_thread) {
		matrix_thread_free(t);
		return;
	}

	t->command = COMMAND_END;
#if defined(WIN32) || defined(_WIN64)
	SetEvent(t->run_event);
	WaitForSingleObject(t->thread_id, INFINITE);
	CloseHandle(t->thread_id);
	CloseHandle(t->run_event);
	CloseHandle(t->finish_event);
#else
	pthread_cond_signal(&t->run_cond);
	pthread_mutex_unlock(&t->run_lock);
	pthread_join(t->thread_id, NULL);
	pthread_cond_destroy(&t->run_cond);
	pthread_mutex_destroy(&t->run_lock);
#endif
}

/*-------------------------------------------------------------------*/
void packed_matrix_init(msieve_obj *obj,
			packed_matrix_t *p, la_col_t *A,
			uint32 nrows, uint32 max_nrows, uint32 start_row, 
			uint32 ncols, uint32 max_ncols, uint32 start_col, 
			uint32 num_dense_rows, uint32 first_block_size) {

	uint32 i, j, k;
	uint32 block_size;
	uint32 num_threads;
	uint32 num_nonzero;
	uint32 num_nonzero_per_thread;

	/* initialize */

	p->unpacked_cols = A;
	p->nrows = nrows;
	p->max_nrows = max_nrows;
	p->start_row = start_row;
	p->ncols = ncols;
	p->max_ncols = max_ncols;
	p->start_col = start_col;
	p->num_dense_rows = num_dense_rows;
	p->num_threads = 1;
	p->vsize = ncols;
#ifdef HAVE_MPI
	p->mpi_size = obj->mpi_size;
	p->mpi_nrows = obj->mpi_nrows;
	p->mpi_ncols = obj->mpi_ncols;
	p->mpi_la_row_rank = obj->mpi_la_row_rank;
	p->mpi_la_col_rank = obj->mpi_la_col_rank;
	p->mpi_la_row_grid = obj->mpi_la_row_grid;
	p->mpi_la_col_grid = obj->mpi_la_col_grid;
#endif

	if (max_nrows <= MIN_NROWS_TO_PACK)
		return;

	p->unpacked_cols = NULL;

	/* decide on the block size. We want to split the
	   cache into thirds; one third for x, one third for
	   b and the last third for streaming access to 
	   each block. If the matrix is small, adjust the
	   block size so that the matrix is divided 2.5
	   ways in each dimension.
	  
	   Note that the following does not compensate for
	   having multiple threads. If each thread gets a 
	   separate processor, no compensation is needed.
	   This could conceivably cause problems if multiple 
	   cores share the same cache, but multicore processors 
	   typically have pretty big caches anyway */

	block_size = obj->cache_size2 / (3 * sizeof(uint64));
	block_size = MIN(block_size, max_nrows / 2.5);
#ifdef LARGEBLOCKS
	block_size = MIN(block_size, 0x40000);	/* this may vary by CPU types */
#else
	block_size = MIN(block_size, 65536);
#endif
	if (block_size == 0)
		block_size = 32768;

	logprintf(obj, "using block size %u for "
			"processor cache size %u kB\n", 
				block_size, obj->cache_size2 / 1024);

	/* determine the number of threads to use */

	num_threads = obj->num_threads;
	if (num_threads < 2 || max_nrows < MIN_NROWS_TO_THREAD)
		num_threads = 1;
	p->num_threads = num_threads = MIN(num_threads, MAX_THREADS);
	p->vsize = ncols / p->num_threads;

	/* compute the number of nonzero elements in the submatrix
	   given to each thread; overestimate the number slightly
	   so that we don't have one thread with almost no columns */

	for (i = num_nonzero = 0; i < ncols; i++)
		num_nonzero += A[i].weight;
	num_nonzero_per_thread = num_nonzero / num_threads + 1000;

	/* divide the matrix into groups of columns, one group
	   per thread, and pack each group separately */

	for (i = j = k = num_nonzero = 0; i < ncols; i++) {

		num_nonzero += A[i].weight;

		if (i == ncols - 1 || num_nonzero >= num_nonzero_per_thread) {
			thread_data_t *t = p->thread_data + k;

			t->my_oid = k++;
			t->initial_cols = A;
			t->col_min = j;
			t->col_max = i;
			t->nrows_in = nrows;
			t->block_size = block_size;
			t->first_block_size = first_block_size;
			t->num_dense_rows = num_dense_rows;
			j = i + 1;
			num_nonzero = 0;
		}
	}

	/* update the number of threads, in case it's 
	   not what we estimated */

	p->num_threads = k;

	/* activate the threads one at a time. The last is the
	   master thread (i.e. not a thread at all). Each thread
	   packs its own portion of the matrix, so that memory
	   allocation is local on NUMA architectures */

	for (i = 0; i < p->num_threads - 1; i++)
		start_worker_thread(p->thread_data + i, 0);

	start_worker_thread(p->thread_data + i, 1);
}

/*-------------------------------------------------------------------*/
void packed_matrix_free(packed_matrix_t *p) {

	uint32 i;

	if (p->unpacked_cols) {
		la_col_t *A = p->unpacked_cols;
		for (i = 0; i < p->ncols; i++) {
			free(A[i].data);
			A[i].data = NULL;
		}
	}
	else {
		/* stop the worker threads; each will free
		   its own memory */

		for (i = 0; i < p->num_threads - 1; i++)
			stop_worker_thread(p->thread_data + i, 0);

		stop_worker_thread(p->thread_data + i, 1);
	}
}

/*-------------------------------------------------------------------*/
size_t packed_matrix_sizeof(packed_matrix_t *p) {

	uint32 i, j;
	size_t mem_use;

	/* account for the vectors used in the lanczos iteration */

	if (p->start_row + p->start_col == 0)
		mem_use = 7 * p->max_ncols;
	else
		mem_use = 7 * MAX(p->nrows, p->ncols);

	/* and for the matrix */

	if (p->unpacked_cols) {
		la_col_t *A = p->unpacked_cols;
		mem_use += p->ncols * (sizeof(la_col_t) +
				(p->num_dense_rows + 31) / 32);
		for (i = 0; i < p->ncols; i++) {
			mem_use += A[i].weight * sizeof(uint32);
		}
	}
	else {
		mem_use += MAX(p->nrows, p->ncols) * sizeof(uint64) *
				(p->num_threads - 1);

		for (i = 0; i < p->num_threads; i++) {
			thread_data_t *t = p->thread_data + i;

			mem_use += t->num_blocks * sizeof(packed_block_t) +
				   t->ncols * sizeof(uint64) *
					((t->num_dense_rows + 63) / 64);

			for (j = 0; j < t->num_blocks; j++) {
				packed_block_t *b = t->blocks + j;
#ifdef LARGEBLOCKS
				if (b->row_off) {
					mem_use += b->num_entries *
							3 * sizeof(uint16);
#else
				if (b->entries) {
					mem_use += b->num_entries *
							sizeof(entry_idx_t);
#endif
				}
				else {
					mem_use += (b->num_entries + 
						    2 * NUM_MEDIUM_ROWS) * 
							sizeof(uint16);
				}
			}
		}
	}
	return mem_use;
}

/*-------------------------------------------------------------------*/
void mul_MxN_Nx64(packed_matrix_t *A, uint64 *x, 
			uint64 *b, uint64 *scratch) {

	/* Multiply the vector x[] by the matrix A (stored
	   columnwise) and put the result in b[]. The MPI 
	   version needs a scratch array because MPI reduction
	   operations apparently cannot be performed in-place */

#ifdef HAVE_MPI
	if (A->mpi_size <= 1) {
#endif
		if (A->unpacked_cols)
			mul_unpacked(A, x, b);
		else
			mul_packed(A, x, b);
#ifdef HAVE_MPI
		return;
	}

	mul_packed(A, x, scratch);

	global_xor(scratch, b, A->nrows, A->mpi_nrows,
			A->mpi_la_row_rank, A->mpi_la_row_grid);
#endif
}

/*-------------------------------------------------------------------*/
void mul_sym_NxN_Nx64(packed_matrix_t *A, uint64 *x, 
			uint64 *b, uint64 *scratch) {

	/* Multiply x by A and write to scratch, then
	   multiply scratch by the transpose of A and
	   write to b. x may alias b, but the two must
	   be distinct from scratch */

#ifdef HAVE_MPI
	if (A->mpi_size <= 1) {
#endif
		if (A->unpacked_cols) {
			mul_unpacked(A, x, scratch);
			mul_trans_unpacked(A, scratch, b);
		}
		else {
			mul_packed(A, x, scratch);
			mul_trans_packed(A, scratch, b);
		}
#ifdef HAVE_MPI
		return;
	}

	mul_packed(A, x, scratch);

	/* make each MPI row combine its own part of A*x */

	global_xor(scratch, b, A->nrows, A->mpi_nrows,
			A->mpi_la_row_rank, A->mpi_la_row_grid);

	mul_trans_packed(A, b, scratch);

	/* make each MPI column combine its own part of A^T * A*x 
	   into the top row of MPI processes */

	global_xor(scratch, b, A->ncols, A->mpi_ncols, 
			A->mpi_la_col_rank, A->mpi_la_col_grid);

#endif
}
