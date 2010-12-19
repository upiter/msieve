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

typedef struct {
	uint64 offset;
	uint32 p;
} hash_entry_t;

typedef struct {
	uint64 start_offset;
	uint64 offset;
} hash_list_t;

/*------------------------------------------------------------------------*/
typedef struct {
	uint32 p;
	uint32 num_roots;
	uint32 pad;
	uint32 mont_w;
	uint64 mont_r;
	uint64 p2;
	hash_list_t roots[MAX_ROOTS];
} p_packed_t;

#define P_PACKED_HEADER_WORDS 4

typedef struct {
	uint32 num_p;
	uint32 num_roots;
	uint32 p_size;
	uint32 p_size_alloc;
	p_packed_t *curr;
	p_packed_t *packed_array;
} p_packed_var_t;

static void 
p_packed_init(p_packed_var_t *s)
{
	memset(s, 0, sizeof(p_packed_var_t));

	s->p_size_alloc = 100;
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
	s->num_p = s->num_roots = s->p_size = 0;
	s->curr = s->packed_array;
}

static p_packed_t * 
p_packed_next(p_packed_t *curr)
{
	return (p_packed_t *)((uint64 *)curr + 
			P_PACKED_HEADER_WORDS + 2 * curr->num_roots);
}

static void 
store_p_packed(uint32 p, uint32 num_roots, uint32 which_poly,
		mpz_t *roots, void *extra)
{
	uint32 i;
	p_packed_var_t *s = (p_packed_var_t *)extra;
	p_packed_t *curr;

	if (which_poly != 0) {
		printf("error: polynomial batches not supported\n");
		exit(-1);
	}

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
	curr->pad = 0;
	curr->num_roots = num_roots;
	for (i = 0; i < num_roots; i++)
		curr->roots[i].start_offset = gmp2uint64(roots[i]);
	curr->p2 = (uint64)p * p;
	curr->mont_w = montmul32_w((uint32)curr->p2);
	curr->mont_r = montmul64_r(curr->p2);

	s->num_p++;
	s->num_roots += num_roots;
	s->curr = p_packed_next(s->curr);
	s->p_size = ((uint8 *)s->curr - 
			(uint8 *)s->packed_array) / sizeof(uint64);
}

/*------------------------------------------------------------------------*/
static uint32
handle_special_q(msieve_obj *obj, hashtable_t *hashtable, 
		p_packed_var_t *hash_array, lattice_fb_t *L, 
		uint32 special_q, uint64 special_q_root,
		uint64 block_size, uint64 *inv_array)
{
	uint32 i, j;
	uint32 quit = 0;
	p_packed_t *tmp;
	uint32 num_entries = hash_array->num_p;
	uint64 special_q2 = (uint64)special_q * special_q;
	uint64 sieve_size = MIN((uint64)(-1),
				2 * L->poly->batch[0].sieve_size / special_q2);
	uint64 sieve_start = 0;
	uint32 num_blocks = 0;
	time_t curr_time;
	double elapsed;

	tmp = hash_array->packed_array;

	if (special_q == 1) {
		for (i = 0; i < num_entries; i++) {
			uint32 num_roots = tmp->num_roots;

			for (j = 0; j < num_roots; j++) {
				tmp->roots[j].offset =
						tmp->roots[j].start_offset;
			}
			tmp = p_packed_next(tmp);
		}
	}
	else {
		for (i = 0; i < num_entries; i++) {
			uint64 p2 = tmp->p2;
			uint32 num_roots = tmp->num_roots;
			uint32 p2_w = tmp->mont_w;
			uint64 qinv = inv_array[i];

			if (qinv == 0) {
				for (j = 0; j < num_roots; j++)
					tmp->roots[j].offset = (uint64)(-1);
				tmp = p_packed_next(tmp);
				continue;
			}

			for (j = 0; j < num_roots; j++) {
				uint64 proot = tmp->roots[j].start_offset;
				tmp->roots[j].offset = montmul64(
						mp_modsub_2(proot,
							special_q_root % p2,
						       	p2), qinv, p2, p2_w);
			}

			tmp = p_packed_next(tmp);
		}
	}

	while (sieve_start < sieve_size) {
		uint64 sieve_end = sieve_start + MIN(block_size,
						sieve_size - sieve_start);

		tmp = hash_array->packed_array;
		hashtable_reset(hashtable);

		for (i = 0; i < num_entries; i++) {

			uint32 num_roots = tmp->num_roots;

			for (j = 0; j < num_roots; j++) {
				uint64 offset = tmp->roots[j].offset;

				if (offset < sieve_end) {
					hash_entry_t *hit;
					hash_entry_t curr_entry;
					uint32 already_seen = 0;

					curr_entry.offset = offset;

					hit = (hash_entry_t *)hashtable_find(
							hashtable, &curr_entry,
							NULL, &already_seen);

					if (already_seen) {
						uint128 res;

						res.w[0] = (uint32)offset;
						res.w[1] = (uint32)(offset >> 32);
						res.w[2] = 0;
						res.w[3] = 0;

						handle_collision(L->poly, 0,
								hit->p, tmp->p, 
								special_q,
								special_q_root,
							       	res);
					}
					else {
						hit->p = tmp->p;
					}
					tmp->roots[j].offset = offset + tmp->p2;
				}
			}

			tmp = p_packed_next(tmp);
		}

		sieve_start = sieve_end;
		num_blocks++;

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING) {
			quit = 1;
			break;
		}
	}

	curr_time = time(NULL);
	elapsed = curr_time - L->start_time;
	if (elapsed > L->deadline)
		quit = 1;

//	printf("%u\n", num_blocks); 
	return quit;
}

/*------------------------------------------------------------------------*/
#define SPECIALQ_BATCH_SIZE 10

static void
batch_invert(uint32 *qlist, uint32 num_q, uint64 *invlist,
		uint32 p, uint64 p2_r, uint32 p2_w)
{
	uint32 i;
	uint64 q2[SPECIALQ_BATCH_SIZE];
	uint64 invprod;
	uint64 p2 = (uint64)p * p;

	invlist[0] = invprod = (uint64)qlist[0] * qlist[0];
	for (i = 1; i < num_q; i++) {
		q2[i] = (uint64)qlist[i] * qlist[i];
		invlist[i] = invprod = montmul64(invprod, q2[i], p2, p2_w);
	}

	invprod = mp_modinv_2(invprod, p2);
	invprod = montmul64(invprod, p2_r, p2, p2_w);
	for (i = num_q - 1; i; i--) {
		invlist[i] = montmul64(invprod, invlist[i-1], p2, p2_w);
		invprod = montmul64(invprod, q2[i], p2, p2_w);
	}
	invlist[i] = invprod;
}

/*------------------------------------------------------------------------*/
static uint32
sieve_specialq_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q, sieve_fb_t *sieve_p, 
		uint32 special_q_min, uint32 special_q_max, 
		uint32 p_min, uint32 p_max)
{
	uint32 i, j;
	uint32 quit = 0;
	p_packed_var_t specialq_array;
	p_packed_var_t hash_array;
	hashtable_t hashtable;
	uint32 num_p;
	uint64 block_size;
	uint64 *invtable = NULL;

	p_packed_init(&specialq_array);
	p_packed_init(&hash_array);
	hashtable_init(&hashtable, 3, 2);
	block_size = (uint64)p_min * p_min;

	sieve_fb_reset(sieve_p, p_min, p_max, 1, MAX_ROOTS);
	while (sieve_fb_next(sieve_p, L->poly, 
			store_p_packed, &hash_array) != P_SEARCH_DONE) {
		;
	}

	num_p = hash_array.num_p;
#if 1
	printf("aprogs: %u entries, %u roots\n", num_p, 
					hash_array.num_roots);
#endif
	/* handle trivial lattice */
	if (special_q_min == 1 && special_q_max == 1) {
		quit = handle_special_q(obj, &hashtable, &hash_array,
				L, 1, 0, block_size, NULL);
		goto finished;
	}

	invtable = (uint64 *)xmalloc(num_p * SPECIALQ_BATCH_SIZE * 
					sizeof(uint64));

	sieve_fb_reset(sieve_special_q, special_q_min, special_q_max, 
			1, MAX_ROOTS);
	while (1) {
		p_packed_t *qptr = specialq_array.packed_array;
		uint32 num_q;
		p_packed_t *tmp;
		uint64 *invtmp;
		mp_t qprod;
		uint32 batch_q[SPECIALQ_BATCH_SIZE];
		uint64 batch_q_inv[SPECIALQ_BATCH_SIZE];

		p_packed_reset(&specialq_array);
		while (sieve_fb_next(sieve_special_q, L->poly, store_p_packed,
					&specialq_array) != P_SEARCH_DONE) {
			if (specialq_array.num_p == SPECIALQ_BATCH_SIZE)
				break;
		}

		num_q = specialq_array.num_p;
		if (num_q == 0)
			break;
#if 1
		printf("special q: %u entries, %u roots\n", num_q, 
					specialq_array.num_roots);
#endif

		mp_clear(&qprod);
		qprod.nwords = qprod.val[0] = 1;

		for (i = 0, tmp = qptr; i < num_q; i++) {
			batch_q[i] = tmp->p;
			mp_mul_1(&qprod, tmp->p, &qprod);
			tmp = p_packed_next(tmp);
		}

		for (i = 0, tmp = hash_array.packed_array; i < num_p; i++) {

			if (mp_gcd_1(mp_mod_1(&qprod, tmp->p), tmp->p) == 1)
				batch_invert(batch_q, num_q, 
						batch_q_inv, tmp->p, 
						tmp->mont_r, tmp->mont_w);
			else
				memset(batch_q_inv, 0, sizeof(batch_q_inv));

			invtmp = invtable + i;
			for (j = 0; j < num_q; j++) {
				*invtmp = batch_q_inv[j];
				invtmp += num_p;
			}
			tmp = p_packed_next(tmp);
		}

		invtmp = invtable;
		for (i = 0; i < num_q; i++) {

			for (j = 0; j < qptr->num_roots; j++) {
				if (handle_special_q(obj,
						&hashtable, &hash_array,
						L, qptr->p, 
						qptr->roots[j].start_offset,
						block_size, invtmp)) {
					quit = 1;
					goto finished;
				}
			}

			qptr = p_packed_next(qptr);
			invtmp += num_p;
		}
	}

finished:
#if 1
	printf("hashtable: %u entries, %5.2lf MB\n", 
			hashtable_get_num(&hashtable),
			(double)hashtable_sizeof(&hashtable) / 1048576);
#endif
	free(invtable);
	hashtable_free(&hashtable);
	p_packed_free(&specialq_array);
	p_packed_free(&hash_array);
	return quit;
}

/*------------------------------------------------------------------------*/
uint32 
sieve_lattice_cpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max)
{
	uint32 quit;
	sieve_fb_t sieve_p;
	uint32 p_min, p_max;
	double p_size_max = L->poly->batch[0].p_size_max;
	uint32 degree = L->poly->degree;

	p_min = sqrt(p_size_max / special_q_max);
	p_max = p_min * params->p_scale;

	gmp_printf("coeff %Zd specialq %u - %u other %u - %u\n",
			L->poly->batch[0].high_coeff,
			special_q_min, special_q_max,
			p_min, p_max);

	sieve_fb_init(&sieve_p, L->poly, 
			100, 5000,
			1, degree,
		       	0);

	quit = sieve_specialq_64(obj, L,
			sieve_special_q, &sieve_p,
			special_q_min, special_q_max,
			p_min, p_max);

	sieve_fb_free(&sieve_p);
	return quit;
}
