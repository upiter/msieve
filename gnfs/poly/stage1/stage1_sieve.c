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

/* structures for storing arithmetic progressions. Rational
   leading coeffs of NFS polynomials are assumed to be the 
   product of two groups of factors p, each of size at most 64
   bits (32 bits is the maximum for degree 4, and is enough 
   for 512-bit factorizations using degree 5), and candidates 
   must satisfy a condition modulo p^2 */

#define MAX_P ((uint64)(-1))

typedef struct {
	uint32 bits; /* in leading rational coeff */
	double p_scale;
	uint32 num_pieces; /* for randomization */
	uint32 small_fb_max;
	uint32 large_fb_max;
} sieve_fb_param_t;

static const sieve_fb_param_t sieve_fb_params[] = {

	/* target composite small_p having 2 factors */
	{ 40, 1.3,  1,   100,  10000},
	{ 48, 1.2,  1,   250,  25000},
	{ 56, 1.1,  1,   750,  50000},
	{ 64, 1.1,  5,  3500, 150000},
	{ 72, 1.1, 10, 15000, 500000},

	/* . . . . 3 factors */
	{ 80, 1.1, 25,  1500,  50000},
	{ 88, 1.1, 50,  3500, 100000},
	{ 96, 1.1, 50,  7500, 250000},
	{104, 1.1, 50, 15000, 500000},

	/* . . . . 4 factors */
	{116, 1.1, 50,  5000, 100000},
	{128, 1.1, 50, 10000, 250000},
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

/*------------------------------------------------------------------------*/
void
handle_collision(poly_search_t *poly, uint32 which_poly,
		uint64 p, uint128 proot, uint128 res, uint64 q)
{
	curr_poly_t *c = poly->batch + which_poly;

	uint64_2gmp(p, poly->tmp1);
	uint64_2gmp(q, poly->tmp2);

	mpz_gcd(poly->tmp3, poly->tmp1, c->high_coeff);
	if (mpz_cmp_ui(poly->tmp3, 1))
		return;
	mpz_gcd(poly->tmp3, poly->tmp2, c->high_coeff);
	if (mpz_cmp_ui(poly->tmp3, 1))
		return;
	mpz_gcd(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, 1))
		return;

	mpz_import(poly->tmp3, 4, -1, sizeof(uint32), 0, 0, &proot);
	mpz_import(poly->tmp4, 4, -1, sizeof(uint32), 0, 0, &res);

	mpz_mul(poly->p, poly->tmp1, poly->tmp2);
	mpz_mul(poly->tmp1, poly->tmp1, poly->tmp1);
	mpz_addmul(poly->tmp3, poly->tmp1, poly->tmp4);
	mpz_sub(poly->tmp3, poly->tmp3, c->mp_sieve_size);
	mpz_add(poly->m0, c->trans_m0, poly->tmp3);

	/* check */
	gmp_printf("poly %2u p %.0lf q %.0lf coeff %Zd\n", 
			which_poly, (double)p, (double)q, poly->p);

	mpz_pow_ui(poly->tmp1, poly->m0, (mp_limb_t)poly->degree);
	mpz_mul(poly->tmp2, poly->p, poly->p);
	mpz_sub(poly->tmp1, c->trans_N, poly->tmp1);
	mpz_tdiv_r(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, (mp_limb_t)0)) {
		printf("crap\n");
		return;
	}

	mpz_mul_ui(poly->tmp1, c->high_coeff, (mp_limb_t)poly->degree);
	mpz_tdiv_qr(poly->m0, poly->tmp2, poly->m0, poly->tmp1);
	mpz_invert(poly->tmp3, poly->tmp1, poly->p);

	mpz_sub(poly->tmp4, poly->tmp3, poly->p);
	if (mpz_cmpabs(poly->tmp3, poly->tmp4) < 0)
		mpz_set(poly->tmp4, poly->tmp3);

	mpz_sub(poly->tmp5, poly->tmp2, poly->tmp1);
	if (mpz_cmpabs(poly->tmp2, poly->tmp5) > 0)
		mpz_add_ui(poly->m0, poly->m0, (mp_limb_t)1);
	else
		mpz_set(poly->tmp5, poly->tmp2);

	mpz_addmul(poly->m0, poly->tmp4, poly->tmp5);

	poly->callback(c->high_coeff, poly->p, poly->m0, 
			c->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	uint32 i;
	lattice_fb_t L;
	sieve_fb_t sieve_small, sieve_large;
	uint64 small_p_min, small_p_max;
	uint64 large_p_min, large_p_max;
	uint32 small_fb_max, large_fb_max;
	uint32 num_pieces;
	uint32 bits;
	double p_scale;
	uint32 degree = poly->degree;
	uint32 max_roots;
	curr_poly_t *middle_poly;
	curr_poly_t *last_poly;

	middle_poly = poly->batch + poly->num_poly / 2;
	last_poly = poly->batch + poly->num_poly - 1;

	if ((poly->degree == 4 && 
	    middle_poly->p_size_max >= MAX_P) ||
	    middle_poly->p_size_max >= (double)MAX_P * MAX_P) {
		printf("error: rational leading coefficient is "
			"too large at %le (%0.3f bits)\n",
	 		 middle_poly->p_size_max, 
			 log(middle_poly->p_size_max)/ M_LN2);
		exit(-1);
	}

	bits = log(middle_poly->p_size_max) / M_LN2;
	for (i = 0; i < NUM_SIEVE_FB_PARAMS; i++) {
		if (bits < sieve_fb_params[i].bits)
			break;
	}
	if (i == NUM_SIEVE_FB_PARAMS) {
		printf("error: no factor base parameters for "
			"%u bit leading rational coefficient\n", bits);
		exit(-1);
	}
	p_scale = sieve_fb_params[i].p_scale;
	small_fb_max = sieve_fb_params[i].small_fb_max;
	large_fb_max = sieve_fb_params[i].large_fb_max;
	num_pieces = sieve_fb_params[i].num_pieces;

	large_p_min = sqrt(middle_poly->p_size_max);
	if (large_p_min >= MAX_P / p_scale)
		large_p_max = MAX_P - 1;
	else
		large_p_max = p_scale * large_p_min;

	small_p_min = large_p_min / p_scale;
	small_p_max = large_p_min - 1;

	gmp_printf("coeff %Zd-%Zd"
		   " %" PRIu64 " %" PRIu64 
		   " %" PRIu64 " %" PRIu64 "\n",
			poly->batch[0].high_coeff,
			last_poly->high_coeff,
			small_p_min, small_p_max,
			large_p_min, large_p_max);

	max_roots = degree;
	if (degree == 5)
		max_roots = 1;

	sieve_fb_init(&sieve_small, poly, 
			5, small_fb_max, 
			1, max_roots);
	sieve_fb_init(&sieve_large, poly, 
			small_fb_max + 1, large_fb_max, 
			1, max_roots);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;
#ifdef HAVE_CUDA
	L.gpu_info = poly->gpu_info;
#endif

	while (1) {
		uint32 done = 1;
		uint64 small_p_min2 = small_p_min;
		uint64 small_p_max2 = small_p_max;
		uint64 large_p_min2 = large_p_min;
		uint64 large_p_max2 = large_p_max;

		if (num_pieces > 1) {
			uint32 small_piece;
			uint32 large_piece;
			
			small_piece = get_rand(&obj->seed1, 
					&obj->seed2) % num_pieces;
			small_p_min2 = small_p_min + small_piece *
						((small_p_max - small_p_min) /
						num_pieces);
			small_p_max2 = small_p_min + (small_piece + 1) *
						((small_p_max - small_p_min) /
						num_pieces);

			large_piece = get_rand(&obj->seed1, 
					&obj->seed2) % num_pieces;
			large_p_min2 = large_p_min + large_piece *
						((large_p_max - large_p_min) /
						num_pieces);
			large_p_max2 = large_p_min + (large_piece + 1) *
						((large_p_max - large_p_min) /
						num_pieces);
		}

#ifdef HAVE_CUDA
		if (degree == 4) {
			if (large_p_max2 < ((uint64)1 << 24))
				L.gpu_module = poly->gpu_module48;
			else
				L.gpu_module = poly->gpu_module64;
		}
		else if (degree == 5) {
			if (large_p_max2 < ((uint64)1 << 24))
				L.gpu_module = poly->gpu_module48;
			else if (large_p_max2 < ((uint64)1 << 32))
				L.gpu_module = poly->gpu_module64;
			else if (large_p_max2 < ((uint64)1 << 36))
				L.gpu_module = poly->gpu_module72;
			else if (large_p_max2 < ((uint64)1 << 48))
				L.gpu_module = poly->gpu_module96;
			else
				L.gpu_module = poly->gpu_module128;
		}
		else {	/* degree 6 */
			if (large_p_max2 < ((uint64)1 << 24))
				L.gpu_module = poly->gpu_module48;
			else if (large_p_max2 < ((uint64)1 << 32))
				L.gpu_module = poly->gpu_module64;
			else if (large_p_max2 < ((uint64)1 << 36))
				L.gpu_module = poly->gpu_module72;
			else if (large_p_max2 < ((uint64)1 << 48))
				L.gpu_module = poly->gpu_module96;
			else
				L.gpu_module = poly->gpu_module128;
		}

		CUDA_TRY(cuModuleGetFunction(&L.gpu_kernel, 
				L.gpu_module, "sieve_kernel"))
		if (degree != 5)
			CUDA_TRY(cuModuleGetGlobal(&L.gpu_p_array, 
				NULL, L.gpu_module, "pbatch"))
#endif

		if (degree == 4) {
			/* bounds may have grown too large */
			if (large_p_max2 >= ((uint64)1 << 32))
				break;

			done = sieve_lattice_deg46_64(obj, &L,
					&sieve_small, &sieve_large,
					(uint32)small_p_min2, 
					(uint32)small_p_max2,
					(uint32)large_p_min2, 
					(uint32)large_p_max2);
		}
		else if (degree == 5) {
			if (large_p_max2 < ((uint64)1 << 32)) {
				done = sieve_lattice_deg5_64(obj, &L,
					&sieve_small, &sieve_large,
					(uint32)small_p_min2, 
					(uint32)small_p_max2,
					(uint32)large_p_min2, 
					(uint32)large_p_max2);
			}
			else if (large_p_max2 < ((uint64)1 << 36)) {
				done = sieve_lattice_deg5_96(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
			else if (large_p_max2 < ((uint64)1 << 48)) {
				done = sieve_lattice_deg5_96(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
			else {
				done = sieve_lattice_deg5_128(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
		}
		else {	/* degree 6 */
			if (large_p_max2 < ((uint64)1 << 32)) {
				done = sieve_lattice_deg46_64(obj, &L,
					&sieve_small, &sieve_large,
					(uint32)small_p_min2, 
					(uint32)small_p_max2,
					(uint32)large_p_min2, 
					(uint32)large_p_max2);
			}
			else if (large_p_max2 < ((uint64)1 << 36)) {
				done = sieve_lattice_deg6_96(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
			else if (large_p_max2 < ((uint64)1 << 48)) {
				done = sieve_lattice_deg6_96(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
			else {
				done = sieve_lattice_deg6_128(obj, &L,
					&sieve_small, &sieve_large,
					small_p_min2, small_p_max2,
					large_p_min2, large_p_max2);
			}
		}

		if (done)
			break;

		small_p_max = small_p_min - 1;
		small_p_min = small_p_min / p_scale;
		if (small_p_min < small_fb_max)
			break;

		if (middle_poly->p_size_max / small_p_max >= MAX_P)
			break;
		large_p_min = middle_poly->p_size_max / small_p_max;

		if (middle_poly->p_size_max / small_p_min >= MAX_P)
			large_p_max = MAX_P - 1;
		else
			large_p_max = middle_poly->p_size_max / small_p_min;
	}

	sieve_fb_free(&sieve_small);
	sieve_fb_free(&sieve_large);
}
