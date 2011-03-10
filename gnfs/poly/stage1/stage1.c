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

/* main driver for stage 1 */

/*------------------------------------------------------------------------*/
static void
stage1_bounds_update(msieve_obj *obj, poly_search_t *poly)
{
	/* determine the parametrs for the collision search,
	   given one leading algebraic coefficient a_d */

	uint32 i, mult;
	uint32 degree = poly->degree;
	double N = mpz_get_d(poly->N);
	double high_coeff = mpz_get_d(poly->high_coeff);
	double skewness_min, m0;
	double coeff_max, p_size_max, cutoff;
	double special_q_min, special_q_max;
	uint32 num_pieces;
#ifndef HAVE_CUDA
	double hash_iters;
#endif

	/* we don't know the optimal skewness for this polynomial
	   but at least can bound the skewness. The value of the
	   third-highest coefficient is from Kleinjung's 2006
	   poly selection algorithm as published in Math. Comp. */

	switch (degree) {
	case 4:
		mult = 4 * 4 * 4 * 4;
		skewness_min = sqrt(pow(N / high_coeff, 1./4.) /
					poly->norm_max);
		coeff_max = poly->norm_max;
		break;

	case 5:
		mult = 5 * 5 * 5 * 5 * 5;
		skewness_min = pow(pow(N / high_coeff, 1./5.) /
					poly->norm_max, 2./3.);
		coeff_max = poly->norm_max / sqrt(skewness_min);
		break;

	case 6:
		mult = 6 * 6 * 6 * 6 * 6 * 6;
		skewness_min = sqrt(pow(N / high_coeff, 1./6.) /
					poly->norm_max);
		coeff_max = poly->norm_max / skewness_min;
		break;

	default:
		printf("error: unexpected poly degree %d\n", degree);
		exit(-1);
	}

	/* we perform the collision search on a transformed version
	   of N and the low-order rational coefficient m0. In the
	   transformed coordinates, a_d is 1 and a_{d-1} is 0. When
	   a hit is found, we undo the transformation to recover
	   the correction to m0 that makes the new polynomial 'work' */

	mpz_mul_ui(poly->trans_N, poly->N, (mp_limb_t)mult);
	for (i = 0; i < degree - 1; i++)
		mpz_mul(poly->trans_N, poly->trans_N, poly->high_coeff);

	mpz_root(poly->trans_m0, poly->trans_N, (mp_limb_t)degree);

	mpz_tdiv_q(poly->m0, poly->N, poly->high_coeff);
	mpz_root(poly->m0, poly->m0, (mp_limb_t)degree);
	m0 = mpz_get_d(poly->m0);

	/* for leading rational coefficient l, the sieve size
	   will be l^2*cutoff. This is based on the norm limit
	   given above, and largely determines how difficult
	   it will be to find acceptable collisions in the
	   search */

	cutoff = coeff_max / m0 / degree;

#ifdef HAVE_CUDA
	/* the GPU code doesn't care how large the sieve 
	   size is, so favor smaller special-q and try 
	   to make the range of other rational factors large */

#define SPECIAL_Q_SCALE 16
	p_size_max = MIN(coeff_max / skewness_min,
			 (double)((uint32)(-1) / SPECIAL_Q_SCALE) *
				 ((uint32)(-1) / P_SCALE - 1) *
				 ((uint32)(-1) / P_SCALE - 1));

	special_q_min = MIN((uint32)(-1) / SPECIAL_Q_SCALE,
				p_size_max / 7200000 / 7200000);
	if (special_q_min > 1) {
		special_q_min = MAX(special_q_min, 251);
		special_q_max = special_q_min * SPECIAL_Q_SCALE;
	}
	else {
		special_q_min = special_q_max = 1;
	}

	/* very large problems are split into pieces, and we
	   randomly choose one to search. This allows multiple
	   machines to search the same a_d */

	num_pieces = (special_q_max - special_q_min)
			/ (log(special_q_max) - 1)
			/ 1000;
	num_pieces = MIN(num_pieces, 2000);

#else
	/* the CPU code is different; its runtime is
	   directly proportional to the sieve size. So 
	   choose the special-q size to limit the number 
	   of times the CPU hashtable code must run. 
	   The parametrization is chosen to favor larger
	   special q for larger inputs, adjusted implictly 
	   for the polynomial degree */

	/* hash_iters determines how often the hashtable code
	   will run. This is the most important factor in the
	   speed of the hashtable code, and is used to control
	   the limits for the special-q. A larger value of
	   hash_iters will result in smaller special-q being
	   selected and thus more sieve offsets being hashed.
	   Up to a point, smaller hash_iters will be faster,
	   but values which are far too small may cause
	   performance to degrade slightly as the 'birthday
	   effect' is reduced. Somewhere, a 'sweet spot'
	   exists, but this depends greatly on the size of
	   problem being sieved. The values suggested below
	   appear to yield decent results for problems
	   currently supported by msieve */

	if (degree < 5)
		hash_iters = 1000; /* be generous for deg4 */
	else
		hash_iters = 50; /* seems reasonable */

	/* we need to be sure that the parameters with the
	   specified value of hash_iters will 'work'. There
	   are at least two things to check:

		(1) l/special_q must be small enough to keep
		    the hashtable size manageable
		(2) 2*(l/special_q)^2*cutoff must fit in a
		    64bit unsigned integer 

	   these conditions kick in only for the largest
	   problems, and this is just a way to keep the
	   search from blowing up on us unexpectedly */

	/* first limit hash_iters to keep l/special_q small.
	   the size of aprogs p will be at most about 2^27,
	   though this is deliberately over-estimated */

	hash_iters = MIN(hash_iters, (double)((uint32)1 << 27) *
					     ((uint32)1 << 27) *
					     cutoff);

	/* next limit hash_iters to keep 2*(l/special_q)^2*cutoff
	   small. Again, we over-estimate a little bit */

	hash_iters = MIN(hash_iters, sqrt((double)(uint64)(-1) *
						  cutoff));

#define SPECIAL_Q_SCALE 5
	/* the factor of 2 below comes from the fact that the
	   total length of the line sieved is 2*sieve_size, since
	   both sides of the origin are sieved at once */

	p_size_max = coeff_max / skewness_min;
	special_q_min = 2 * P_SCALE * P_SCALE * cutoff * coeff_max
			/ skewness_min / hash_iters;

	/* special_q must be <2^32. If it is too big, we can
	   reduce the problem size a bit further to compensate */

	if (special_q_min > (uint32)(-1) / SPECIAL_Q_SCALE) {

		p_size_max *= (uint32)(-1) / special_q_min / SPECIAL_Q_SCALE;
		special_q_min = (uint32)(-1) / SPECIAL_Q_SCALE;
	}

	if (special_q_min > 1) {
		/* very small ranges of special q might be empty, so
		   impose a limit on the minimum size */

		special_q_min = MAX(special_q_min, 11);
		special_q_max = special_q_min * SPECIAL_Q_SCALE;
	}
	else {
		/* only trivial lattice */

		special_q_min = special_q_max = 1;
	}

	num_pieces = (special_q_max - special_q_min)
			/ (log(special_q_max) - 1)
			/ 110000;
	num_pieces = MIN(num_pieces,
				sqrt(p_size_max / special_q_max) /
				(log(p_size_max / special_q_max) / 2 - 1) /
				(P_SCALE / (P_SCALE - 1)) /
				10);

#endif

	if (num_pieces > 1) { /* randomize the special_q range */

		double piece_ratio = pow((double)special_q_max / special_q_min,
					 (double)1 / num_pieces);
		uint32 piece = get_rand(&obj->seed1,
					&obj->seed2) % num_pieces;

		printf("randomizing rational coefficient: "
			"using piece #%u of %u\n",
			piece + 1, num_pieces);

		special_q_min *= pow(piece_ratio, (double)piece);
		special_q_max = special_q_min * piece_ratio;
	}

	poly->special_q_min = (uint32)special_q_min;
	poly->special_q_max = (uint32)special_q_max;
	poly->special_q_fb_max = MIN((uint32)special_q_max, 100000);

	poly->coeff_max = coeff_max;
	poly->p_size_max = p_size_max;

	/* Kleinjung's improved algorithm computes a 'correction'
	   to m0, and the new m0 will cause a_{d-2} to be small enough
	   if it is smaller than sieve_size */

	poly->sieve_size = p_size_max * p_size_max * cutoff;
	mpz_set_d(poly->mp_sieve_size, poly->sieve_size);
}

/*------------------------------------------------------------------------*/
static void
poly_search_init(poly_search_t *poly, poly_stage1_t *data)
{
	mpz_init_set(poly->N, data->gmp_N);
	mpz_init(poly->high_coeff);
	mpz_init(poly->trans_N);
	mpz_init(poly->trans_m0);
	mpz_init(poly->mp_sieve_size);
	mpz_init(poly->m0);
	mpz_init(poly->p);
	mpz_init(poly->tmp1);
	mpz_init(poly->tmp2);
	mpz_init(poly->tmp3);
	mpz_init(poly->tmp4);
	mpz_init(poly->tmp5);

	mpz_init_set(poly->gmp_high_coeff_begin, 
			data->gmp_high_coeff_begin);
	mpz_init_set(poly->gmp_high_coeff_end, 
			data->gmp_high_coeff_end);

	poly->degree = data->degree;
	poly->norm_max = data->norm_max;
	poly->callback = data->callback;
	poly->callback_data = data->callback_data;

#ifdef HAVE_CUDA
	CUDA_TRY(cuCtxCreate(&poly->gpu_context, 
			CU_CTX_BLOCKING_SYNC,
			poly->gpu_info->device_handle))

	/* load two GPU kernels, one for special-q
	   collision search and one for ordinary collision 
	   search */

	CUDA_TRY(cuModuleLoad(&poly->gpu_module_sq, 
				"stage1_core_sq.ptx"))
	CUDA_TRY(cuModuleLoad(&poly->gpu_module_nosq, 
				"stage1_core_nosq.ptx"))
#endif
}

/*------------------------------------------------------------------------*/
static void
poly_search_free(poly_search_t *poly)
{
	mpz_clear(poly->N);
	mpz_clear(poly->high_coeff);
	mpz_clear(poly->trans_N);
	mpz_clear(poly->trans_m0);
	mpz_clear(poly->mp_sieve_size);
	mpz_clear(poly->m0);
	mpz_clear(poly->p);
	mpz_clear(poly->tmp1);
	mpz_clear(poly->tmp2);
	mpz_clear(poly->tmp3);
	mpz_clear(poly->tmp4);
	mpz_clear(poly->tmp5);
	mpz_clear(poly->gmp_high_coeff_begin);
	mpz_clear(poly->gmp_high_coeff_end);
#ifdef HAVE_CUDA
	CUDA_TRY(cuCtxDestroy(poly->gpu_context)) 
#endif
}

/*------------------------------------------------------------------------*/
static void
search_coeffs(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	uint32 i, j, p;
	uint32 digits = mpz_sizeinbase(poly->N, 10);
	uint32 deadline_per_coeff;
	double start_time = get_cpu_time();

	/* determine the CPU time limit; I have no idea if
	   the following is appropriate */

	if (digits <= 100)
		deadline_per_coeff = 5;
	else if (digits <= 105)
		deadline_per_coeff = 20;
	else if (digits <= 110)
		deadline_per_coeff = 30;
	else if (digits <= 120)
		deadline_per_coeff = 50;
	else if (digits <= 130)
		deadline_per_coeff = 100;
	else if (digits <= 140)
		deadline_per_coeff = 200;
	else if (digits <= 150)
		deadline_per_coeff = 400;
	else if (digits <= 175)
		deadline_per_coeff = 800;
	else if (digits <= 200)
		deadline_per_coeff = 1600;
	else
		deadline_per_coeff = 3200;

	printf("deadline: %u seconds per coefficient\n", deadline_per_coeff);

	/* set up lower limit on a_d */

	mpz_sub_ui(poly->high_coeff, poly->gmp_high_coeff_begin, (mp_limb_t)1);
	mpz_fdiv_q_ui(poly->high_coeff, poly->high_coeff, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);
	mpz_mul_ui(poly->high_coeff, poly->high_coeff, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);

	while (1) {
		/* increment a_d */

		mpz_add_ui(poly->high_coeff, poly->high_coeff,
				(mp_limb_t)HIGH_COEFF_MULTIPLIER);

		if (mpz_cmp(poly->high_coeff, poly->gmp_high_coeff_end) > 0)
			break;

		mpz_divexact_ui(poly->tmp1, poly->high_coeff, 
					(mp_limb_t)HIGH_COEFF_MULTIPLIER);

		/* trial divide the a_d and skip it if it
		   has any large prime factors */

		for (i = p = 0; i < PRECOMPUTED_NUM_PRIMES; i++) {
			p += prime_delta[i];

			if (p > HIGH_COEFF_PRIME_LIMIT)
				break;

			for (j = 0; j < HIGH_COEFF_POWER_LIMIT; j++) {
				if (mpz_divisible_ui_p(poly->tmp1, 
						(mp_limb_t)p))
					mpz_divexact_ui(poly->tmp1, 
						poly->tmp1, (mp_limb_t)p);
				else
					break;
			}
		}
		if (mpz_cmp_ui(poly->tmp1, (mp_limb_t)1))
			continue;

		/* a_d is okay, search it */

		stage1_bounds_update(obj, poly);
		sieve_lattice(obj, poly, deadline_per_coeff);

		if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
			break;

		if (deadline) {
			double curr_time = get_cpu_time();
			double elapsed = curr_time - start_time;

			if (elapsed > deadline)
				break;
		}
	}
}

/*------------------------------------------------------------------------*/
void
poly_stage1_init(poly_stage1_t *data,
		 stage1_callback_t callback, void *callback_data)
{
	memset(data, 0, sizeof(poly_stage1_t));
	mpz_init_set_ui(data->gmp_N, (mp_limb_t)0);
	mpz_init_set_ui(data->gmp_high_coeff_begin, (mp_limb_t)0);
	mpz_init_set_ui(data->gmp_high_coeff_end, (mp_limb_t)0);
	data->callback = callback;
	data->callback_data = callback_data;
}

/*------------------------------------------------------------------------*/
void
poly_stage1_free(poly_stage1_t *data)
{
	mpz_clear(data->gmp_N);
	mpz_clear(data->gmp_high_coeff_begin);
	mpz_clear(data->gmp_high_coeff_end);
}

/*------------------------------------------------------------------------*/
void
poly_stage1_run(msieve_obj *obj, poly_stage1_t *data)
{
	/* pass external configuration in and run the search */

	poly_search_t poly;
#ifdef HAVE_CUDA
	gpu_config_t gpu_config;

	gpu_init(&gpu_config);
	if (gpu_config.num_gpu == 0) {
		printf("error: no CUDA-enabled GPUs found\n");
		exit(-1);
	}
	if (obj->which_gpu >= (uint32)gpu_config.num_gpu) {
		printf("error: GPU %u does not exist "
			"or is not CUDA-enabled\n", obj->which_gpu);
		exit(-1);
	}
	logprintf(obj, "using GPU %u (%s)\n", obj->which_gpu,
			gpu_config.info[obj->which_gpu].name);

	poly.gpu_info = gpu_config.info + obj->which_gpu; 
#endif

	poly_search_init(&poly, data);

	search_coeffs(obj, &poly, data->deadline);

	poly_search_free(&poly);
}
