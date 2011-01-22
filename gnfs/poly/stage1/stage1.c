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

/*------------------------------------------------------------------------*/
static void
stage1_bounds_update(msieve_obj *obj, poly_search_t *poly)
{
	uint32 i, mult;
	uint32 degree = poly->degree;
	double N = mpz_get_d(poly->N);
	double high_coeff = mpz_get_d(poly->high_coeff);
	double skewness_min, m0;
	double coeff_max, p_size_max;
	uint32 special_q_min, special_q_max;
	uint32 num_pieces;
#ifndef HAVE_CUDA
	uint32 max_blocks;
	double tmp;
#endif

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

	mpz_mul_ui(poly->trans_N, poly->N, (mp_limb_t)mult);
	for (i = 0; i < degree - 1; i++)
		mpz_mul(poly->trans_N, poly->trans_N, poly->high_coeff);

	mpz_root(poly->trans_m0, poly->trans_N, (mp_limb_t)degree);

	mpz_tdiv_q(poly->m0, poly->N, poly->high_coeff);
	mpz_root(poly->m0, poly->m0, (mp_limb_t)degree);
	m0 = mpz_get_d(poly->m0);

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

	if (degree < 5)
		max_blocks = 1000;
	else
		max_blocks = 50;

#define SPECIAL_Q_SCALE 5
	tmp = 2 * coeff_max * coeff_max / skewness_min
		/ m0 / degree / max_blocks;
	tmp = MAX(tmp, coeff_max / skewness_min /
			((double)((uint32)1 << 27) *
				 ((uint32)1 << 27) /
				 max_blocks));
	tmp *= P_SCALE * P_SCALE;
	special_q_min = MIN((uint32)(-1) / SPECIAL_Q_SCALE, tmp);
	if (special_q_min > 1) {
		special_q_min = MAX(special_q_min, 11);
		special_q_max = special_q_min * SPECIAL_Q_SCALE;
	}
	else {
		special_q_min = special_q_max = 1;
	}

	tmp = MAX(1, tmp / special_q_min);
	p_size_max = coeff_max / skewness_min / tmp;

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

	poly->special_q_min = special_q_min;
	poly->special_q_max = special_q_max;
	poly->special_q_fb_max = MIN(special_q_max, 100000);

	poly->coeff_max = coeff_max;
	poly->p_size_max = p_size_max;
	poly->sieve_size = coeff_max * p_size_max * p_size_max
				/ m0 / degree;
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

	mpz_sub_ui(poly->high_coeff, poly->gmp_high_coeff_begin, (mp_limb_t)1);
	mpz_fdiv_q_ui(poly->high_coeff, poly->high_coeff, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);
	mpz_mul_ui(poly->high_coeff, poly->high_coeff, 
			(mp_limb_t)HIGH_COEFF_MULTIPLIER);

	while (mpz_cmp(poly->high_coeff, poly->gmp_high_coeff_end) < 0) {

		mpz_add_ui(poly->high_coeff, poly->high_coeff,
				(mp_limb_t)HIGH_COEFF_MULTIPLIER);

		mpz_divexact_ui(poly->tmp1, poly->high_coeff, 
					(mp_limb_t)HIGH_COEFF_MULTIPLIER);
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
