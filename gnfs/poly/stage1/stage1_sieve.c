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

#define MIN_SPECIAL_Q 17

static const sieve_fb_param_t sieve_fb_params[] = {

#ifdef HAVE_CUDA
	{ 40, 1.5,    1, 100,         1,          1},
	{ 48, 1.3,    1,  25,         1,       1500},
	{ 56, 1.2,   25,  10,      1000,     250000},
	{ 64, 1.2,  100,   5,    150000,   15000000},
	{ 72, 1.1,  500,   5,  10000000,  500000000},
	{ 80, 1.1, 2500,   5, 250000000, 2500000000},
#else
	{ 40, 2.0,    1,   10},
	{ 48, 2.0,    1,   20},
	{ 56, 1.7,    1,   80},
	{ 64, 1.5,    1,  200},
	{ 72, 1.3,    1,  200},
	{ 80, 1.2,    1,  200},
	{ 88, 1.1,    1,  200},
#endif
};

#define NUM_SIEVE_FB_PARAMS (sizeof(sieve_fb_params) / \
				sizeof(sieve_fb_params[0]))

/*------------------------------------------------------------------------*/
static void
get_poly_params(double bits, sieve_fb_param_t *params)
{
	uint32 i;
	const sieve_fb_param_t *low, *high;
	double j, k, dist, max_bits;

	if (bits < sieve_fb_params[0].bits) {
		*params = sieve_fb_params[0];

		return;
	}

	max_bits = sieve_fb_params[NUM_SIEVE_FB_PARAMS - 1].bits;
	if (bits >= max_bits) {
#ifdef HAVE_CUDA
		if (bits > max_bits + 5) {
			printf("error: no factor base parameters for "
				"%.0lf bit leading rational "
				"coefficient\n", bits + 0.5);
			exit (-1);
		}
#endif
		*params = sieve_fb_params[NUM_SIEVE_FB_PARAMS - 1];

		return;
	}

	for (i = 0; i < NUM_SIEVE_FB_PARAMS - 1; i++) {
		if (bits < sieve_fb_params[i+1].bits)
			break;
	}

	low = &sieve_fb_params[i];
	high = &sieve_fb_params[i+1];
	dist = high->bits - low->bits;
	j = bits - low->bits;
	k = high->bits - bits;

	params->bits = bits;
	params->p_scale = (low->p_scale * k +
				high->p_scale * j) / dist;
	params->num_pieces = exp((log(low->num_pieces) * k +
					log(high->num_pieces) * j) / dist);

#ifdef HAVE_CUDA
	params->max_diverge = (low->max_diverge * k +
				high->max_diverge * j) / dist;
	params->special_q_min = exp((log(low->special_q_min) * k +
					log(high->special_q_min) * j) / dist);
	params->special_q_max = exp((log(low->special_q_max) * k +
					log(high->special_q_max) * j) / dist);
#else
	params->num_blocks = (low->num_blocks * k +
				high->num_blocks * j) / dist;
#endif
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	lattice_fb_t L;
	sieve_fb_t sieve_special_q;
	uint32 special_q_min, special_q_max;
	uint32 num_pieces;
	sieve_fb_param_t params;
	double bits;
	curr_poly_t *middle_poly = poly->batch + poly->num_poly / 2;
	curr_poly_t *last_poly = poly->batch + poly->num_poly - 1;

	printf("p = %.2lf bits, sieve = %.2lf bits\n",
			log(middle_poly->p_size_max) / M_LN2,
			log(middle_poly->sieve_size) / M_LN2);

#ifdef HAVE_CUDA
	/* the GPU code doesn't care how large the sieve 
	   size is, so favor smaller special-q and try 
	   to make the range of other rational factors large */

	bits = log(middle_poly->p_size_max) / M_LN2;
	get_poly_params(bits, &params);
	special_q_min = params.special_q_min;
	special_q_max = params.special_q_max;
#else
	/* the CPU code is different; its runtime is
	   directly proportional to the sieve size. So 
	   choose the special-q size to fix the number 
	   of times the CPU hashtable code will run.
	   The parametrization is chosen to favor larger
	   special q, adjusted implictly for the polynomial 
	   degree */

	bits = log(middle_poly->sieve_size) / M_LN2;
	get_poly_params(bits, &params);
	special_q_min = MIN((uint32)(-1), 
				middle_poly->sieve_size / 
				(middle_poly->p_size_max * 
				 params.num_blocks));
	special_q_min = MAX(special_q_min, 1);
	if (poly->degree != 5)
		special_q_min *= 5;
			
	special_q_max = special_q_min * params.p_scale;
#endif

	num_pieces = params.num_pieces;

	sieve_fb_init(&sieve_special_q, poly,
			5, MIN(10000, special_q_max),
			1, poly->degree,
			1);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;

	gmp_printf("coeff %Zd-%Zd specialq %u - %u\n",
		   poly->batch[0].high_coeff, last_poly->high_coeff,
		   special_q_min, special_q_max);

	if (num_pieces > 1) { /* randomize the special_q range */
		double piece_ratio = pow((double)special_q_max / special_q_min,
					 (double)1 / num_pieces);
		uint32 piece = get_rand(&obj->seed1,
					&obj->seed2) % num_pieces;

		special_q_min *= pow(piece_ratio, (double)piece);
		special_q_max = special_q_min * piece_ratio;
	}

	while (1) {
		uint32 quit;
		uint32 special_q_min2, special_q_max2;

		if (special_q_min <= 1) {
			special_q_min2 = special_q_max2 = 1;
		}
		else {
			special_q_min2 = MAX(special_q_min, MIN_SPECIAL_Q);
			if (special_q_min2 > special_q_max)
				break;
			else
				special_q_max2 = MIN(special_q_max,
					special_q_min2 * params.p_scale);
		}

#ifdef HAVE_CUDA
		quit = sieve_lattice_gpu(obj, &L, &params, &sieve_special_q,
				special_q_min2, special_q_max2);
#else
		quit = sieve_lattice_cpu(obj, &L, &params, &sieve_special_q,
				special_q_min2, special_q_max2);
#endif

		if (quit)
			break;

		special_q_min = special_q_max2 + 1;
	}

	sieve_fb_free(&sieve_special_q);
}
