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

#ifndef _STAGE1_H_
#define _STAGE1_H_

#include <poly_skew.h>
#include <cuda_xface.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLY_BATCH_SIZE 40

#define MAX_POLYSELECT_DEGREE 6

#if MAX_POLY_DEGREE < MAX_POLYSELECT_DEGREE
#error "supported poly degree must be at least 6"
#endif

#define HIGH_COEFF_MULTIPLIER 12
#define HIGH_COEFF_PRIME_LIMIT 100
#define HIGH_COEFF_POWER_LIMIT 2

/* 128-bit integers */

typedef struct {
	uint32 w[4];
} uint128;

/*-----------------------------------------------------------------------*/

/* search bounds */

typedef struct {
	mpz_t gmp_high_coeff_begin;
	mpz_t gmp_high_coeff_end;
	double norm_max; 
	double coeff_max;
	double p_size_max;
} bounds_t;

void stage1_bounds_init(bounds_t *bounds, poly_stage1_t *data);
void stage1_bounds_free(bounds_t *bounds);
void stage1_bounds_update(bounds_t *bounds, double N, 
			double high_coeff, uint32 degree);

/*-----------------------------------------------------------------------*/

typedef struct {
	mpz_t high_coeff; 
	mpz_t trans_N;
	mpz_t trans_m0;

	double coeff_max;
	double p_size_max;

	double sieve_size;
	mpz_t mp_sieve_size;
} curr_poly_t;

typedef struct {

	uint32 degree;
	uint32 num_poly;
	curr_poly_t batch[POLY_BATCH_SIZE];

	mpz_t N; 
	mpz_t m0; 
	mpz_t p;
	mpz_t tmp1;
	mpz_t tmp2;
	mpz_t tmp3;
	mpz_t tmp4;
	mpz_t tmp5;

#ifdef HAVE_CUDA
	CUcontext gpu_context;
	gpu_info_t *gpu_info; 
	CUmodule gpu_module48; 
	CUmodule gpu_module64; 
#endif

	stage1_callback_t callback;
	void *callback_data;
} poly_search_t;

void poly_search_init(poly_search_t *poly, poly_stage1_t *data);
void poly_search_free(poly_search_t *poly);

/*-----------------------------------------------------------------------*/

/* Rational leading coeffs of NFS polynomials are assumed 
   to be the product of three groups of factors p; each group 
   can be up to 32 bits in size and the product of (powers 
   of) up to MAX_P_FACTORS distinct primes */

#define MAX_P_FACTORS 7
#define MAX_ROOTS 128

#define P_SEARCH_DONE ((uint32)(-2))

/* structure for building arithmetic progressions */

typedef struct {
	uint32 p;
	uint8 num_roots[POLY_BATCH_SIZE];
	uint32 roots[POLY_BATCH_SIZE][MAX_POLYSELECT_DEGREE];
	uint32 cofactor_max;
} aprog_t;

typedef struct {
	aprog_t *aprogs;
	uint32 num_aprogs;
	uint32 num_aprogs_alloc;
} aprog_list_t;

/* structures for finding arithmetic progressions via
   explicit enumeration */

typedef struct {
	uint32 num_factors;
	uint32 factors[MAX_P_FACTORS + 1];
	uint32 products[MAX_P_FACTORS + 1];
} p_enum_t;

#define ALGO_ENUM  0x2
#define ALGO_PRIME 0x4

typedef struct {
	uint32 num_roots_min;
	uint32 num_roots_max;
	uint32 avail_algos;
	uint32 fb_only;
	uint32 degree;
	uint32 p_min, p_max;

	aprog_list_t aprog_data;

	prime_sieve_t p_prime;

	p_enum_t p_enum;

	mpz_t p, p2, m0, nmodp2, tmp1, tmp2;
	mpz_t accum[MAX_P_FACTORS + 1];
	mpz_t roots[MAX_ROOTS];
} sieve_fb_t;

void sieve_fb_init(sieve_fb_t *s, poly_search_t *poly,
			uint32 factor_min, uint32 factor_max,
			uint32 fb_roots_min, uint32 fb_roots_max,
			uint32 fb_only);

void sieve_fb_free(sieve_fb_t *s);

void sieve_fb_reset(sieve_fb_t *s, uint32 p_min, uint32 p_max,
			uint32 num_roots_min, uint32 num_roots_max);

typedef void (*root_callback)(uint32 p, uint32 num_roots, 
				uint32 which_poly, mpz_t *roots, 
				void *extra);

uint32 sieve_fb_next(sieve_fb_t *s, 
			poly_search_t *poly, 
			root_callback callback,
			void *extra);

/*-----------------------------------------------------------------------*/

typedef struct {
	void *orig_p_array, *trans_p_array;
	void *orig_q_array, *trans_q_array;
	void *special_q_array;

#ifdef HAVE_CUDA
	CUdeviceptr gpu_p_array;
	CUdeviceptr gpu_q_array;
	CUdeviceptr gpu_found_array;
	void *found_array;
	uint32 found_array_size;
	void *p_marshall;
	void *q_marshall;
#endif

	poly_search_t *poly;

	time_t start_time;
	uint32 deadline;
} lattice_fb_t;

/* lower-level sieve routines */

uint32
sieve_lattice_deg46_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q, 
		uint32 special_q_min, uint32 special_q_max,
		sieve_fb_t *sieve_small_p,
		uint32 small_p_min, uint32 small_p_max,
		sieve_fb_t *sieve_large_p,
		uint32 large_p_min, uint32 large_p_max);

uint32
sieve_lattice_deg5_64(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q, 
		uint32 special_q_min, uint32 special_q_max,
		sieve_fb_t *sieve_small_p,
		uint32 small_p_min, uint32 small_p_max,
		sieve_fb_t *sieve_large_p,
		uint32 large_p_min, uint32 large_p_max);

void
handle_collision(poly_search_t *poly, uint32 which_poly,
			uint32 p1, uint32 p2, uint32 special_q,
			uint64 special_q_root, uint128 res);

/* main search routines */

typedef struct {
	uint32 bits; /* used to interpolate into table */
	double p_scale;

#ifdef HAVE_CUDA
	uint32 num_pieces; /* for randomization */
	uint32 max_diverge;
	uint32 special_q_min;
	uint32 special_q_max;
#else
	double num_blocks;
#endif
} sieve_fb_param_t;

void sieve_lattice(msieve_obj *obj, poly_search_t *poly, 
				uint32 deadline);

uint32 sieve_lattice_gpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);

uint32 sieve_lattice_cpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_param_t *params,
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_H_ */
