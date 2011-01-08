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

#define P_SCALE 1.5

typedef struct {

	uint32 degree;

	double norm_max; 
	double coeff_max;
	double p_size_max;
	double sieve_size;

	uint32 special_q_min;
	uint32 special_q_max;
	uint32 special_q_fb_max;

	mpz_t gmp_high_coeff_begin;
	mpz_t gmp_high_coeff_end;
	mpz_t high_coeff; 
	mpz_t trans_N;
	mpz_t trans_m0;
	mpz_t N; 
	mpz_t m0; 
	mpz_t p;
	mpz_t tmp1;
	mpz_t tmp2;
	mpz_t tmp3;
	mpz_t tmp4;
	mpz_t tmp5;
	mpz_t mp_sieve_size;

#ifdef HAVE_CUDA
	CUcontext gpu_context;
	gpu_info_t *gpu_info; 
	CUmodule gpu_module_sq; 
	CUmodule gpu_module_nosq; 
#endif

	stage1_callback_t callback;
	void *callback_data;
} poly_search_t;

/*-----------------------------------------------------------------------*/

/* Rational leading coeffs of NFS polynomials are assumed 
   to be the product of three groups of factors p; each group 
   can be up to 32 bits in size and the product of (powers 
   of) up to MAX_P_FACTORS distinct primes */

#define MAX_P_FACTORS 7
#define MAX_ROOTS 128
#define MAX_POWER 4

#define P_SEARCH_DONE ((uint32)(-2))

/* structure for building arithmetic progressions */

typedef struct {
	uint32 p;
	uint32 num_roots;
	uint32 max_power;
	uint32 power[MAX_POWER];
	uint32 roots[MAX_POWER][MAX_POLYSELECT_DEGREE];
	uint32 cofactor_max;
	uint32 cofactor_roots_max;
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
	uint32 powers[MAX_P_FACTORS + 1];
	uint32 cofactors[MAX_P_FACTORS + 1];
	uint32 cofactor_roots[MAX_P_FACTORS + 1];
} p_enum_t;

#define ALGO_ENUM  0x1
#define ALGO_PRIME 0x2

typedef struct {
	uint32 num_roots_min;
	uint32 num_roots_max;
	uint32 avail_algos;
	uint32 fb_only;
	uint32 degree;
	uint32 p_min, p_max;
	uint64 roots[MAX_ROOTS];

	aprog_list_t aprog_data;

	prime_sieve_t p_prime;

	p_enum_t p_enum;

	mpz_t p, p2, m0, nmodp2, tmp1, tmp2, gmp_root;
} sieve_fb_t;

void sieve_fb_init(sieve_fb_t *s, poly_search_t *poly,
			uint32 factor_min, uint32 factor_max,
			uint32 fb_roots_min, uint32 fb_roots_max,
			uint32 fb_only);

void sieve_fb_free(sieve_fb_t *s);

void sieve_fb_reset(sieve_fb_t *s, uint32 p_min, uint32 p_max,
			uint32 num_roots_min, uint32 num_roots_max);

typedef void (*root_callback)(uint32 p, uint32 num_roots, uint64 *roots, 
				void *extra);

uint32 sieve_fb_next(sieve_fb_t *s, 
			poly_search_t *poly, 
			root_callback callback,
			void *extra);

/*-----------------------------------------------------------------------*/

typedef struct {
	void *p_array, *q_array, *sq_array;

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

void
handle_collision(poly_search_t *poly, uint32 p1, uint32 p2,
		uint32 special_q, uint64 special_q_root, uint128 res);

/* main search routines */

void sieve_lattice(msieve_obj *obj, poly_search_t *poly, 
				uint32 deadline);

/* low-level routines */

#ifdef HAVE_CUDA
uint32 sieve_lattice_gpu_sq(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);

uint32 sieve_lattice_gpu_nosq(msieve_obj *obj, lattice_fb_t *L);

#else

uint32 sieve_lattice_cpu(msieve_obj *obj, lattice_fb_t *L, 
		sieve_fb_t *sieve_special_q,
		uint32 special_q_min, uint32 special_q_max);
#endif

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_H_ */
