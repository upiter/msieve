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

/* Interface for selecting GNFS polynomials whose top
   three coefficients are small. We use Kleinjung's improved
   algorithm presented at the 2008 CADO Factoring Workshop,
   with many modifications */

#include <poly_skew.h>
#include <cuda_xface.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_POLYSELECT_DEGREE 6

#if MAX_POLY_DEGREE < MAX_POLYSELECT_DEGREE
#error "supported poly degree must be at least 6"
#endif

/* we try to limit the high-order algebraic poly coefficient 
   to the set with many small primes, to increase the odds 
   that the resulting polynomial will have unusually many 
   projective roots modulo small primes. All high coeffs
   are chosen to be divisible by HIGH_COEFF_MULTIPLIER */

#define HIGH_COEFF_MULTIPLIER 12
#define HIGH_COEFF_PRIME_LIMIT 100
#define HIGH_COEFF_POWER_LIMIT 2

/* for practical reasons, we limit the size of product of
   small primes that we look for in the high-order algebraic
   coefficient. High coeffs larger than
   HIGH_COEFF_MULTIPLIER * HIGH_COEFF_SIEVE_LIMIT
   will have additional factors that may be large */

#define HIGH_COEFF_SIEVE_LIMIT 1e12

/*-----------------------------------------------------------------------*/

/* main structure used in stage 1 */

typedef struct {

	uint32 degree;

	/* approximately (N/a_d) ^ (1/d) for input N, high
	   coefficient a_d and poly degree d */

	double m0;

	/* bound on the norm used in stage 1; this is the maximum
	   value of (poly coefficient i) * (optimal skew)^i across
	   all poly coefficients. The low-order poly coefficients
	   are bounded in size by m0, because the polynomial is
	   essentially N split into d+1 pieces. Our job is to find
	   a 'stage 1 hit' that obeys the norm bound even for the
	   high-order algebraic coefficients */

	double norm_max; 

	/* (computed) bound on the third-highest algebraic
	   poly coefficient. Making this small is the only
	   thing stage 1 can do; the other coefficients can
	   only be optimized in stage 2 */

	double coeff_max;

	/* bound on the leading rational poly coefficient */

	double p_size_max;

	/* the range on a_d, provided by calling code */

	mpz_t gmp_high_coeff_begin;
	mpz_t gmp_high_coeff_end;

	/* internal values used */

	mpz_t high_coeff; 
	mpz_t trans_N;
	mpz_t trans_m0;
	mpz_t N; 
	mpz_t m; 
	mpz_t p;
	mpz_t tmp1;
	mpz_t tmp2;
	mpz_t tmp3;
	mpz_t tmp4;
	mpz_t tmp5;

#ifdef HAVE_CUDA

	/* main structures for GPU-based sieving */

	CUcontext gpu_context;
	gpu_info_t *gpu_info;
	CUmodule gpu_module;
#endif

	/* function to call when a collision is found */

	stage1_callback_t callback;
	void *callback_data;
} poly_search_t;

/*-----------------------------------------------------------------------*/

/* Kleinjung's algorithm essentially reduces to an
   all-against-all search between two large collections of
   arithmetic progressions, looking for pairs of progressions
   that satisfy a specific modular property. The following
   controls how much larger the largest element in a collection
   is, compared to the smallest */

#define P_SCALE 2.4

/* Rational leading coeffs of NFS polynomials are assumed 
   to be the product of 2 or 3 coprime groups of factors p; 
   each p is < 2^32 and the product of (powers of) up to 
   MAX_P_FACTORS distinct primes. The arithmetic progressions 
   mentioned above are of the form r_i + k * p^2 for 'root' r_i. 
   p may be prime or composite, and may have up to MAX_ROOTS 
   roots. 
   
   We get a collision when we can find two progressions 
   aprog1(k) = r_i1 + k*p1^2 and aprog2(k) = r_i2+k*p2^2, 
   and integers k1 and k2, such that

   	- p1 and p2 are coprime
	- aprog1(k1) = aprog2(k2)
	- the common value is "close to" m0
   
   If p has several prime factors p_i, the exact number of 
   roots that a given p has is the product of the number of 
   d_th roots of N modulo each p_i. For degree 4 polyomials, N 
   has either 0, 2, or 4 fourth roots mod p_i. For degree 5, 
   N has either 1 or 5 fifth roots mod p_i. For degree 6, 
   N has either 0, 2 or 6 sixth roots mod p_i. So especially
   when p is large and p_i are small, a composite p can 
   contribute a large number of progressions to the collision 
   search

   We will need to generate many p along with all their r_i
   fairly often, and need efficient methods to do so */

#define MAX_P_FACTORS 7
#define MAX_ROOTS 128

#define P_SEARCH_DONE ((uint32)(-2))

/* structure for building arithmetic progressions */

typedef struct {
	uint32 power;
	uint32 roots[MAX_POLYSELECT_DEGREE];
} aprog_power_t;

typedef struct {
	uint32 p;

	/* number of 'degree-th roots' of N (mod p) */
	uint32 num_roots;

	/* largest e for which p^e < 2^32 */
	uint32 max_power;

	/* power p^e_j and its 'degree-th roots' of N (mod p^e_j) */
	aprog_power_t *powers;

	/* maximum product of other p which may be
	   combined with this p */
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
	/* number of distinct primes p_i in the current enum product */
	uint32 num_factors;

	/* index into aprog list for each distinct prime p_i */
	uint32 curr_factor[MAX_P_FACTORS + 1];

	/* exponent e_i of each prime p_i */
	uint32 curr_power[MAX_P_FACTORS + 1];

	/* the 'running product' of p_1^e_1..p_{i-1}^e_{i-1} for
	   each i, or 1 if i==1 */
	uint32 curr_prod[MAX_P_FACTORS + 1];

	/* number of roots in each 'running product' */
	uint32 curr_num_roots[MAX_P_FACTORS + 1];
} p_enum_t;

#define ALGO_ENUM  0x1
#define ALGO_PRIME 0x2

/* a factory for building arithmetic progressions */

typedef struct {
	uint32 num_roots_min;
	uint32 num_roots_max;
	uint32 avail_algos;
	uint32 fb_only;
	uint32 degree;
	uint32 p_min, p_max;
	uint64 roots[MAX_ROOTS];
	mpz_poly_t tmp_poly;

	aprog_list_t aprog_data;

	prime_sieve_t p_prime;

	p_enum_t p_enum;

	mpz_t p, pp, nmodpp, tmp1, tmp2, tmp3, gmp_root;
} sieve_fb_t;

/* externally visible interface */

/* initialize the factory. p is allowed to have small prime
   factors p_i between factor_min and factor_max, and N
   will have between fb_roots_min and fb_roots_max d_th
   roots modulo each of these p_i. Additionally, if fb_only
   is zero, a sieve is used to find any prime p which are
   larger than factor_max */

void sieve_fb_init(sieve_fb_t *s, poly_search_t *poly,
			uint32 factor_min, uint32 factor_max,
			uint32 fb_roots_min, uint32 fb_roots_max,
			uint32 fb_only);

void sieve_fb_free(sieve_fb_t *s);

/* set up for a run of p production. The generated p will 
   all be between p_min and p_max, and the number of roots
   for each p is between num_roots_min and num_roots_max */

void sieve_fb_reset(sieve_fb_t *s, uint32 p_min, uint32 p_max,
			uint32 num_roots_min, uint32 num_roots_max);

/* function that 'does something' when a single p 
   and all its roots is found */

typedef void (*root_callback)(uint32 p, uint32 num_roots, uint64 *roots, 
				void *extra);

/* find the next p and all of its roots. The code returns
   P_SEARCH_DONE if no more p exist, otherwise it calls 
   callback() and returns p.

   p which are products of small primes are found first, then
   large prime p (since prime p are slower and you may not want
   all of them). Prime p will be found in ascending order but 
   no order may be assumed for composite p returned by 
   consecutive calls */

uint32 sieve_fb_next(sieve_fb_t *s, 
			poly_search_t *poly, 
			root_callback callback,
			void *extra);

/*-----------------------------------------------------------------------*/

/* what to do when the collision search finds a 'stage 1 hit' */

void
handle_collision(poly_search_t *poly, uint64 p, uint32 special_q,
		uint64 special_q_root, int64 res);

/* main search routine */

#ifdef HAVE_CUDA

/* GPU search routine */

double sieve_lattice_gpu(msieve_obj *obj,
			poly_search_t *poly, double deadline);

#else

/* CPU search routine */

double sieve_lattice_cpu(msieve_obj *obj,
			poly_search_t *poly, double deadline);

#endif

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_H_ */
