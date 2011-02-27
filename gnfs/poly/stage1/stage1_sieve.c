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

/* wrapper for the collision search */

/*------------------------------------------------------------------------*/
void
handle_collision(poly_search_t *poly, uint32 p1, uint32 p2,
		uint32 special_q, uint64 special_q_root, uint128 res)
{
	/* the proposed rational coefficient is p1*p2*special_q;
	   all these pieces must be coprime */

	if (mp_gcd_1(special_q, p1) != 1 ||
	    mp_gcd_1(special_q, p2) != 1 ||
	    mp_gcd_1(p1, p2) != 1)
		return;

	mpz_set_ui(poly->p, (unsigned long)p1);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)p2);
	mpz_mul_ui(poly->p, poly->p, (unsigned long)special_q);

	mpz_gcd(poly->tmp3, poly->p, poly->high_coeff);
	if (mpz_cmp_ui(poly->tmp3, 1))
		return;

	/* the corresponding correction to m0 is 
	   special_q_root + res * special_q^2 - sieve_size,
	   and can be positive or negative */

	uint64_2gmp(special_q_root, poly->tmp1);
	mpz_import(poly->tmp2, 4, -1, sizeof(uint32), 0, 0, &res);
	mpz_set_ui(poly->tmp3, (unsigned long)special_q);

	mpz_mul(poly->tmp3, poly->tmp3, poly->tmp3);
	mpz_addmul(poly->tmp1, poly->tmp2, poly->tmp3);
	mpz_sub(poly->tmp1, poly->tmp1, poly->mp_sieve_size);
	mpz_add(poly->m0, poly->trans_m0, poly->tmp1);

	/* a lot can go wrong before this function is called!
	   Check that Kleinjung's modular condition is satisfied */

	mpz_pow_ui(poly->tmp1, poly->m0, (mp_limb_t)poly->degree);
	mpz_mul(poly->tmp2, poly->p, poly->p);
	mpz_sub(poly->tmp1, poly->trans_N, poly->tmp1);
	mpz_tdiv_r(poly->tmp3, poly->tmp1, poly->tmp2);
	if (mpz_cmp_ui(poly->tmp3, (mp_limb_t)0)) {
		gmp_printf("poly %Zd %Zd %Zd\n",
				poly->high_coeff, poly->p, poly->m0);
		printf("crap\n");
		return;
	}

	/* the pair works, now translate the computed m0 back
	   to the original polynomial. We do this via a cheesy
	   version of the extended Euclidean algorithm.
	
	   Try to make the computed correction as small as 
	   possible. This is important because we have several
	   choices that will all work but some will not satisfy
	   the bound on a_{d-2} when we run stage 2 */

	mpz_mul_ui(poly->tmp1, poly->high_coeff, (mp_limb_t)poly->degree);
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

	gmp_printf("poly %Zd %Zd %Zd\n",
			poly->high_coeff, poly->p, poly->m0);

	poly->callback(poly->high_coeff, poly->p, poly->m0,
			poly->coeff_max, poly->callback_data);
}

/*------------------------------------------------------------------------*/
void
sieve_lattice(msieve_obj *obj, poly_search_t *poly, uint32 deadline)
{
	uint32 quit = 0;
	uint32 special_q_min = poly->special_q_min;
	uint32 special_q_max = poly->special_q_max;
	uint32 special_q_fb_max = poly->special_q_fb_max;
	lattice_fb_t L;
	sieve_fb_t sieve_special_q;

	printf("p = %.2lf bits, sieve = %.2lf bits\n",
			log(poly->p_size_max) / M_LN2,
			log(poly->sieve_size) / M_LN2);

	/* set up the special q factory; it may not be
	   needed, but special-q may have arbitrary factors */

	sieve_fb_init(&sieve_special_q, poly,
			5, special_q_fb_max,
			1, poly->degree,
			0);

	L.poly = poly;
	L.start_time = time(NULL);
	L.deadline = deadline;

	if (special_q_min == 1) { /* handle trivial case */

#ifdef HAVE_CUDA
		quit = sieve_lattice_gpu_nosq(obj, &L);
#else
		quit = sieve_lattice_cpu(obj, &L, &sieve_special_q, 1, 1);
#endif

	}

	if (quit || special_q_max == 1)
		goto finished;

	/* large problems with a long time limit may exhaust
	   one batch of special q; in that case, if the search is
	   broken up into pieces then just go on to the next 
	   batch of special-q */

	while (1) {

		uint32 special_q_min2 = special_q_min;
		uint32 special_q_max2 = MIN(special_q_max,
						special_q_min * P_SCALE);

#ifdef HAVE_CUDA
		quit = sieve_lattice_gpu_sq(obj, &L, &sieve_special_q,
						special_q_min2,
						special_q_max2);
#else
		quit = sieve_lattice_cpu(obj, &L, &sieve_special_q,
						special_q_min2,
						special_q_max2);
#endif

		if (quit || special_q_max2 > special_q_max / P_SCALE)
			break;

		special_q_min = special_q_max2 + 1;
	}

finished:
	sieve_fb_free(&sieve_special_q);
}
