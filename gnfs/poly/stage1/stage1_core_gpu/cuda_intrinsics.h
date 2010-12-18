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

#if defined(__CUDACC__) && !defined(CUDA_INTRINSICS_H)
#define CUDA_INTRINSICS_H

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
#define HAVE_FERMI
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef unsigned int uint32;
typedef unsigned long long uint64;

#define MIN(x, y) ((x) < (y) ? (x) : (y))

/*------------------- Low-level functions ------------------------------*/

__device__ uint32
__uaddo(uint32 a, uint32 b) {
	uint32 res;
	asm("add.cc.u32 %0, %1, %2; /* inline */ \n\t" 
	    : "=r" (res) : "r" (a) , "r" (b));
	return res;
}

__device__ uint32
__uaddc(uint32 a, uint32 b) {
	uint32 res;
	asm("addc.cc.u32 %0, %1, %2; /* inline */ \n\t" 
	    : "=r" (res) : "r" (a) , "r" (b));
	return res;
}

__device__ uint32
__umul24hi(uint32 a, uint32 b) {
	uint32 res;
	asm("mul24.hi.u32 %0, %1, %2; /* inline */ \n\t" 
	    : "=r" (res) : "r" (a) , "r" (b));
	return res;
}

/*----------------- Squaring ----------------------------------------*/

__device__ uint64 
wide_sqr32(uint32 a)
{
	uint32 a0, a1;

	asm("{ .reg .u64 %dprod; \n\t"
	    "mul.wide.u32 %dprod, %2, %2; \n\t"
	    "cvt.u32.u64 %0, %dprod;      \n\t"
	    "shr.u64 %dprod, %dprod, 32;  \n\t"
	    "cvt.u32.u64 %1, %dprod;      \n\t"
	    "}                   \n\t"
	    : "=r"(a0), "=r"(a1)
	    : "r"(a));

	return (uint64)a1 << 32 | a0;
}

/* -------------------- Modular subtraction ------------------------*/

__device__ uint64 
modsub64(uint64 a, uint64 b, uint64 p) 
{
	uint32 r0, r1;
	uint32 a0 = (uint32)a;
	uint32 a1 = (uint32)(a >> 32);
	uint32 b0 = (uint32)b;
	uint32 b1 = (uint32)(b >> 32);
	uint32 p0 = (uint32)p;
	uint32 p1 = (uint32)(p >> 32);

	asm("{  \n\t"
	    ".reg .pred %pborrow;           \n\t"
	    ".reg .u32 %borrow;           \n\t"
	    "mov.b32 %borrow, 0;           \n\t"
	    "sub.cc.u32 %0, %2, %4;        \n\t"
	    "subc.cc.u32 %1, %3, %5;        \n\t"
	    "subc.u32 %borrow, %borrow, 0; \n\t"
	    "setp.ne.u32 %pborrow, %borrow, 0;  \n\t"
	    "@%pborrow add.cc.u32 %0, %0, %6; \n\t"
	    "@%pborrow addc.u32 %1, %1, %7; \n\t"
	    "} \n\t"
	    : "=r"(r0), "=r"(r1)
	    : "r"(a0), "r"(a1), 
	      "r"(b0), "r"(b1), 
	      "r"(p0), "r"(p1));

	return (uint64)r1 << 32 | r0;
}

/*-------------------------- Modular inverse -------------------------*/

__device__ uint32 
modinv32(uint32 a, uint32 p) {

	uint32 ps1, ps2, dividend, divisor, rem, q, t;
	uint32 parity;

	q = 1; rem = a; dividend = p; divisor = a;
	ps1 = 1; ps2 = 0; parity = 0;

	while (divisor > 1) {
		rem = dividend - divisor;
		t = rem - divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t; t -= divisor;
		if (rem >= divisor) { q += ps1; rem = t;
		if (rem >= divisor) {
			q = dividend / divisor;
			rem = dividend - q * divisor;
			q *= ps1;
		} } } } } } } } }

		q += ps2;
		parity = ~parity;
		dividend = divisor;
		divisor = rem;
		ps2 = ps1;
		ps1 = q;
	}
	
	if (parity == 0)
		return ps1;
	else
		return p - ps1;
}

/*------------------- Montgomery arithmetic --------------------------*/
#ifdef HAVE_FERMI
#define montmul48(a,b,n,w) montmul64(a,b,n,w)
#else
__device__ uint64 
montmul48(uint64 a, uint64 b,
		uint64 n, uint32 w) {

	uint32 a0 = (uint32)a;
	uint32 a1 = (uint32)(a >> 24);
	uint32 b0 = (uint32)b;
	uint32 b1 = (uint32)(b >> 24);
	uint32 n0 = (uint32)n;
	uint32 n1 = (uint32)(n >> 24);
	uint32 acc0, acc1;
	uint32 q0, q1;
	uint32 prod_lo, prod_hi;
	uint64 r;

	acc0 = __umul24(a0, b0);
	acc1 = __umul24hi(a0, b0) >> 16;
	q0 = __umul24(acc0, w);
	prod_lo = __umul24(q0, n0);
	prod_hi = __umul24hi(q0, n0) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc0 = acc0 >> 24 | acc1 << 8;

	prod_lo = __umul24(a0, b1);
	prod_hi = __umul24hi(a0, b1) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(0, prod_hi);
	prod_lo = __umul24(a1, b0);
	prod_hi = __umul24hi(a1, b0) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	prod_lo = __umul24(q0, n1);
	prod_hi = __umul24hi(q0, n1) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	q1 = __umul24(acc0, w);
	prod_lo = __umul24(q1, n0);
	prod_hi = __umul24hi(q1, n0) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc0 = acc0 >> 24 | acc1 << 8;

	prod_lo = __umul24(a1, b1);
	prod_hi = __umul24hi(a1, b1) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(0, prod_hi);
	prod_lo = __umul24(q1, n1);
	prod_hi = __umul24hi(q1, n1) >> 16;
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);

	r = (uint64)acc1 << 32 | acc0;
	if (r >= n)
		return r - n;
	else
		return r;
}
#endif

__device__ uint64 
montmul64(uint64 a, uint64 b,
		uint64 n, uint32 w) {

	uint32 a0 = (uint32)a;
	uint32 a1 = (uint32)(a >> 32);
	uint32 b0 = (uint32)b;
	uint32 b1 = (uint32)(b >> 32);
	uint32 n0 = (uint32)n;
	uint32 n1 = (uint32)(n >> 32);
	uint32 acc0, acc1, acc2;
	uint32 q0, q1;
	uint32 prod_lo, prod_hi;
	uint64 r;

	acc0 = a0 * b0;
	acc1 = __umulhi(a0, b0);
	q0 = acc0 * w;
	prod_lo = q0 * n0;
	prod_hi = __umulhi(q0, n0);
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc2 = __uaddc(0, 0);

	prod_lo = a0 * b1;
	prod_hi = __umulhi(a0, b1);
	acc0 = __uaddo(acc1, prod_lo);
	acc1 = __uaddc(acc2, prod_hi);
	acc2 = __uaddc(0, 0);
	prod_lo = a1 * b0;
	prod_hi = __umulhi(a1, b0);
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc2 = __uaddc(acc2, 0);
	prod_lo = q0 * n1;
	prod_hi = __umulhi(q0, n1);
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc2 = __uaddc(acc2, 0);
	q1 = acc0 * w;
	prod_lo = q1 * n0;
	prod_hi = __umulhi(q1, n0);
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc2 = __uaddc(acc2, 0);

	prod_lo = a1 * b1;
	prod_hi = __umulhi(a1, b1);
	acc0 = __uaddo(acc1, prod_lo);
	acc1 = __uaddc(acc2, prod_hi);
	acc2 = __uaddc(0, 0);
	prod_lo = q1 * n1;
	prod_hi = __umulhi(q1, n1);
	acc0 = __uaddo(acc0, prod_lo);
	acc1 = __uaddc(acc1, prod_hi);
	acc2 = __uaddc(acc2, 0);

	r = (uint64)acc1 << 32 | acc0;
	if (acc2 || r >= n)
		return r - n;
	else
		return r;
}

/*------------------ Initializing Montgomery arithmetic -----------------*/

#ifdef HAVE_FERMI
#define montmul24_w(n) montmul32_w(n)
#else
__device__ uint32 
montmul24_w(uint32 n) {

	uint32 res = 8 - (n % 8);
	res = __umul24(res, 2 + __umul24(n, res));
	res = __umul24(res, 2 + __umul24(n, res));
	return __umul24(res, 2 + __umul24(n, res));
}
#endif

__device__ uint32 
montmul32_w(uint32 n) {

	uint32 res = 2 + n;
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	res = res * (2 + n * res);
	return res * (2 + n * res);
}

#ifdef HAVE_FERMI
#define montmul48_r(n,w) montmul64_r(n,w)
#else
__device__ uint64 
montmul48_r(uint64 n, uint32 w) {

	uint32 shift;
	uint32 i;
	uint64 shifted_n;
	uint64 res;

	shift = __clzll(n);
	shifted_n = n << shift;
	res = -shifted_n;

	for (i = 64 - shift; i < 60; i++) {
		if (res >> 63)
			res = res + res - shifted_n;
		else
			res = res + res;

		if (res >= shifted_n)
			res -= shifted_n;
	}

	res = res >> shift;
	res = montmul48(res, res, n, w);
	return montmul48(res, res, n, w);
}
#endif

__device__ uint64 
montmul64_r(uint64 n, uint32 w) {

	uint32 shift;
	uint32 i;
	uint64 shifted_n;
	uint64 res;

	shift = __clzll(n);
	shifted_n = n << shift;
	res = -shifted_n;

	for (i = 64 - shift; i < 72; i++) {
		if (res >> 63)
			res = res + res - shifted_n;
		else
			res = res + res;

		if (res >= shifted_n)
			res -= shifted_n;
	}

	res = res >> shift;
	res = montmul64(res, res, n, w);
	res = montmul64(res, res, n, w);
	return montmul64(res, res, n, w);
}

#ifdef __cplusplus
}
#endif

#endif /* defined(__CUDACC__) && !defined(CUDA_INTRINSICS_H) */

