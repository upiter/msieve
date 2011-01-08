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

#ifndef _STAGE1_CORE_SQ_H_
#define _STAGE1_CORE_SQ_H_

#ifdef __CUDACC__
#include "cuda_intrinsics.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPECIALQ_BATCH_SIZE 40

/* structure indicating a collision */

typedef struct {
	uint32 p;
	uint32 q;
	uint32 k;
	uint32 pad;
	uint64 offset;
	uint64 proot;
} found_t;

/* the outer loop needs parallel access to different p,
   so we store in SOA format. */

#define P_SOA_BATCH_SIZE 2048

typedef struct {
	uint32 p[P_SOA_BATCH_SIZE];
	uint64 start_root[P_SOA_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE][P_SOA_BATCH_SIZE];
	float lsize[SPECIALQ_BATCH_SIZE][P_SOA_BATCH_SIZE];
} p_soa_t;

#define Q_SOA_BATCH_SIZE (3*30*384)

typedef struct {
	uint32 p[Q_SOA_BATCH_SIZE];
	uint64 start_root[Q_SOA_BATCH_SIZE];
	uint64 roots[SPECIALQ_BATCH_SIZE+1][Q_SOA_BATCH_SIZE];
	float lsize[SPECIALQ_BATCH_SIZE][Q_SOA_BATCH_SIZE];
} q_soa_t;

#ifdef __cplusplus
}
#endif

#endif /* !_STAGE1_CORE_SQ_H_ */
