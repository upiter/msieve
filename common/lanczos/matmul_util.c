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

#include "lanczos.h"

void accum_xor(uint64 *dest, uint64 *src, uint32 n) {

	uint32 i;

	for (i = 0; i < (n & ~7); i += 8) {
		dest[i + 0] ^= src[i + 0];
		dest[i + 1] ^= src[i + 1];
		dest[i + 2] ^= src[i + 2];
		dest[i + 3] ^= src[i + 3];
		dest[i + 4] ^= src[i + 4];
		dest[i + 5] ^= src[i + 5];
		dest[i + 6] ^= src[i + 6];
		dest[i + 7] ^= src[i + 7];
	}
	for (; i < n; i++)
		dest[i] ^= src[i];
}


#ifdef HAVE_MPI

	/* The following is a functional replacement for
	   MPI_Allreduce(), but for large problems can be
	   configured to switch to a bucket accumulation
	   method that is asymptotically faster when the
	   vector is large and needs to be globally accumulated
	   and redistributed across a large number of nodes.

	   The algorithm uses the bucket strategy from the
	   paper "Global Combine on Mesh Architectures with 
	   Wormhole Routing". The implementation below is
	   based on code kindly contributed by Ilya Popovyan */

#if 0
#define GLOBAL_BREAKOVER 5000
#else
#define GLOBAL_BREAKOVER (uint32)(-1) /* turn off the fancy method */
#endif

void global_xor(uint64 *send_buf, uint64 *recv_buf, 
		uint32 total_size, MPI_Comm comm) {
	
	uint32 i;
	uint32 m, size, chunk, remainder;
	uint32 num_nodes, my_id, next_node, prev_node;
	MPI_Status status;
	MPI_Request req;
	uint64 *curr_buf;
		
	/* only get fancy for large buffers; even the
	   fancy method is only faster when many nodes 
	   are involved */

	if (total_size < GLOBAL_BREAKOVER) {
		MPI_TRY(MPI_Allreduce(send_buf, recv_buf, total_size,
				MPI_LONG_LONG, MPI_BXOR, comm))
		return;
	}

	/* split data */

	MPI_TRY(MPI_Comm_size(comm, (int *)&num_nodes))
	MPI_TRY(MPI_Comm_rank(comm, (int *)&my_id))
	chunk = total_size / num_nodes;
	remainder = total_size % num_nodes;	
	
	/* we expect a circular topology here */

	next_id = mp_modadd_1(my_id, 1, num_nodes);
	prev_id = mp_modsub_1(my_id, 1, num_nodes);
			
	/* stage 1
	   P_m sends P_{m+1} the m-th chunk of data while receiving 
	   another chunk from P_{m-1}, and does the summation op 
	   on the received chunk and (m-1)-th own chunk */

	m = my_id;
	size = chunk;
	if (my_id == num_nodes - 1)
		size += remainder;

	curr_buf = send_buf;

	for (i = 0; i < num_nodes - 1; i++) {
				
		/* asynchronously send the current chunk */

		MPI_TRY(MPI_Isend(curr_buf + m * chunk, size, 
				MPI_LONG_LONG, next_id, 97, 
				comm, &req))

		/* switch to the recvbuf after the first send */

		curr_buf = recv_buf;
				
		size = chunk;
		if ((int32)(--m) < 0) {
			m += num_nodes;
			size += remainder;
		}

		/* don't wait for send to finish, start the recv 
		   from the previous node */

		MPI_TRY(MPI_Recv(curr_buf + m * chunk, size,
				MPI_LONG_LONG, prev_id, 97, 
				comm, &status))

		/* combine the new chunk with our own */

		accum_xor(curr_buf + m * chunk,
			  send_buf + m * chunk, size);
		
		/* now wait for the send to end */

		MPI_TRY(MPI_Wait(&req, &status))
	}	
		
	/* stage 2
	   P_m sends P_{m+1} m-th chunk of data, now containing 
	   a full summation of all m-th chunks in the comm,
	   while recieving another chunk from P_{m-1} and 
	   puts it to (m-1)-th own chunk */

	curr_buf = recv_buf + m * chunk;
	for (i = 0; i < num_nodes - 1; i++){
		
		/* async send to chunk the next proc in circle */

		MPI_TRY(MPI_Isend(curr_buf, size, MPI_LONG_LONG, 
				next_id, 98, comm, &req))
		
		size = chunk;
		curr_buf -= chunk;
		if(curr_buf < recv_buf) {
			curr_buf += chunk * num_nodes;			
			size += remainder;
		}		
		
		/* don't wait for send to finish, start the recv 
		   from the previous proc in circle, put the new 
		   data just where it should be in recv_buf */

		MPI_TRY(MPI_Recv(curr_buf, size, MPI_LONG_LONG,
				prev_id, 98, comm, &status))
				
		/* now wait for the send to end */

		MPI_TRY(MPI_Wait(&req, &status))
	}
}

#endif /* HAVE_MPI */
