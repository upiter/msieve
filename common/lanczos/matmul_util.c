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

#if 1
#define GLOBAL_BREAKOVER 5000
#else
#define GLOBAL_BREAKOVER (uint32)(-1) /* turn off the fancy method */
#endif


/*------------------------------------------------------------------*/
static void global_xor_async(uint64 *send_buf, uint64 *recv_buf, 
			uint32 total_size, uint32 num_nodes, 
			uint32 my_id, MPI_Comm comm) {
	
	uint32 i;
	uint32 m, size, chunk, remainder;
	uint32 next_id, prev_id;
	MPI_Status mpi_status;
	MPI_Request mpi_req;
	uint64 *curr_buf;
		
	/* split data */

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
				comm, &mpi_req))

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
				comm, &mpi_status))

		/* combine the new chunk with our own */

		accum_xor(curr_buf + m * chunk,
			  send_buf + m * chunk, size);
		
		/* now wait for the send to end */

		MPI_TRY(MPI_Wait(&mpi_req, &mpi_status))
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
				next_id, 98, comm, &mpi_req))
		
		size = chunk;
		curr_buf -= chunk;
		if (curr_buf < recv_buf) {
			curr_buf += chunk * num_nodes;			
			size += remainder;
		}		
		
		/* don't wait for send to finish, start the recv 
		   from the previous proc in circle, put the new 
		   data just where it should be in recv_buf */

		MPI_TRY(MPI_Recv(curr_buf, size, MPI_LONG_LONG,
				prev_id, 98, comm, &mpi_status))
				
		/* now wait for the send to end */

		MPI_TRY(MPI_Wait(&mpi_req, &mpi_status))
	}
}

static void global_xor_sendrecv(uint64 *send_buf, uint64 *recv_buf, 
		uint32 total_size, uint32 num_nodes, 
		uint32 my_id, MPI_Comm comm) {
	
	uint32 i;
	uint32 m, n, size_m, size_n; 
	uint32 chunk, remainder;
	uint32 next_id, prev_id;
	MPI_Status mpi_status;
	uint64 *curr_buf;
		
	/* split data */

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
	size_m = chunk;
	if (m == num_nodes - 1)
		size_m += remainder;

	n = prev_id;
	size_n = chunk;
	if (n == num_nodes - 1)
		size_n += remainder;

	curr_buf = send_buf;

	for (i = 0; i < num_nodes - 1; i++) {
				
		/* asynchronously send the current chunk and receive
		   the previous one */

		MPI_TRY(MPI_Sendrecv(curr_buf + m * chunk, size_m, 
				MPI_LONG_LONG, next_id, 97,
				recv_buf + n * chunk, size_n,
				MPI_LONG_LONG, prev_id, 97,
				comm, &mpi_status))

		/* switch to the recvbuf after the first send */

		curr_buf = recv_buf;

		/* combine the new chunk with our own */

		accum_xor(curr_buf + n * chunk,
			  send_buf + n * chunk, size_n);

		/* now change m and n */

		m = n;
		size_m = size_n;

		size_n = chunk;
		if ((int32)(--n) < 0) {
			n += num_nodes;
			size_n += remainder;
		}
	}	
		
	/* stage 2
	   P_m sends P_{m+1} m-th chunk of data, now containing 
	   a full summation of all m-th chunks in the comm,
	   while recieving another chunk from P_{m-1} and 
	   puts it to (m-1)-th own chunk */

	for (i = 0; i < num_nodes - 1; i++) {
				
		/* asynchronously send the current chunk and receive
		   the previous one */

		MPI_TRY(MPI_Sendrecv(recv_buf + m * chunk, size_m, 
				MPI_LONG_LONG, next_id, 98,
				recv_buf + n * chunk, size_n,
				MPI_LONG_LONG, prev_id, 98,
				comm, &mpi_status))

		/* now change m and n */

		m = n;
		size_m = size_n;

		size_n = chunk;
		if ((int32)(--n) < 0) {
			n += num_nodes;
			size_n += remainder;
		}
	}
}

static void global_xor_asyncall(uint64 *send_buf, uint64 *recv_buf, 
		uint32 total_size, uint32 num_nodes, 
		uint32 my_id, MPI_Comm comm) {
	
	int32 i;
	uint32 size;
	uint32 chunk, remainder;
	uint32 next_id, prev_id;
	uint32 num_finished;
	MPI_Request send_requests[MAX_MPI_GRID_DIM];
	MPI_Request recv_requests[MAX_MPI_GRID_DIM];
	MPI_Status mpi_status[MAX_MPI_GRID_DIM];
	int32 recv_done[MAX_MPI_GRID_DIM];
		
	/* split data */

	chunk = total_size / num_nodes;
	remainder = total_size % num_nodes;	
	
	/* we expect a circular topology here */

	next_id = mp_modadd_1(my_id, 1, num_nodes);
	prev_id = mp_modsub_1(my_id, 1, num_nodes);

	/* stage 1; start by sending our chunk */

	size = chunk;
	if (my_id == num_nodes - 1)
		size += remainder;

	MPI_TRY(MPI_Isend(send_buf + my_id * chunk, size,
			MPI_LONG_LONG, next_id, my_id,
			comm, send_requests + my_id))

	/* set up all of the receives */

	for (i = 0; i < (int32)num_nodes; i++) {
		size = chunk;
		if (i == (int32)num_nodes - 1)
			size += remainder;

		MPI_TRY(MPI_Irecv(recv_buf + i * chunk, size,
				MPI_LONG_LONG, prev_id, i, 
				comm, recv_requests + i))
	}

	/* now wait for all the receives to finish */

	num_finished = 0;
	do {
		int32 curr_finished;

		MPI_TRY(MPI_Waitsome(num_nodes, recv_requests,
					&curr_finished, recv_done,
					mpi_status))

		for (i = 0; i < curr_finished; i++) {
			uint32 curr_chunk = recv_done[i];

			/* we own chunk my_id; don't pass it on
			   once it has traversed the ring */

			if (curr_chunk == my_id)
				continue;

			/* fold in send_buf, pass it on */

			size = chunk;
			if (curr_chunk == num_nodes - 1)
				size += remainder;

			accum_xor(recv_buf + curr_chunk * chunk,
				  send_buf + curr_chunk * chunk, size);

			MPI_TRY(MPI_Isend(recv_buf + curr_chunk * chunk, 
					size, MPI_LONG_LONG, 
					next_id, curr_chunk, comm, 
					send_requests + curr_chunk))
		}

		num_finished += curr_finished;
	} while (num_finished < num_nodes);

	/* clear out all the send requests */

	MPI_TRY(MPI_Waitall(num_nodes, send_requests, mpi_status))
			
	/* stage 2; start by sending our chunk */

	size = chunk;
	if (my_id == num_nodes - 1)
		size += remainder;

	MPI_TRY(MPI_Isend(recv_buf + my_id * chunk, size,
			MPI_LONG_LONG, next_id, 
			my_id + MAX_MPI_GRID_DIM,
			comm, send_requests + my_id))

	/* set up all of the receives at once; there is
	   one receive for each chunk that is not ours */

	for (i = 0; i < (int32)num_nodes; i++) {
		if (i == (int32)my_id) {
			recv_requests[i] = MPI_REQUEST_NULL;
			continue;
		}

		size = chunk;
		if (i == (int32)num_nodes - 1)
			size += remainder;

		MPI_TRY(MPI_Irecv(recv_buf + i * chunk, size,
				MPI_LONG_LONG, prev_id, 
				i + MAX_MPI_GRID_DIM, 
				comm, recv_requests + i))
	}

	/* don't send chunk next_id to process next_id */

	send_requests[next_id] = MPI_REQUEST_NULL;

	/* now wait for all the receives to finish */

	num_finished = 0;
	do {
		int32 curr_finished;

		MPI_TRY(MPI_Waitsome(num_nodes, recv_requests,
					&curr_finished, recv_done,
					mpi_status))

		for (i = 0; i < curr_finished; i++) {
			uint32 curr_chunk = recv_done[i];

			if (curr_chunk == my_id || curr_chunk == next_id)
				continue;

			/* pass curr_chunk on */

			size = chunk;
			if (curr_chunk == num_nodes - 1)
				size += remainder;

			MPI_TRY(MPI_Isend(recv_buf + curr_chunk * chunk, 
					size, MPI_LONG_LONG, next_id, 
					curr_chunk + MAX_MPI_GRID_DIM, comm, 
					send_requests + curr_chunk))
		}

		num_finished += curr_finished;
	} while (num_finished < num_nodes);

	/* clear out all the send requests */

	MPI_TRY(MPI_Waitall(num_nodes, send_requests, mpi_status))
}

/*------------------------------------------------------------------*/
void global_xor(uint64 *send_buf, uint64 *recv_buf, 
		uint32 total_size, uint32 num_nodes, 
		uint32 my_id, MPI_Comm comm) {
	
	/* only get fancy for large buffers; even the
	   fancy method is only faster when many nodes 
	   are involved */

	if (total_size < GLOBAL_BREAKOVER || num_nodes < 2) {
		MPI_TRY(MPI_Allreduce(send_buf, recv_buf, total_size,
				MPI_LONG_LONG, MPI_BXOR, comm))
		return;
	}

#if 1
	global_xor_async(send_buf, recv_buf, 
		total_size, num_nodes, my_id, comm);
#elif 0
	global_xor_sendrecv(send_buf, recv_buf, 
		total_size, num_nodes, my_id, comm);
#elif 0
	global_xor_asyncall(send_buf, recv_buf, 
		total_size, num_nodes, my_id, comm);
#endif
}

#endif /* HAVE_MPI */
