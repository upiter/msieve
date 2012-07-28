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

#include "poly_skew.h"

typedef struct {
	FILE *all_poly_file;
	poly_config_t *config;
} stage2_callback_data_t;

/*------------------------------------------------------------------*/
static void stage1_callback(mpz_t high_coeff, mpz_t p, mpz_t m, 
				double coeff_bound, void *extra) {
	
	gmp_printf("%Zd %Zd %Zd\n", high_coeff, p, m);
	poly_stage2_run((poly_stage2_t *)extra, high_coeff, p, m, 
			coeff_bound, NULL);
}

/*------------------------------------------------------------------*/
static void stage1_callback_log(mpz_t high_coeff, mpz_t p, mpz_t m, 
				double coeff_bound, void *extra) {
	
	FILE *mfile = (FILE *)extra;
	gmp_printf("%Zd %Zd %Zd\n", high_coeff, p, m);
	gmp_fprintf(mfile, "%Zd %Zd %Zd\n",
			high_coeff, p, m);
	fflush(mfile);
}

/*------------------------------------------------------------------*/
static void
stage2_callback(void *extra, uint32 degree, 
		mpz_t * coeff1, mpz_t * coeff2,
		double skewness, double size_score,
		double root_score, double combined_score,
		uint32 num_real_roots)
{
	uint32 i;
	poly_select_t poly;
	mpz_poly_t *rpoly;
	mpz_poly_t *apoly;
	stage2_callback_data_t *data = (stage2_callback_data_t *)extra;
	poly_config_t *config = data->config;

	memset(&poly, 0, sizeof(poly_select_t));
	rpoly = &poly.rpoly;
	apoly = &poly.apoly;
	mpz_poly_init(rpoly);
	mpz_poly_init(apoly);

	rpoly->degree = 1;
	for (i = 0; i <= 1; i++)
		mpz_set(rpoly->coeff[i], coeff2[i]);

	apoly->degree = degree;
	for (i = 0; i <= degree; i++)
		mpz_set(apoly->coeff[i], coeff1[i]);

	poly.root_score = root_score;
	poly.size_score = size_score;
	poly.combined_score = combined_score;
	poly.skewness = skewness;

	printf("save %le %.4lf %.2lf %le rroots %u\n", size_score,
			root_score, skewness, combined_score,
			num_real_roots);

	fprintf(data->all_poly_file, 
		"# norm %le alpha %lf e %.3le rroots %u\nskew: %.2lf\n", 
		size_score, root_score, combined_score, 
		num_real_roots, skewness);
	for (i = 0; i <= degree; i++)
		gmp_fprintf(data->all_poly_file, "c%u: %Zd\n", i, coeff1[i]);
	for (i = 0; i <= 1; i++)
		gmp_fprintf(data->all_poly_file, "Y%u: %Zd\n", i, coeff2[i]);
	fflush(data->all_poly_file);

	save_poly(config, &poly);
	mpz_poly_free(rpoly);
	mpz_poly_free(apoly);
}

/*------------------------------------------------------------------*/
void find_poly_core(msieve_obj *obj, mpz_t n,
			poly_param_t *params,
			poly_config_t *config,
			uint32 degree) {

	poly_stage1_t stage1_data;
	poly_stage2_t stage2_data;
	stage2_callback_data_t stage2_callback_data;
	char buf[2048];
	FILE *stage1_outfile = NULL;
	uint32 do_both_stages = 0;
	double coeff_scale = 3.0;
	const char *lower_limit = NULL;
	const char *upper_limit = NULL;

	/* make sure the configured stages have the bounds that
	   they need. We only have to check maximum bounds, since
	   if not provided then no polynomials will be found */

	if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
	    params->stage1_norm == 0) {
		printf("error: stage 1 bound not provided\n");
		exit(-1);
	}

	if ((obj->flags & MSIEVE_FLAG_NFS_POLY2) &&
	    params->stage2_norm == 0) {
		printf("error: stage 2 bound not provided\n");
		exit(-1);
	}

	if ((obj->flags & MSIEVE_FLAG_NFS_POLY1) &&
	    (obj->flags & MSIEVE_FLAG_NFS_POLY2))
		do_both_stages = 1;

	/* parse arguments */

	if (obj->nfs_args != NULL) {
		const char *tmp;

		upper_limit = strchr(obj->nfs_args, ',');
		if (upper_limit != NULL) {
			lower_limit = upper_limit - 1;
			while (lower_limit > obj->nfs_args &&
				isdigit(lower_limit[-1])) {
				lower_limit--;
			}
			upper_limit++;
		}

		tmp = strstr(obj->nfs_args, "min_coeff=");
		if (tmp != NULL)
			lower_limit = tmp + 10;

		tmp = strstr(obj->nfs_args, "max_coeff=");
		if (tmp != NULL)
			upper_limit = tmp + 10;
	}

	/* set up stage 1 */

	if (obj->flags & MSIEVE_FLAG_NFS_POLY1) {

		/* if we're doing both stage 1 and 2, every hit
		   in stage 1 is immediately forwarded to stage 2.
		   For stage 1 alone, all the stage 1 hits are buffered
		   to file first */

		if (do_both_stages) {
			poly_stage1_init(&stage1_data, 
					stage1_callback, &stage2_data);
		}
		else {
			sprintf(buf, "%s.m", obj->savefile.name);
			stage1_outfile = fopen(buf, "a");
			if (stage1_outfile == NULL) {
				printf("error: cannot open poly1 file\n");
				exit(-1);
			}
			poly_stage1_init(&stage1_data, stage1_callback_log, 
					stage1_outfile);
		}

		/* fill stage 1 data */

		mpz_set(stage1_data.gmp_N, n);
		stage1_data.degree = degree;
		stage1_data.norm_max = params->stage1_norm;
		stage1_data.deadline = params->deadline;

		if (lower_limit != NULL)
			gmp_sscanf(lower_limit, "%Zd",
					stage1_data.gmp_high_coeff_begin);
		else
			mpz_set_ui(stage1_data.gmp_high_coeff_begin, 
					(unsigned long)1);

		if (upper_limit != NULL)
			gmp_sscanf(upper_limit, "%Zd",
					stage1_data.gmp_high_coeff_end);
		else
			mpz_set_d(stage1_data.gmp_high_coeff_end,
				pow(mpz_get_d(stage1_data.gmp_N), 1.0 / 
					(double)(degree * (degree - 1))) / 
				coeff_scale );

		/* the time deadline is ignored if performing stage 1 on a range */

		if (lower_limit != NULL && upper_limit != NULL)
			stage1_data.deadline = 0;
		else
			logprintf(obj, "time limit set to %.2f CPU-hours\n",
				stage1_data.deadline / 3600.0);

		{ /* SB: tried L[1/3,c] fit; it is no better than this */
			double e0 = (params->digits >= 121) ? 
						(0.0607 * params->digits + 2.25):
				                (0.0526 * params->digits + 3.23);
			if (degree == 4)
				e0 = 0.0625 * params->digits + 1.69;
			e0 = exp(-log(10) * e0); 
#ifdef HAVE_CUDA
			e0 *= 1.15;
#endif
			logprintf(obj, "expecting poly E from %.2le to > %.2le\n",
				e0, 1.15 * e0);
			/* seen exceptional polys with +40% but that's */
			/* very rare. The fit is good for 88..232 digits */
		}
 
		logprintf(obj, "searching leading coefficients from "
				"%.0lf to %.0lf\n",
				mpz_get_d(stage1_data.gmp_high_coeff_begin),
				mpz_get_d(stage1_data.gmp_high_coeff_end));
	}

	/* set up stage 2 */

	if (obj->flags & MSIEVE_FLAG_NFS_POLY2) {

		poly_stage2_init(&stage2_data, obj, stage2_callback, 
				&stage2_callback_data);

		/* fill stage 2 data */

		mpz_set(stage2_data.gmp_N, n);
		stage2_data.degree = degree;
		stage2_data.max_norm = params->stage2_norm;
		stage2_data.min_e = params->final_norm;

		sprintf(buf, "%s.p", obj->savefile.name);
		stage2_callback_data.config = config;
		stage2_callback_data.all_poly_file = fopen(buf, "a");
		if (stage2_callback_data.all_poly_file == NULL) {
			printf("error: cannot open poly2 file\n");
			exit(-1);
		}
	}


	if (do_both_stages) {
		poly_stage1_run(obj, &stage1_data);

		fclose(stage2_callback_data.all_poly_file);
		poly_stage1_free(&stage1_data);
		poly_stage2_free(&stage2_data);
	}
	else if (obj->flags & MSIEVE_FLAG_NFS_POLY1) {
		poly_stage1_run(obj, &stage1_data);
		fclose(stage1_outfile);
		poly_stage1_free(&stage1_data);
	}
	else if (obj->flags & MSIEVE_FLAG_NFS_POLY2) {
		uint32 i;
		mpz_t ad, p, m;
		mpz_t full_apoly[MAX_POLY_DEGREE + 1];

		mpz_init(ad);
		mpz_init(p);
		mpz_init(m);
		for (i = 0; i <= MAX_POLY_DEGREE; i++)
			mpz_init(full_apoly[i]);

		sprintf(buf, "%s.m", obj->savefile.name);
		stage1_outfile = fopen(buf, "r");
		if (stage1_outfile == NULL) {
			printf("error: cannot open poly2 input file\n");
			exit(-1);
		}

		while (1) {
			int c;
			char *tmp = buf;
			mpz_t *arg = NULL;

			if (fgets(buf, sizeof(buf), stage1_outfile) == NULL)
				break;

			if (gmp_sscanf(tmp, "%Zd%n", ad, &c) != 1)
				continue;
			tmp += c;

			if (mpz_cmp_ui(ad, 0) == 0) {
				for (i = 0; i <= degree; i++) {
					if (gmp_sscanf(tmp, "%Zd%n", 
					    full_apoly[degree - i], &c) != 1)
						break;
					tmp += c;
				}

				mpz_set(ad, full_apoly[degree]);
				arg = full_apoly;
			}
			if (gmp_sscanf(tmp, "%Zd %Zd", p, m) != 2)
				continue;

			if (arg == NULL)
				gmp_printf("poly %Zd %Zd %Zd\n", ad, p, m);

			poly_stage2_run(&stage2_data, ad, p, m, 1e100, arg);

			if (obj->flags & MSIEVE_FLAG_STOP_SIEVING)
				break;
		}

		mpz_clear(ad);
		mpz_clear(p);
		mpz_clear(m);
		for (i = 0; i <= MAX_POLY_DEGREE; i++)
			mpz_clear(full_apoly[i]);
		fclose(stage1_outfile);
		fclose(stage2_callback_data.all_poly_file);
		poly_stage2_free(&stage2_data);
	}
}
