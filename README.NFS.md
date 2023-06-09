# MSIEVE: A Library for Factoring Large Integers
# Number Field Sieve Module Documentation

Jason Papadopoulos


## Introduction

This file describes the implementation of the number field sieve that
Msieve uses. By the time of this writing (version 1.51), the NFS code in
Msieve has been the focus of almost continuous development on my part for
about 6 years, but there are parts of it that still need doing. In
order to truly be a fire-and-forget tool for NFS factoring, the current
code needs a lattice sieve which is fast enough to be useful, and that is
a very large undertaking given the time constraints I have. Because the
code has no lattice sieve available, use of the NFS module must always
be explicitly specified in the Msieve demo binary.

Nonetheless, the implementation of all the other phases of the Number Field
Sieve is pretty much production-quality by now, so that if you have a
lattice sieve from some other open-source NFS project, Msieve can do
everything else you need.

The following sections will walk though the process of performing an NFS
factorization using the Msieve demo binary, assuming you are running on
a single machine and have a small-to-medium-size problem to solve. I will
also explain the format of the intermediate files that are generated,
to allow the construction of compatible tools.


## Before you begin

One of the unfortunate side effects of cryptography in general, and RSA in
particular, is that it's just so damn cool. Factoring big numbers has an
undeniable whiff of the subversive to it, and lots of people are attracted
to that. More and more stuff in the information age is protected by RSA,
and that means more and more people have an interest in breaking it,
usually by factoring the RSA modulus. The number field sieve is the only
practical tool for breaking RSA; if you try anything else, you will fail.
Period. So, given that bank accounts, computer games, and factoring contest
prize money is all protected by RSA, and other NFS tools require a lot of
sophistication from the people using them, I suspect that flocks of people
will want to use Msieve to solve factoring problems that are much too large
(i.e. 155 digits and up).

If the above describes you, then for your own sake please take it slowly.
You will find out that the tools you need to use are all the result of
research efforts by disparate groups that are not very polished and not
explicitly meant to work together. This matters because in order to succeed
with a big factorization you will have to use multiple tools.


## Running NFS

Factoring an integer using NFS has 3 main steps:

1. Select Polynomial
2. Collect Relations via Sieving
3. Combine Relations

Msieve uses a line sieve for step 2, which is better than nothing but not
by much. The line sieve can run in parallel, which allows distributed sieving
much like QS does. Step 1 could theoretically run in parallel, but is not
currently set up to do so.

By default, the Msieve demo application will perform all steps in order.
Each step can also be performed individually, for anyone who is familiar
with how NFS works and needs maximum control of their factorization. The
combining phase can run either all at once or in three separate subphases.
Step 3 leaves intermediate information on disk that can be restarted from.
In addition, the sieving leaves a reminder on disk that allows restarting
the sieving from the previous point.

While the flow of work above is similar to the way the Quadratic Sieve code
works, the details are very different and in particular the amount of
non-sieving overhead in the Number Field Sieve is much higher than with
the Quadratic Sieve. This means that an NFS factorization is never going
to take less than a few minutes, even if the sieving finished instantly.
So running NFS will take longer than running QS until the number to be
factored is larger than a cutoff that depends on the quality of both the
NFS and QS code.

Right now, assuming you use somebody else's NFS lattice sieve, that cutoff
size is somewhere around 90 digits. Of course, if you used the super-fast QS
code in YAFU instead, the cutoff moves up to over 105 digits.  If you
further insisted on using only the crappy NFS sieving code that is in
Msieve the cutoff point will move up to maybe 120 digits. The difference
in runtime between the fastest tool and the slowest is a few hours for 100-
digit numbers, but at the 120-digit size you're looking at time differences
measured in days to weeks. So if the object is to minimize the time spent
crunching away on your computer, using the correct tool for the job can
save you massive amounts of work.

As a basic constraint, the NFS code in Msieve can run the combining phase
for any size factorization if you already have the results of step 1 and 2,
but steps 1 and 2 are only allowed to run on numbers larger than about
85 digits in size.


## Intermediate Files

The Msieve library needs to generate a large amount of intermediate
information in temporary files when running an NFS job. These files are
of three general types:

- a log file saves all the logging information generated by the library.
  This is useful for diagnostic purposes, and by default any factors found
  are logged. The default logfile name is 'msieve.log' in the directory
  where the Msieve demo binary is invoked. You can choose a different
  logfile name if you are running multiple copies of Msieve and want to keep
  all the intermediate output in the same directory, but don't want
  multiple Msieve runs to all go to the same logfile

- a factor base file; this is used to store the output from step 1, and
  stores internal information used in the sieving (if you use Msieve for
  step 2). Its default filename is 'msieve.fb' in the directory where the
  msieve demo binary is invoked. You can change the name of the factor base
  file to perform multiple factorizations in the same directory.

- a data file; this stores the relations generated from NFS sieving. Its
  default name is 'msieve.dat' and you can change the name to let multiple
  Msieve runs each use their own data file in the same directory. The
  combining phase (step 3) works from a single data file, and steps 1 and 3
  generate many different intermediate files whose names are of the form
  '<data_file_name>.<suffix>', all in the current directory in which
  Msieve is invoked. These intermediate files are used to restart different
  phases of the polynomial selection or combining phases. In principle,
  knowing the format of these intermediate files lets you substitute
  different NFS tools at particular stages of the NFS process.


## Configuring NFS Processing

For the largest factorizations, hard experience has shown that no amount of
my attempted cleverness will allow the code to guess the right thing to do by
itself, and you may have to help it out a little. Previous Msieve versions had
a very limited way of modifying the way the library worked, and so from v1.51
onward you can pass a free-form string into the library with an arbitrary list
of options set. The sections below will show what that string should contain
in order to assert your will upon the factorization process. In all cases, the
Msieve demo binary allows one string passed in, so when fiddling with multiple
different things you need to enclose them all in doublequotes. For example,
when specifying multiple options for NFS polynomial selection you will need
something like

```sh
	-np "option1=X option2=Y option3=Z"
```

Finally, note that the library only takes a single string with options, so if
you are using multiple NFS arguments (i.e. -np -ns for polynomial selection and
sieving) then all the options for all the arguments need to be passed in one
string.

## Polynomial Selection

Step 1 of NFS involves choosing a polynomial-pair (customarily shortened to
'a polynomial') to use in the other NFS phases. The polynomial is
completely specific to the number you need factored, and there is an
effectively infinite supply of polynomials that will work. The quality of
the polynomial you select has a dramatic effect on the sieving time; a
*good* polynomial can make the sieving proceed two or three times faster
compared to an average polynomial. So you really want a *good* polynomial,
and for large problems should be prepared to spend a fair amount of time
looking for one.

Just how long is too long, and exactly how you should look for good
polynomials, is currently an active research area. The approximate
consensus is that you should spend maybe 3-5% of the anticipated sieving
time looking for a good polynomial. Msieve's polynomial search code has
been the focus of a great deal of work, since this subject is more
interesting to me than most any other part of NFS.

The means by which one finds out how 'good' a polynomial is, is also an
active research area. We measure the goodness of a polynomial primarily by
its Murphy E score; this is the probability, averaged across all the
possible relations we could encounter during the sieving, that an 'average'
relation will be useful for us. This is usually a very small number, and
the E score to expect goes down as the number to be factored becomes larger.
A larger E score is better.

Besides the E score, the other customary measure of polynomial goodness
is the 'alpha score', an approximate measure of how much of an average
relation is easily 'divided out' by dividing by small primes. The E score
computation requires that we know the approximate alpha value, but alpha
is also of independent interest. Good alpha values are negative, and negative
alpha with large aboslute value is better. Both E and alpha were first
formalized in Murphy's wonderful dissertation on NFS polynomial selection.

With that in mind, here's an example polynomial for a 100-digit input
of no great significance:

```
R0: -2000270008852372562401653
R1:  67637130392687
A0: -315744766385259600878935362160
A1:  76498885560536911440526
A2:  19154618876851185
A3: -953396814
A4:  180
skew 7872388.07, size 9.334881e-014, alpha -5.410475, combined = 1.161232e-008
```

As mentioned, this 'polynomial' is actually a pair of polynomials, the
Rational polynomial R1 * x + R0 and the 4-th degree Algebraic polynomial

```
	A4 * x^4 + A3 * x^3 + A2 * x^2 + A1 * x + A0
```

Msieve always computes rational polynomials that are linear; the algebraic
polynomial is of degree 4, 5, or 6 depending on the size of the input.
Current Msieve versions will choose degree-4 polynomials for inputs up
to about 110 digits, and degree-5 polynomials for inputs up to about 220
digits. Finding polynomial pairs whose degrees are more balanced (i.e.
a rational polynomial with degree higher than 1) is another active research
area.

The 'combined' score is the Murphy E value for this polynomial, and is
pretty good in this case. The other thing to note about this polynomial-pair
is that the leading algebraic coefficient is very small, and each other
coefficient looks like it's a fixed factor larger than the next higher-
degree coefficient. That's because the algebraic polynomial expects the
sieving region to be 'skewed' by a factor equal to the reported skew above.
The polynomial selection determined that the 'average size' of relations
drawn from the sieving region is smallest when the region is 'short and
wide' by a factor given by the skew. The big advantage to skewing the
polynomial is that it allows the low-order algebraic coefficients to be
large, which in turn allows choosing them to optimize the alpha value.
The modern algorithms for selecting NFS polynomials are optimized to work
when the skew is very large.

NFS polynomial selection is divided into three stages. Stage 1 chooses
the leading algebraic coefficient and tries to find the two rational
polynomial coefficients so that the top three algebraic coefficients
are small. Because stage 1 doesn't try to find the entire algebraic
polynomial, it can use powerful sieving techniques to speed up this portion
of the search. When stage 1 finds a 'hit', composed of the rational
and the leading algebraic polynomial coefficient, stage 2 and 3 then find
the complete polynomial pair and try to optimize both the alpha and E
values. You can think of stage 1 as a very compute-intensive net that
tries to drag in something good, and stage 2 and 3 as a shorter but still
compute-intensive process that tries to polish things. Stage 2 attempts
to convert each hit from stage 1 into a complete polynomial with optimized
size, i.e. a polynomial whose values are unusually small on average. From
that starting point, stage 3 attempts to perturb the size optimized
polynomial so that it has slightly worse size but much better E-value.
A single polynomial from stage 2 can generate many polynomials in stage 3.

The Msieve demo binary can run the full polynomial selection algorithm
(all stages) when invoked with '-np'. With no other argument to -np,
the search starts from a leading algebraic coefficient of 1 and keeps
working until a time limit is reached. You can specify a range of leading
coefficients to search with '-np X,Y' and in this case the only limit
is on the time spent searching a single coefficient in that range. The time
limit assumes you are only using one computer to perform the search, which
is increasingly inappropriate when you get to large numbers (more than
`~140` digits).

If you have multiple computers to run the search, you will have to manually
split up the range of leading coefficients between them.  Note that when
the number to be factored is really large (say, 155 digits and up), the
search space is so huge that each coefficient in the range is broken up
into many pieces and only one piece, chosen randomly, is searched. This
lets you give multiple computers the same range of coefficients to search
and reasonably expect them to not duplicate each other's work.

By default, '-np' will run stage 1 and immediately run stages 2 and 3 on any
hit that is found. It is also possible to run each stage by itself, and
this can be useful if you want to do stage 1 on a graphics card (see below).
You run stage 1 alone with '-np1', and in that case any stage 1 hits
are written to a file <data_file_name>.m , where each hit is a line with
the triplet

<leading_alg_coefficient> R1 R0

and R0 is written as positive in the file. This is the same format used by
the polynomial selection tools in GGNFS, so stages 2 and 3 from either suite
can be run on output from either Msieve or GGNFS (or other packages, since
it should be easy to match up to the format above).

Stage 2, the size optimization, can be specified to the demo binary with
'-nps'; this reads lines of <data_file_name>.m and produces size-optimized
polynomials in <data_file_name>.ms, with each line having the form

<all algebraic coefficients by decreasing degree> R1 R0 alpha score

The 'alpha' is an approximate score for the alpha value and the score is an
average size of polynomial values (lower scores are better). The last two
items are optional, since size-optimized polynomials from other tools may
not know these numbers.

Finally, the root optimization can be specified with '-npr' given to the
demo binary. This reads <data_file_name>.ms and produces <factor_base_file>
and <data_file_name>.p; The first file is used by the rest of the NFS code,
and the second is for reference only. Currently <factor_base_file> gets
the polynomial with the highest E value, while the `*.p` file gets all the
complete polynomials found by stage 2 whose E-value exceeds a cutoff determined
internally. It is possible to specify several of the stages in a single
command line, i.e. "-np1 -nps" will perform stage 1 plus the size optimization,
and this is handy to avoid large intermediate output files from stage 1.

For the largest size problems, you will find that it is much faster to
run stage 1 and the size optimization together, then sort the size-optimized
results and run the root optimization only on the top few (dozens,
hundreds, perhaps thousands of) polynomials. This is because the internal
cutoffs for good scores are pretty loose for large problems, meaning that
large numbers of polynomials will make it to the root optimization stage
even though they are unlikely to produce the best overall polynomials.
Since the root optimization takes hundreds of times longer than the size
optimization, this will eat up large amounts of search time needlessly.

The code saves everything from '-npr' because it isn't entirely clear how
often the polynomial with the highest E score really sieves the fastest. The
E score was invented to reduce the number of polynomials that had to be
test-sieved, but Msieve's polynomial selection does not do test sieving,
and so for large input numbers it might be important to take the top few
polynomials and experiment with them to determine which will achieve the
highest sieving rate. Because this process is painful, for smaller inputs
you can just use the polynomial with the highest E-value.

There are many unanswered research issues with NFS polynomial selection, so
this stage allows a lot of tweaking using input arguments to the library.
Every run logs the values of these parameters which are used, allowing
you to adjust them in subsequent runs. The following configuration arguments
are usable by the library:

- `polydegree=X`
	- select polynomials with degree X
- `min_coeff=X`
	- minimum leading coefficient to search in stage 1
- `max_coeff=X`
	- maximum leading coefficient to search in stage 1
- `stage1_norm=X`
	- the maximum norm value for -np1
- `stage2_norm=X`
	- the maximum norm value for -nps and -npr
- `min_evalue=X`
	- the minimum E-value score of saved polyomials
- `poly_deadline=X`
	- stop searching after X seconds (0 means search forever)
- `X,Y`
	- same as `min_coeff=X max_coeff=Y`

The format of polynomials in `<data_file_name>.p` matches the format used
by the sieving tools in GGNFS. You can guess why :)


## Polynomial Selection Using Graphics Cards

Msieve's stage 1 polynomial selection code uses the improved algorithm
of Thorsten Kleinjung, as described at the 2008 CADO Workshop on Integer
Factorization. I realized in 2009 that it was possible to run stage 1
on graphics card processors (GPUs), though it's taken a huge amount of
work to get there and my original ideas were completely silly. GPUs are
cheap and have loads of compute units, allowing the possibility of higher
throughput than an ordinary desktop computer if used correctly.

So the Msieve source includes a branch that can drive an Nvidia GPU
to perform stage 1 of NFS polynomial selection. If the code is compiled
that way, running with '-np1' will use your GPU and output stage 1 hits
in the same format as described above. The GPU code will *not* in general
find the same polynomials that the CPU code does, so don't expect identical
output.

To run stage 1 this way, you have to have

- a graphics card with an Nvidia GPU

- the latest Nvidia drivers

- the latest Nvidia CUDA toolkit, or if using a precompiled binary
  the CUDA runtime API library from the latest Nvidia CUDA toolkit

The heart of stage 1 is a large collision search, where we're looking
for pairs of 64-bit numbers that match, in a large unsorted collection
of 64-bit numbers. Ordinarily the fastest way to perform that collision
search is to use a hashtable, but it turns out that GPUs have so much
memory bandwidth that it's better to just sort the list and look for
repeated numbers. Starting in 2011 there's been a lot of research into
making GPUs sort efficiently, and the end result is several third-party
libraries that can sort incredibly fast on latter-day graphics cards.

Msieve now includes the source for one of those libraries, from Back 40
Computing, and building the GPU-enabled version of Msieve will package
up the sorting into an externally callable library that is loaded at
runtime. The GPU branch also includes its own custom GPU kernels that
create at high speed the collections to sort, as well as postprocess
the sorted results on the GPU.

By default, the library will attempt to load the sort engine that is
appropriate for your GPU, but this can be overriden with additional
arguments:

- `sortlib=X`
	- use GPU sorting library in file X
- `gpu_mem_mb=X`
	- use at most X megabytes of GPU memory

Finally, if you actually have several graphics boards in your machine,
you can use the '-g' flag in the Msieve demo binary to choose one of those
boards out of the list available.

Anytime people find out you've moved code to a GPU, invariably there are
two burning questions. The first is 'How fast is it now', and while it
took several years and several 100% code rewrites to get to this point,
I can say with confidence that the speedup is just astounding. The GPU
code can finish in seconds an amount of work that takes minutes for the
CPU version of stage 1, and regularly works on the order of 50 to 100 *times*
faster. Not percent faster, *times* faster. It's so much faster than the
CPU code that running stage 1 on a GPU leaves the card mostly idle unless
you are factoring numbers larger than `~140` digits in size! To improve the
throughput on such small problems, Msieve's GPU polynomial selection is
multithreaded, and specifying '-t X' to the demo binary will set up X threads
to feed the GPU and one more to perform the size and root optimization.
Especially for smaller degree 5 problems, where the size of one block of
work is not nearly sufficient to keep a GPU busy, running with three or
four threads can dramatically improve the throughput of stage 1. For larger
problems, there is nothing to be gained by using more threads, as the
GPU is nearly fully busy.

The second burning question is 'you stupid loser, why don't you make this
thing work on my ATI card, using OpenCL or something?' I assure you that
I know ATI builds GPU processors, and that I can use languages like OpenCL
to run code on them.  Further, I know that computer hardware is a poor
subsititute for religion (I've read all the tiresome agruments on usenet:
Intel vs AMD, DDR vs Rambus, x86 vs Alpha...they produce massive volumes
hot air and little else).

I also know that ATI's implementation of OpenCL intentionally generates
slow code on their GPUs to avoid an incompatibility between the OpenCL
memory model and ATI hardware. Further, do you know of anybody who has
implemented multiple-precision arithmetic on an ATI GPU? The last time I
asked around, there was no way to access low-level quantities like carry
bits or high-half multiply results using OpenCL. Do you know of any efficient
radix-sorting libraries written for OpenCL? 90% of the performance in the
current implementation depends on a fast sort implementation, which is ready-
made for CUDA. The fact of the matter is that I only have the time to do
research on factorization algorithms; I regrettably don't have the time
to do fundamental research on GPU algorithms as well.

All of this may change in a year's time, when ATI's tools and the OpenCL
specification itself both become more mature, but I haven't chosen the tools
I use because I'm a dummy, and if you don't like my decision then you
should try to light a candle rather than curse the darkness like everyone
else does.


## Sieving for Relations

As mentioned in the introduction, Msieve only contains a line sieve. The
last few years have proved pretty conclusively that NFS requires a lattice
sieve to achieve the best efficiency, and the difference between good
implementations of line and lattice sieves is typically a factor of FIVE
in performance. That difference is just too large to ignore, so once again
I recommend that you not exclusively use Msieve for the sieving phase of
NFS. That being said, here are more details on the sieving implementation.

The Msieve demo binary will perform NFS (line) sieving when invoked with
'-ns X,Y', where X and Y specify the range of lines (inclusive) that will be
sieved. The sieving code reads the NFS factor base file produced by the
polynomial selection code, and writes `<data_file_name>` with any relations
found. If shut down gracefully, it also writes `<data_file_name>`.last, which
contains the index of the last line that was completely sieved, so that if
such a file exists then running the sieving again will pick up the computation
from the next line and not from the beginning.

The factor base file is the repository for a hodgepodge of different
information that is used by the various NFS modules of the Msieve library.
At a minimum, `<factor_base_file>` must contain

```
N <number_to_be_factored_in_decimal>
SKEW <skew_of_polynomial>
R0 <low_order_rational_poly_coefficient>
R1 <high_order_rational_poly_coefficient>
A0 <low_order_algebraic_poly_coefficient>
A1 <next_algebraic_coefficient>
A2 <next_algebraic_coefficient>
...
```

Polynomial coefficients can appear in any order, and missing coefficients
are assumed to be zero. The degree of the algebraic polynomial is determined
by its nonzero coefficient of highest degree. The SKEW is not currently used,
and is set to 1.0 if missing.

If you already have an NFS polynomial, for example because you are using
the Special Number Field Sieve or used another tool to generate it, then
creating a factor base file as above is sufficient to run the sieving on
that polynomial.

Running the sieving adds a lot more information to this file. The NFS factor
base is appended to the above, and also the parameters to be used when
running the line sieve. These parameters are set to internally generated
defaults, but may be overridden by specifying them explicitly in the factor
base file when sieving starts:

- FRNUM
	- Number of rational factor base entries
- FRMAX
	- The largest rational factor base entry
- FANUM
	- Number of algebraic factor base entries
- FAMAX
	- The largest algebraic factor base entry
- SRLPMAX
	- Bound on rational large primes
- SALPMAX
	- Bound on algebraic large primes
- SLINE
	- Sieve from -SLINE to +SLINE
- SMIN
	- Start of sieve line (-SLINE if missing)
- SMAX
	- End of sieve line (+SLINE if missing)

The first line in the factor base file that doesn't start with 'F' or 'S'
is assumed to be the beginning of the factor base to be used for sieving,
and if no such line exists then the factor base is created. Whenever the
sieving runs, it does the following during initialization:

- read N and the polynomial, check they are compatible
- generate default parameters
- override parameters if they exist in the file
- read in the factor base, or generate it if missing
- overwrite `<factor_base_file>` with polynomial, current parameters and the newly generated factor base

Sieving in NFS works by assuming the rational and algebraic polynomials are
in some variable x, then replacing x by the fraction a/b, where a and b are
integers that don't have factors in common. Line sieving fixes the value
of b and then looks for all the values of 'a' between -SLINE and +SLINE
where the homogeneous form of the rational polynomial

```c
	b * R(a/b)
```

and the homogeneous form of the algebraic polynomial

```c
	b^(degree(A)) * A(a/b)
```

both factor completely into small primes. Any (a,b) pair that factors
completely in this way is a relation that can proceed to the combining
phase, and you may need millions, even billions of such relations for
the combining phase to succeed in factoring N.

Relations are written to <data_file_name>, one relation per line. The first
line of <data_file_name> is set to 'N <number to be factored in decimal>'; if
sieving restarts and the first line of the data file is not as specified,
the sieving will assume these are relations from another factorization
and will truncate the file. Don't worry, if you are only running the
combining phase this behavior is disabled, and you can give Msieve a
file with just the relations.

Relations are written to the file in GGNFS format. A relation in GGNFS
format looks like:

```
a,b:rat1,rat2,...ratX:alg1,alg2,...ratY
```

where

- **a** is the relation 'a' value in decimal

- **b** is the relation 'b' value (must be between 1 and 2^32-1)

- **rat1,rat2,...ratX** is a comma-delimited list of the X factors of the
		homogeneous form of the rational NFS polynomial, each
		written in hexadecimal (without any '0x' in front)

- **alg1,alg2,...algY** is a comma-delimited list of the Y factors of the
		homogeneous form of the algebraic NFS polynomial, each
		written in hexadecimal (without any '0x' in front)

Factors in each list can occur in arbitrary order, and a given factor only
needs to occur in the list once even if it divides one of the polynomials
multiple times. In addition, to conserve space in what can potentially be
very large text files, factors less than 1000 can be completely omitted
from the factor lists. The combining phase in Msieve will recover these
small factors manually, by trial division.

The Msieve, CADO-NFS and GGNFS sieving tools all output relations
in this format, so that any of these tools can use relations generated
by any of the other tools.

The sieving code will insist on choosing for itself the number of large
primes and the cutoffs for using non-sieving methods. This is because it
is supposed to choose these based on the size of numbers actually
encountered during the sieving. In addition, the sieving code uses a
batch factoring algorithm due to Dan Bernstein, which makes it possible
to find relations containing three algebraic and/or rational large primes
with a minimum of extra effort over and above the time ordinarily needed
to find relations with only two large primes. This unfortunately means
that the line sieve uses up an unusually large amount of memory, up to
several hundreds of megabytes even for medium-size problems.


## Distributed Computing

As with the quadratic sieve module, it is possible to use multiple computers
to speed up the sieving step, which is by far the most time-consuming part
of the NFS process. A straightforward recipe for doing so is as follows:

- Run Msieve once and specify that only polynomial
  generation take place. This will produce a tiny
  factor base file containing the selected polynomial.

- Make a copy the factor base file for each copy of Msieve
  that will be sieving.

- Start each copy of Msieve with a different range to sieve.
  Each copy will automatically generate its own factor base
  and stick it into the factor base file

Some notes on this process:

1. You can always just make up the polynomial you want Msieve to use,
   instead of waiting for the library to generate its own. This is desirable
   if you already know the polynomial to use. Stick the polynomial into
   a text file and run the recipe like normal.

2. NFS works better if you budget a sizable chunk of time for selecting
   polnomials. If you're impatient, or just want something for a quick
   test, interrupting Msieve while polynomial generation is in progress
   will immediately print the current best polynomial to the factor base
   file. You can also put a time limit on polynomial generation from
   the command line.

3. Because Msieve uses a line siever, the range to sieve is measured in
   'lines' This is a number 'b' between 1 and infinity, and specifying
   the range to sieve involves just specifying a starting and ending value
   of b

4. *Unlike* the quadratic sieve, the rate at which relations accumulate
   is not constant. Small b values will generate many more relations than
   large b values. Further, the library cannot just make up work to do
   at random because it's likely that different sieving machines will
   repeat each other's work. This means that for NFS, the bookkeeping
   burden is on the user and not on the computer. One way to handle
   this is to have a script assign relatively small ranges of work when
   it notices sieving machines finishing their current range. A much
   better way, not implemented, would be for the library to be told how
   many machines are sieving and which number (1 to total) identifies
   the current sieving machine. Then each sieving machine only does 1
   out of every 'total' sieve lines. This automatically balances the
   load fairly with no bookkeeping overhead, as long as all the sieving
   machines are about the same speed.

5. If interrupted, a sieving machine will complete its current line and
   state the line that it finished. A script can then parse the logfile
   or the screen output and use that to restart from that point later on.


## NFS Combining Phase

The last phase of NFS factorization is a group of tasks collectively
referred to as 'NFS postprocessing'. You need the factor base file described
in the sieving section (only the polynomial is needed, not the actual
factor base entries), and all of the relations from the sieving. If you
have performed sieving in multiple steps or on multiple machines, all
of the relations that have been produced need to be combined into a single
giant file. And by giant I mean *really big*; the largest NFS jobs that
I know about currently have involved relation files up to 100GB in size.
Even a fairly small 100-digit factorization generates perhaps 500MB of
disk files, so you are well advised to allow plenty of space for relations.
Don't like having to deal with piling together thousands of files into
one? Sorry, but disk space is cheap now.

With the factor base and relation data file available, the Msieve demo binary
will perform NFS postprocessing with the '-nc' switch. For smaller jobs,
it's convenient to let the library do all the intermediate postprocessing
steps in order. However, for larger jobs or for any job where data has
to be moved from machine to machine, it is probably necessary to divide
the postprocessing into its three fundamental tasks. These are described
below, along with the data they need and the format of the output they
produce.


## NFS Filtering

The first phase of NFS postprocessing is the filtering step, invoked by
giving '-nc1' and not '-nc' to the demo binary. This analyzes the input
relation file, sets up the rest of the filtering to ignore relations
that will not be useful (usually 90% of them or more), and produces
a 'cycle file' that describes the huge matrix to be used in the next
postprocessing stage.

To do that, every relation is assigned a unique number, corresponding to
its line number in the relation file. Relations are numbered starting
from zero, and part of the filtering also adds 'free relations' to the
dataset. Free relations are so-called because it does not require any
sieving to find them; these are a unique feature of the number field
sieve, although there will never be very many of them.

Filtering is a very complex process, and the filtering in Msieve is designed
to proceed in a fully automated fashion.  The intermediate steps of Msieve's
filtering are not designed to allow for user intervention, although it
may be instructive to build compatible filtering tools. There are a few
configuration arguments that can be passed to Msieve's filtering:

- `filter_mem_mb=X`
	- try to limit filtering memory use to X megabytes
- `filter_maxrels=X`
	- limit the filtering to using the first X relations in the data file
- `filter_lpbound=X`
	- have filtering start by only looking at ideals of size X or larger
- `target_density=X`
	- attempt to produce a matrix with X entries per column
- `max_weight=X`
	- for datasets with many extra relations, start partial merging with ideals of weight up to X
- `X`, `Y`
	- same as `filter_lpbound=X filter_maxrels=Y`

Ordinarily you would want to use all relations, since you spent the time
to compute them in the first place, but sometimes the other postprocessing
stages have trouble when given too many relations to deal with, and this
lets you limit the dataset size without having to manually trim relations
out of the data file.

'target_density=X' controls how hard the filtering will work to produce a
matrix that is small. Setting X to a value larger than the default of 70.0
will cause the memory use of the filtering to be possibly much higher, and
you need to make sure you have a large number of excess relations, over and
above the number needed for filtering to converge to a reasonable matrix.
In practice, setting the target density to a value over 130.0 will cause your
computer to run out of memory for large problems. Nonetheless, for the
largest problems making the filtering work harder can save a noticeable
amount of time in the linear algebra.

If you do not have enough relations for filtering to succeed, no output
is produced other than complaints to that effect. If there are 'enough'
relations for filtering to succeed, the result is a 'cycle file'. This
is a binary file named '`<data_file_name>.cyc`', full of 32-bit integers
in the native byte order of the filtering machine. The format is as follows:

- The first 32-bit word gives the number of matrix columns C that the
  linear algebra should expect for the matrix generated by the filtering

- Then the file has C column descriptors. A descriptor has a 32-bit word
  giving the number of relations that contribute to the current matrix
  column, and then an unordered list of the (32-bit) relation numbers
  from <data_file_name> that appear in the current column.

The use of 32-bit relation numbers means that the filtering cannot currently
handle more than about 4 billion relations.

How many relations is 'enough'? This is unfortunately another hard question,
and answering it requires either compiling large tables of factorizations of
similar size numbers, running the filtering over and over again, or
performing simulations after a little test-sieving. There's no harm in
finding more relations than you strictly need for filtering to work at
all, although if you mess up and find twice as many relations as you need
then getting the filtering to work can also be difficult. In general the
filtering works better if you give it somewhat more relations than it
strictly needs, maybe 10% more. As more and more relations are added, the
size of the generated matrix becomes smaller and smaller, partly because the
filtering can throw away more and more relations to keep only the 'best'
ones.

Of course finding the exact point at which filtering begins to succeed
is a painful exercise as well :)


## NFS Linear Algebra

The linear algebra step constructs the matrix that was generated from the
filtering, and finds a group of vectors that lie in the nullspace of that
matrix. Matrices generated by Msieve are slightly wider than they are tall,
which means that a properly constructed matrix will have C columns and
perhaps (C-100) rows. The value of C goes up as the number to be factored
gets larger, and very large factorizations can generate huge matrices. The
largest value of C that I know of is around 93 million; at that size you'd
need 27GB of memory just to store it.

Needless to say, finding nullspace vectors for a really big matrix is an
enormous amount of work. To do the job, Msieve uses the block Lanczos algorithm
with a large number of performance optimizations. Even with fast code like
that, solving the matrix can take anywhere from a few minutes (factoring
a 100-digit input leads to a matrix of size `~200000`) to several months
(using the special number field sieve on 280-digit numbers from the
Cunningham Project usually leads to matrices of size `~18` million). Even
worse, the answer has to be *exactly* correct all the way through; there's
no throwing away intermediate results that are bad, like the other NFS
phases can do. So solving a big matrix is a severe test of both your computer
and your patience.

The Msieve demo binary performs NFS linear algebra when run with '-nc2'.
This requires all the relations from `<dat_file_name>` and also the cycles
generated by the filtering and previously written to '`<dat_file_name>.cyc`'.
When complete, the solver writes the nullspace vectors to a 'dependency
file' whose name is '`<dat_file_name>.dep`'; this is a file full of 64-bit
words, in the native byte order of the machine, with one word for each of
the C columns in the matrix. If the solver found D solutions, then the
D low-order bits of each word matter. If bit i in word j is a 1, then the
j_th column of the matrix participates in solution number i. Since the
cycle file gives the relation numbers for matrix column j, the dependency
file lets you construct a collection of relations that will make the NFS
square root step (described next) work correctly. Since the dependency
file is composed of 64-bit words, there can never be more than 64 nullspace
solutions; usually there will be 25-35 of them, and each of these has about
a 50-50 chance of factoring the input number at the conclusion of the NFS
square root step.

The low-level details of the linear algebra are very messy; most of the time
goes into a sparse matrix multiply, and the code uses blocking to make as
much use of cache as possible. The choice of block size (in 64-bit words) is
determined automatically, but if a bad choice is made it can be overridden
via configuration strings:

- `la_block=X`
	- set the L1 block size to X (default 8192, should fit in L1 cache but not be too small)
- `la_superblock=X`
	- set the L2 block size to X (default is 3/4 of the largest cache detected)

Both the matrix and all of the solutions are numbers in a finite field of
size 2, so if a matrix entry or any solution entry is not zero, then it has
to be 1. Hence we don't need to explicitly store the value at a particular
position in the matrix, since all the nonzero matrix values will always be 1.
This lets us get away with only storing the *coordinates* of nonzero matrix
entries. To do that, the linear algebra first constructs the complete matrix
that will be solved and writes it to disk, as a file named '`<dat_file_name>.mat`'.
Then it optimizes the matrix slightly, possibly rearranging the matrix
columns, and writes the new matrix and a new cycle file that is optimized
the same way. The linear algebra does not require the cycle file contents
when the solver is running, only during the initial matrix build.

While the actual .mat file is only useful for the linear algebra phase, other
tools that can analyze or permute or even solve the matrix could benefit from
being able to parse the .mat file. To that end:

The matrix file is an array of 32-bit words, with byte ordering determined
by the machine that produced the matrix (it will be little-endian for just
about everybody). The nonzero matrix entries are stored by column, because
they are easy to generate that way. The first word in the file is the number
of rows, the second is the number of dense rows ('D'), and the third is the
number of matrix columns. Each column is then dumped out. For each column,
there is a 32-bit word indicating the number of sparse entries in the column,
then a list of that many nonzero row positions (each a 32-bit word, with
words in arbitrary order), then floor((D+31)/32) words for the dense rows
in that column. The latter should be treated as an array of D bits in
little-endian word order.

Rows are numbered from 0 to (number of rows - 1). Rows 0 to D-1 are considered
dense, with a set bit in the bitfield indicating a nonzero entry for that
column. The sparse entries in each column are numbered from D onward.

There are two tools in the linear algebra to make a long-running, error-prone
process like finding nullspace vectors more robust. The first is a periodic
integrity check on the current solution, which happens automatically. This
is very good at detecting computation or memory errors that occur for whatever
reason (heat, marginal timing, overclocking, cosmic rays, etc). The second
tool is periodic checkpointing; for matrices larger than about a million
columns, the solver will periodically (about once per hour) package up the
entire current state of the solver and dump it to a checkpoint file. The most
recent pair of checkpoint files is saved, and either one can be used to restart
the solver at a later time. The demo binary also installs a signal handler
that can catch Ctrl-C or process termination interrupts, and will generate
a checkpoint immediately before quitting. The two checkpoint files are named
'`<dat_file_name>.chk`' and '`<dat_file_name>.bak.chk`' although Msieve can only
restart from files named like the former.

Checkpoint files are not very large, about 60x the number of columns in
the matrix, but they are completely specific to a given matrix. Mess up
the .mat file, permute it in any way, and the checkpoint will not work
anymore. This also extends to restarting the linear algebra when you only
meant to restart from a checkpoint; do not assume that the linear algebra
will generate the exact same matrix two times in a row. To restart the linear
algebra from a checkpoint file, run the Msieve demo binary with '-ncr'
instead of '-nc2'.


## Multithreaded Linear Algebra

The linear algebra is fully multithread aware, and if the demo binary is
started with `-t X` then the matrix solver uses X threads. Note that the
solver is primarily memory bound, and using as many threads as you have
cores on your multicore processor will probably not give the best performance.
The best number of threads to use depends on the underlying machine; more
recent processors have much more powerful memory controllers and can continue
speeding up as more and more threads are used. A good rule of thumb to start
off is to try two threads for each physical CPU package on your motherboard;
even if it's not the fastest choice, just two or four threads gets the vast
majority of the potential speedup for the vast majority of machines.

There are a few other things to note about the multithreading. Multiple
threads are only used for matrices larger than about 250k columns, regardless
of what you specify on the command line. A matrix that small only takes 10 or
15 minutes to solve with one thread, so this isn't exactly a hardship. Also,
if the matrix is large enough to have checkpointing turned on, the format of
the matrix file is completely independent of the number of threads specified.
So you can build the matrix once, interrupt the solver so that a checkpoint is
generated, then restart from the checkpoint with a different number of threads.
This lets you experiment with different runtime configurations without chewing
up large amounts of time and memory rebuilding the matrix needlessly.

Finally, note that the matrix solver is a 'tightly parallel' computation, which
means if you give it four threads then the machine those four threads run on
must be mostly idle otherwise. The linear algebra will soak up most of the
memory bandwidth your machine has, so if you divert any of it away to something
else then the completion time for the linear algebra will suffer.

As for memory use, solving the matrix for a 512-bit input is going to
require around 2GB of memory in the solver, and a fast modern processor running
the solver with four threads will need about 36 hours. A slow, less modern
processor that is busy with other stuff could take up to a week!


## Parallel Linear Algebra with MPI

The most effective way to speed up the linear algebra solver is to throw more
bus wires at it, and that means harnessing the power of multiple separate
computers to solve a single matrix. As of August 2010 Msieve now includes
a high-performance MPI driver that is the focus of current research by several
people and myself. So the following describes an experimental but very
powerful way of running the linear algebra.

To use MPI, you have to have an MPI library installed, or work somewhere that
does. I only have practice with OpenMPI on my one test rig, but it's straight-
forward to get started with that combination. Just because you use MPI doesn't
mean you have to have a cluster to run it on; MPI abstracts away any underlying
hardware from the number of separate copies of your program that it runs.
This is not the place to describe how to set up your actual hardware and
software for a MPI run, because there are too many details to deal with.

Assuming that you have done so, though, and have built an MPI-aware copy
of the Msieve library, then you can solve a matrix on an M x N grid of MPI
processes by running the demo binary with '-nc2 M,N' and giving the resulting
command line to your MPI library's parallel launch script, i.e. to 'mpirun'
if that's what your MPI uses.

All the MPI processes need access to a shared directory containing the .mat
file, and each MPI process gets its own logfile by appending a number to the
base logfile name. The matrix file on disk needs to be constructed with MPI in
mind; the format of the .mat file is unchanged, but the matrix building code
adds an auxiliary file called `<dat_file_name>.mat.idx` that gives the file
offsets each MPI process will use to read in its own chunk of the .mat file.
The offsets are chosen so that each MPI process gets about the same number
of nonzero entries in its chunk of the matrix, and a single auxiliary file
can handle any decomposition of the matrix up to a 35x35 MPI grid.

Currently the matrix file itself is constructed by only one MPI process,
the others wait on a barrier until it finishes. This is wasteful, especially
if you are using someone else's cluster time. If the matrix is built already,
you can start the MPI linear algebra but not actually build the matrix by
giving '-nc2 "skip_matbuild=1 M,N" ' to the demo binary. This lets you build
the matrix on a single machine, then stop the run and reuse the matrix on
a cluster.

If you run MPI Msieve with the '-t X' argument as well, *each* MPI process
runs its part of the solver with X threads. If you expect one MPI process
will use one physical machine, this is a handy way of soaking up more of the
memory bandwidth that machine has to offer. Using X threads on a MxN grid
means you have `M*N*X` threads running, but you can also increase M and N
so that X is 1 and more of the computation is exposed to your MPI library.
In general there's a tradeoff between using more MPI processes and fewer
processes with more threads, and the best combination depends on your hardware
and the efficiency of your MPI library. If you are using a large shared-memory
NUMA machine to run MPI linear algebra, experience has shown that the machine
must be as idle as possible (even background or scheduled idle-time tasks
will throw off the linear algebra timing), and you will still need various
black-magic arguments to your MPI launch system to pin MPI processes to
phyiscal cores, and prevent them from migrating.

The on-disk format of the matrix is independent of the choice of M and N,
and checkpointing is still available to interrupt an MPI run in progress.
However, there are some caveats to relying on checkpoints with an MPI run.
First of all, a checkpoint file is not produced when you interrupt a run in
progress; I don't know how to do this in a portable way because different
MPI libraries handle the propagation of signals to MPI processes in
different ways, and you won't be able to trust a checkpoint file to be
constructed correctly with that restriction. So you only get a checkpoint
when it is periodically written, usually once an hour. The other restriction
with checkpoint files is that while you can change the number of threads
and restart from a checkpoint file, if the file was written by an MxN grid
of MPI processes then you can only change N when you restart from that
checkpoint. This is because the matrix reading code permutes the matrix rows
to try to equalize the distribution of nonzero matrix elements across each
of the M MPI processes in an MPI column, and that makes the checkpoints
specific to the value of M used.

You should not expect a linear speedup when using P processes. The actual
speedup you will get depends almost completely on the interconnect between
your compute nodes, and only then on how fast each node is. With a cluster
of PCs, there is a huge difference in performance between using gigabit
ethernet for an interconnect and using infiniband; the latter is much faster
and scales better too. We've experimentally found that the best decomposition
for P MPI processes will reduce the solve time by a factor of about P^0.6
using gigabit ethernet, while an infiniband interconnect scales the solve
time by around P^0.71

For the record, the largest matrix Msieve has currently been used on had about
93 million columns, with the solution taking about 50 days running on 576 cores.
(It could theoretically have finished much faster, but had to stall due to
various cluster management issues). The same code has solved several matrices
with 30 to 40 million columns, plus another monster with 63 million columns.
A notable case used an earlier version of the code to solve a matrix with 43
million columns, using a 30x30 grid of MPI processes on a supercomputer at
Moscow State University, in just 63 hours.


## NFS Square Root

With the solution file from the linear algebra in hand, the last phase of
NFS postprocessing is the square root. The quadratic sieve also has a square
root phase, and it's trivial for that algorithm, but the square root within
NFS is kind of a big deal. Not because it takes a huge amount of time compared
to the other postprocessing phases, but because traditionally what we call
'an NFS square root' is actually two square roots, an easy one over the
integers and a very complex one over the algebraic number field described
by the NFS polynomial we selected. Traditionally, the best algorithm for
the algebraic part of the NFS square root is the one described by Montgomery
and Nguyen, but that takes some quite sophisticated algebraic number theory
smarts. Msieve was the first widely available NFS package that proved it
was also straightforward to use a more basic, more memory-intensive algorithm
for the algebraic square root, as long as high-performance large-integer
arithmetic was available. We're a little spoiled in the 21st century for
all those things, since FFT multiplication is much better understood now
and computers are a million times faster than they were in the early 1990s
when the basic algorithm was first invented. The bottom line is that if
you have enough memory to solve the linear algebra, then you also have enough
memory to compute the square root in the basic way.

Every solution generated by the linear algebra is called 'a dependency',
because it is a linearly dependent vector for the matrix we built. The
square root in Msieve proceeds one dependency at a time; it requires all the
relations from the data file, the cycle file from the filtering, and the
dependency file from the linear algebra. Technically the square root can
be sped up if you process multiple dependencies in parallel, but doing
one at a time makes it possible to adapt the precision of the numbers used
to save a great deal of memory toward the end of each dependency.

The NFS square root is run by giving '-nc3' to the demo binary. This will
process all dependencies in order, one at a time. Nonexistent dependencies
are (obviously) skipped. You can also specify a range of dependencies with
configuration options:

- `dep_first=X`
	- start at dependency `X`, `1<=X<=64`
- `dep_last=Y`
	- end with dependency `Y`, `1<=Y<=64`
- `X`, `Y`
	- same as `dep_first=X dep_last=Y`

If you have multiple separate machines, you can give each one its own
(X,Y) range, since each dependency is completely independent of the others.

Each dependency has a 50% chance of finding a nontrivial factorization of the
input. The code includes a great deal of consistency checking and error
checking throughout the square root phase, because a lot can go wrong in this
or one of the earlier NFS phases. So if you don't see any errors but
no factors are reported either, then that dependency was computed correctly
but was just unlucky. If you do see an error, the only information you really
have is that the final answers computed by the square root will not work,
and this unfortunately doesn't give any clue about what is really wrong.

Very rarely, the special number field sieve will use polynomials that are
slightly 'bad' in that you cannot find a prime number p for which the algebraic
polynomial modulo p is irreducible. Since the basic algorithm really wants
such a p, Msieve will complain and then try to find a substitute p that
will work anyway. Thus far it has always succeeded, but in the meantime these
kinds of pathological polynomials will generally need more dependencies
before the input is factored. Virtually all such pathological polynomials
have degree 4, though once in a great while you can find one with degree
6 or even 8.

There are many optimizations that are still possible for the NFS square root
in Msieve, both to reduce memory use and increase efficiency. A single
dependency can be multithreaded in a straightforward way, and removing some
error checking would allow a reduction in the volume of data read from disk.
Likewise, the CADO toolkit modifies the basic square root algorithm to use
multiple primes p, perform the square root for each of them (possibly on
separate machines), and then combine the results in a way keeps memory
use down even further. All of these innovations would be nice to add to the
current Msieve code, but they only make a difference for extremely large
problems, and even then the algorithm as currently implemented needs a few
gigabytes of memory and perhaps 8 hours of runtime per dependency. So
historically the square root has not been a focus of much of my development
time.
