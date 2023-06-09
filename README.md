# MSIEVE: A Library for Factoring Large Integers

Jason Papadopoulos


## Introduction

Msieve is the result of my efforts to understand and optimize how integers are factored using the most powerful modern algorithms.

This documentation corresponds to version 1.54 of the Msieve library. Do not expect to become a factoring expert just by reading it. I've included a relatively complete list of references that you can and should look up if you want to treat the code as more than a black box to solve your factoring problems.


## What Msieve Does

Factoring is the study (half math, half engineering, half art form) of taking big numbers and expessing them as the product of smaller numbers.

If I find out `15 = 3 * 5`, I've performed an integer factorization on the number 15. As the number to be factored becomes larger, the difficulty involved in completing its factorization explodes, to the point where you can invent secret codes that depend on the difficulty of factoring and reasonably expect your encrypted data to stay safe.

There are plenty of algorithms for performing integer factorization.

The Msieve library implements most of them from scratch, and relies on optional external libraries for the rest of them.

Trial division and Pollard Rho is used on all inputs; if the result is less than 25 digits in size, tiny custom routines do the factoring.

For larger numbers, the code switches to the **GMP-ECM** library and runs the **P-1**, **P+1** and **ECM** algorithms, expending a user-configurable amount of effort to do so.

If these do not completely factor the input number, the library switches to the heavy artillery. Unless told otherwise, Msieve runs the self-initializing **Quadratic Sieve** (QS) algorithm, and if this doesn't factor the input number then you've found a library problem.

If you know what you're doing, Msieve also contains a complete implementation of the **Number Field Sieve** (NFS), that has helped complete some of the largest public factorization efforts known. Information specific to the quadratic sieve implementation is contained in [Readme.qs](README.QS.md), while the number field sieve variant is described in [Readme.nfs](README.NFS.md)

The maximum size of numbers that can be given to the library is hardwired at compile time. Currently the code can handle numbers up to `~310` digits; however, you should bear in mind that I don't expect the library to be able to complete a factorization larger than about `120` digits by itself.

The larger size inputs can only really be handled by the number field sieve, and the NFS sieving code is not efficient or robust enough to deal with problems larger than that. Msieve *can* complete very large NFS factorizations as long as you use the NFS sieving tools from other open-source packages.


## Goals

Msieve was written with several goals in mind:

- **To be fast.** I originally started this project in 2003 because I was dismayed at the lack of progress in high-performance implementations of the quadratic sieve. By 2006 msieve was the fastest quadratic sieve available anywhere, by a wide margin. Libraries like YAFU have since pushed the performance envelope for the quadratic sieve much harder, and the focus in msieve has shifted to the number field sieve.

- **To be complete and comprehensive.** I've tried to make as many parts of the library state-of-the-art as possible. Parts of the NFS code have turned into research platforms in their own right.

- **To be simple to use.** The only input is the integer to be factored. Everything else happens automatically, though a certain amount of user control is possible if you know what you're doing.

- **To be free (as in beer).** The entire code base is released into the public domain. This is hobby stuff for me, and the more it's used the better. I have made every effort to avoid using external code that is licensed under the GPL, opting instead for less restrictive licenses.

If you choose to use Msieve, please let me know how it goes. I welcome bug reports, suggestions, optimizations, ports to new platforms, complaints, boasts, whatever.


## Getting Msieve

My web page ([www.boo.net/~jasonp][jasonp.site]) used to be the main distribution venue for Msieve source and binaries. As the codebase has grown and more people have voluneered to help with it, it became less and less convenient to base things there, and so Msieve development and binary releases are now available on SourceForge ([msieve.sourceforge.net][msieve.sf]).

The source distribution comes with a unix [makefile][msieve.makefile] you can use if you want to build msieve from source. If you have Microsoft Visual Studio, Brian Gladman has kindly provided a set of build files that will generate Windows binaries.


## Using Msieve

Just to be confusing, there are two things that I call 'Msieve' interchangeably.

The source distribution builds a self-contained static library 'libmsieve.a', that actually performs factorizations, and also builds a 'msieve' demo application that uses the library. The library has a very lightweight inter- face defined in [msieve.h][msieve.header], and can be used in other applications. While the demo application is (slightly) multithreaded, most the library is single- threaded and all of its state is passed in. The linear algebra code used in the quadratic- and number field sieve is multithread aware, and the entire library is supposed to be multithread-safe.

The demo application has only one job: to act as a delivery vehicle for integers to be factored. Numbers can come from a text file, from redirected input, from the command line directly, or can be manually typed in one at a time. Batch input is also supported, so that you can just point the application to a text file with a collection of numbers, one per line. By default, all output goes to a logfile and a summary goes to the screen.

For the complete list of options, try:

```sh
msieve -h
```

## Arithmetic expressions

Starting with v1.08, the inputs to msieve can be integer arithmetic expressions using any of the following operators:

|		|		|				|				|
|---:	|:---	|-------------	|------------	|
|+	|-	|plus, minus			|(lowest priority)
|%	|	|integer remainder
|*	|/	|multiply, integer divide
|^	|	|power
|(	|)	|grouping				|(highest priority)

Hence for example:

```
	(10^53 - 1) / 9
```

gives the value:

```
11111111111111111111111111111111111111111111111111111
```

The integers used in an expression can be of any length but all intermediate results and the final result are restricted to 275 or less decimal digits.


## Intermediate information

While factoring an integer, the library can produce a very large amount of intermediate information. This information is stored in one or more auxiliary savefiles, and the savefiles can be used to restart an interrupted factorization. Note that factoring one integer and then another integer will overwrite the savefiles from the first integer.


## Memory usage

The amount of memory that's needed will depend on the size of the number to be factored and the algorithm used. If running the quadratic sieve or the number field sieve, the memory requirements increase towards the end of a factorization, when all of the intermediate results are needed at the same time. For a 100-digit quadratic sieve factorization, most of the time Msieve needs 55-65MB of memory, with the last stage of the factorization needing 100-130MB. The final part of the number field sieve can use up incredible amounts of memory; for example, completing the factorization of a 512-bit number like an RSA key needs about 2GB of memory.

\pagebreak
## Frequently Asked Questions

### Factor much bigger numbers

**Q:** I want to factor much bigger numbers. Can't Msieve solve problems larger than you say?

**Q:** I'm trying to break my ex-girlfriend's RSA key with Msieve, and it's not working. What's wrong?

**A:** The quadratic sieve really is not appropriate for factoring numbers over `~110` digits in size. On a fast modern CPU, a 110-digit QS factorization takes nearly 120 hours for Msieve, and the time increases steeply beyond that. If you have really big factorization needs, there are several packages that you can use to make up for the low-performance parts of Msieve's NFS code:

- [CADO-NFS][cado.nfs.web] is a complete NFS suite that includes state-of-the-art (or nearly so) implementations of all phases of the Number Field Sieve. The sieving tools in CADO-NFS produce output compatible with GGNFS and Msieve, so it's up to you which package you want to use for a given phase of NFS. In addition, CADO-NFS is under constant very active development, and in my opinion represents the future of the high-performance NFS community.

- [GGNFS][ggnfs.sf] has state-of-the-art tools for NFS sieving and NFS polynomial selection. Using the latter is optional, since Msieve actually has very good NFS polynomial selection code, but the former is an absolute must if you are factoring numbers larger than about 95 digits.

- [Factor-by-GNFS][factor-by-gnfs.sf] is an older, separate package that's not really compatible.

See later for a list of web resources that give an idea of the largest actual factorizations that Msieve has performed.


### Network aware client-server Msieve version

**Q:** Can you make Msieve network aware? Can you make it a client-server thingy? Can I use the internet to factor numbers?

**A:** The demo application for the Msieve library is just that, a demo. I don't know anything about network programming and I'm not qualified to build a client-server application that's safe in the face of internet threats. If you have these kinds of smarts, you can use Msieve in your own code and I'll help as much as I can. The demo is good enough for people with a few machines on a small private LAN, and this is `~100%` of the user community right now.


### Library documentation

**Q:** Where's all the documentation on how your library actually works?

**A:** There really isn't any. Msieve is a spare-time project, and I have so little spare time that if I tried to produce formal documents on the internals of the library then I'd never have time to actually work on those internals. The parts of the code that are relatively stable have extensive in-line documentation, but whatever I happen to be working on currently is pretty much comment-free. If you have specific questions about how a specific part of the code works, just ask...


### Msieve on a cluster

**Q:** How can I modify Msieve to work on a cluster?

**A:** Distributed sieving is so easy that you don't need high-performance parallel programming techniques or message passing libraries to do it. If you're lucky enough to have a cluster then the batch scheduling system that comes with the cluster is more than enough to implement cluster-aware sieving. Of course if you have access to that much firepower you owe it to yourself to use an NFS package of some sort.


### Multi-core processors support

**Q:** Can you modify Msieve to run on multi-core processors?

**A:** As described above, the really intensive part of the QS and NFS algorithms is the sieving, and it's a waste of effort to multithread that. You won't save any time compared to just running two copies of Msieve. The final stage *can* benefit from multithreading, and the intensive parts of that are already multithread-aware. This can be improved, but multithreading more parts of the library is a low priority for me.


### License

**Q:** Why put Msieve into the public domain and not make it GPL? Wouldn't GPL-ed code protect your work and encourage contributions?

**A:** Msieve is a learning exercise, not a collaborative effort per se. I don't expect other developers to help, though many have and it's appreciated. As for protection, there's only one way to generate income from this code: use it to win factoring contests. While the number field sieve can win factoring contests, you personally do not have the resources to do so. So there's no reason to put licensing restrictions on this code.


### Multiple precision library

**Q:** Your multiple precision library sucks. Why didn't you use GMP?

**A:** I know it sucks. Using GMP would have left MSVC users in the cold, and even building GMP is a major exercise that requires essentially a complete and up-to-date unix environment. The nice thing about sieve- based factoring is that for big factorizations the actual multiple precision math takes about 1% of the total runtime, so it actually doesn't matter all that much which low-level multiple precision library is used.

Actually, the parts of the code that use my own library are pretty old, and new development does require GMP. This became necessary after modifications to the NFS code started to require arithmetic on very large numbers, so GMP is now a requirement. One of my eventual goals is to phase out (most of) my own bignum library in favor of GMP.

\pagebreak
## Credits

Especially as the code became more useful, credit is due to several people
who pushed it very hard.

**Tom Womack**, **Greg Childers**, **Bruce Dodson**, **Hallstein Hansen**, **Paul Leyland** and **Richard Wackerbarth** have continually thrown enormous size problems at the NFS postprocessing code of Msieve, and have been very patient as I've frantically tried to keep up with them. If you try to use NFS and it just works, even when other programs fail, you primarily have these guys to thank.

**Jayson King** has donated huge amounts of time into seriously improving NFS polynomial selection.

**Jeff Gilchrist** has done a lot of testing, feedback on 64-bit windows, and general documentation writing.

**Brian Gladman** contributed an expression evaluator, the Visual Studio build system, a lot of help with the numerical integration used in the NFS polynomial selector, and various portability fixes.

**Tom Cage** (RIP) found lots of bugs and ground through hundreds of factorizations with early versions of Msieve.

**Jay Berg** did a lot of experimenting with very big factorizations in earlier Msieve versions

The regulars in the Factoring forum of www.mersenneforum.org (especially **Jes**, **Luigi**, **Sam**, **Dennis**, **Sander**, **Mark R.**, **Peter S.**, **Jeff G.**) have also done tons of factoring work.

**Alex Kruppa** and **Bob Silverman** all contributed useful theoretical stuff. Bob's NFS siever code has been extremely helpful in getting me to understand how things work.

I thank my lucky stars that **Chris Card** figured out how NFS filtering works before I had to.

**Bob Silverman**, **Dario Alpern** and especially **Bill Hart** helped out with the NFS square root.

**forzles** helped improve the multithreading in the linear algebra.

**Falk Hueffner** and **Francois Glineur** found several nasty bugs in earlier versions.

**J6M** did the AIX port.

**Larry Soule** did a lot of work incorporating GMP into the code, which I regrettably don't expect to use.

I know I left out people, but that's my fault and not theirs.

\pagebreak
## Internet Factorization Resources

There are lots of places on the internet that go into the details of factorization algorithms, but a few are very handy if you have large problems to solve.

The best place to go for getting started with factorization is the [Factoring subforum][mersenneforum.factoring] of mersenneforum.org; there is a thriving community there of people working on factoring, including quite a few professionals.


### Links to factoring programs

A nice list of factoring software:

- [mersenneforum.org/showthread.php?t=3255][mersenneforum.programs]


### Links to Factoring Projects

A nice list of internet-coordinated projects that factor numbers of all types, primarily of mathematical interest:

- [mersenneforum.org/showthread.php?t=9611][mersenneforum.projects]


### Beginners Guide to NFS factoring using GGNFS and MSIEVE

Jeff's factorization program guide is required reading if you want to use Msieve for large-size problems:

- [gilchrist.ca/jeff/factoring/nfs_beginners_guide.html][jeff.guide]


### NFS@Home

NFS@Home is a distributed computing project run by Greg Childers (California State University Fullerton), that is tackling the largest public factoring projects. Powered by [BOINC][boinc.web].

- [escatter11.fullerton.edu/nfs][nfs@home.web]


### RSALS

RSALS is another distributed project that handles somewhat smaller numbers.
RSALS merged with the NFS@Home project.

- [boinc.unsads.com/rsals][rsals.web]

\pagebreak
## Quadratic Sieve References

### Prime Numbers: A Computational Perspective

The book "Prime Numbers: A Computational Perspective", by Richard Crandall and Carl Pomerance, is an excellent introduction to the quadratic sieve and many other topics in computational number theory.


### Factoring Large Integers with the Self-Initializing Quadratic Sieve

Scott Contini's thesis, "Factoring Large Integers with the Self-Initializing Quadratic Sieve", is an implementer's dream; it fills in all the details of the sieving process that Crandall and Pomerance gloss over.


### Block Sieving Algorithms

Wambach and Wettig's 1995 paper "Block Sieving Algorithms" gives an introduction to making sieving cache-friendly. Msieve uses very different (more efficient) algorithms, but you should try to understand these first.


### Factoring with Two Large Primes

Lenstra and Manasse's 1994 paper "Factoring with Two Large Primes" describes in great detail the cycle-finding algorithm that is the heart of the combining stage of Msieve. More background information on spanning trees and cycle- finding can be found in Manuel Huber's 2003 paper "Implementation of Algorithms for Sparse Cycle Bases of Graphs". This was the paper that connected the dots for me (pun intended).


### SQUFOF

There are three widely available descriptions of SQUFOF.

### Prime Numbers and Computer Methods for Factorization

An introductory one is Hans Riesel's section titled "Shanks' Factoring Method SQUFOF" in his book "Prime Numbers and Computer Methods for Factorization".

Daniel Shanks was a professor at the University of Maryland while I was a student there, and his work got me interested in number theory and computer programming. I dearly wish I met him before he died in 1996.


### Square Form Factorization

The much more advanced one is "Square Form Factorization", a PhD dissertation by Jason Gower (which is the reference I used when implementing the algorithm).

Henri Cohen's book (mentioned below) also has an extended treatment of SQUFOF.


### The Multiple Polynomial Quadratic Sieve: A Platform-Independent Distributed Application

Brandt Kurowski's 1998 paper "The Multiple Polynomial Quadratic Sieve: A Platform-Independent Distributed Application" is the only reference I could find that describes the Knuth-Schroeppel multiplier algorithm.


### Factorization Using the Quadratic Sieve Algorithm

Davis and Holdrige's 1983 paper "Factorization Using the Quadratic Sieve Algorithm" gives a surprising theoretical treatment of how QS works. Reading it felt like finding some kind of forgotten evolutionary offshoot, strangely different from the conventional way of implementing QS.


### A Block Lanczos Algorithm for Finding Dependencies over GF(2)

Peter Montgomery's paper "A Block Lanczos Algorithm for Finding Dependencies over GF(2)" revolutionized the factoring business. The paper by itself isn't enough to implement his algorithm; you really need someone else's implementation to fill in a few critical gaps.


### Parallel Block Lanczos for Solving Large Binary Systems

Michael Peterson's recent thesis "Parallel Block Lanczos for Solving Large Binary Systems" gives an interesting reformulation of the block Lanczos algorithm, and gives lots of performance tricks for making it run fast.


### Blocked Iterative Sparse Linear System Solvers for Finite Fields

Kaltofen's paper 'Blocked Iterative Sparse Linear System Solvers for Finite Fields' is a good condensing of Montgomery's original block Lanczos paper.


### Recent Advances in Direct Methods for Solving Unsymmetric Sparse Systems of Linear Equations

Gupta's IBM technical report 'Recent Advances in Direct Methods for Solving Unsymmetric Sparse Systems of Linear Equations' doesn't have anything in an NFS context, but there's gotta be some useful material in it for factoring people.


## Number Field Sieve References

### An Introduction to the Number Field Sieve

Matthew Briggs' 'An Introduction to the Number Field Sieve' is a very good introduction; it's heavier than C&P in places and lighter in others.


### A Beginner's Guide to the General Number Field Sieve

Michael Case's 'A Beginner's Guide to the General Number Field Sieve' has more detail all around and starts to deal with advanced stuff.


### Integer Factorization

Per Leslie Jensen's thesis 'Integer Factorization' has a lot of introductory detail on NFS that other references lack.


### The Number Field Sieve

Peter Stevenhagen's "The Number Field Sieve" is a whirlwind introduction the algorithm.


### The Number Field Sieve

Steven Byrnes' "The Number Field Sieve" is a good simplified introduction as well.


### The Number Field Sieve

Lenstra, Lenstra, Manasse and Pollard's paper 'The Number Field Sieve' is nice for historical interest.


### Factoring Estimates for a 1024-bit RSA Modulus

'Factoring Estimates for a 1024-bit RSA Modulus' should be required reading for anybody who thinks it would be a fun and easy project to break a commercial RSA key.


### On the Security of 1024-bit RSA and 160-bit Elliptic Curve Cryptography

'On the Security of 1024-bit RSA and 160-bit Elliptic Curve Cryptography' is a 2010-era update to the previous paper.


### Polynomial Selection for the Number Field Sieve Algorithm

Brian Murphy's thesis, 'Polynomial Selection for the Number Field Sieve Algorithm', is simply awesome. It goes into excruciating detail on a very undocumented subject.


### On Polynomial Selection for the General Number Field Sieve

Thorsten Kleinjung's 'On Polynomial Selection for the General Number Field Sieve' explains in detail a number of improvements to NFS polynomial selection developed since Murphy's thesis.


### NFS polynomial selection

Kleinjung's latest algorithmic ideas on NFS polynomial selection are documented at the 2008 CADO Factoring Workshop:

- [cado.gforge.inria.fr/workshop/abstracts.html][cado.abstracts.web]


### High-Performance Optimization of GNFS Polynomials

My talk 'High-Performance Optimization of GNFS Polynomials' gives some details of my own research on NFS polynomial selection:

- [event.cwi.nl/wcnt2011/program.html][wcnt2011.web]



### Rotations and Translations of Number Field Sieve Polynomials

Jason Gower's 'Rotations and Translations of Number Field Sieve Polynomials' describes some very promising improvements to the polynomial generation process. As far as I know, nobody has actually implemented them.


### Improvements to the polynomial selection process

D.J. Bernstein has two papers in press and several slides on some improvements to the polynomial selection process, that I'm just dying to implement.


### Sieving Using Bucket Sort

Aoki and Ueda's 'Sieving Using Bucket Sort' described the kind of memory optimizations that a modern siever must have in order to be fast.


### NFS with Four Large Primes: An Explosive Experiment

Dodson and Lenstra's 'NFS with Four Large Primes: An Explosive Experiment' is the first realization that maybe people should be using two large primes per side in NFS after all.


### Continued Fractions and Lattice Sieving

Franke and Kleinjung's 'Continued Fractions and Lattice Sieving' is the only modern reference available on techniques used in a high- performance lattice siever.


### Optimal Parametrization of SNFS

Bob Silverman's 'Optimal Parametrization of SNFS' has lots of detail on parameter selection and implementation details for building a line siever.


### On the amount of Sieving in Factorization Methods

Ekkelkamp's 'On the amount of Sieving in Factorization Methods'goes into a lot of detail on simulating NFS postprocessing.


### Strategies in Filtering in the Number Field Sieve

Cavallar's 'Strategies in Filtering in the Number Field Sieve'is really the only documentation on NFS postprocessing.


### A Self-Tuning Filtering Implementation for the Number Field Sieve

My talk 'A Self-Tuning Filtering Implementation for the Number Field Sieve' describes research that went into Msieve's filtering code.


### On the Reduction of Composed Relations from the Number Field Sieve

Denny and Muller's extended abstract 'On the Reduction of Composed Relations from the Number Field Sieve' is an early attempt at NFS filtering that's been almost forgotten by now, but their techniques can work on top of ordinary NFS filtering.


### Square Roots of Products of Algebraic Numbers

Montgomery's 'Square Roots of Products of Algebraic Numbers' describes the standard algorithm for the NFS square root phase.


### A Montgomery-Like Square Root for the Number Field Sieve

Nguyen's 'A Montgomery-Like Square Root for the Number Field Sieve'is also standard stuff for this subject; I haven't read this or the previous paper in great detail, but that's because the convetional NFS square root algorithm is still a complete mystery to me.


### Algebraic Algorithms Using P-adic Constructions

David Yun's 'Algebraic Algorithms Using P-adic Constructions' provided a lot of useful theoretical insight into the math underlying the simplex brute-force NFS square root algorithm that msieve uses.


--------


Decio Luiz Gazzoni Filho adds:

### The Development of the Number Field Sieve

The collection of papers 'The Development of the Number Field Sieve' (Springer Lecture Notes In Mathematics 1554) should be absolutely required reading -- unfortunately it's very hard to get ahold of. It's always marked 'special order' at Amazon.com, and I figured I shouldn't even try to order as they'd get back to me in a couple of weeks saying the book wasn't available. I was very lucky to find a copy available one day, which I promptly ordered. Again, I cannot recommend this book enough; I had read lots of literature on NFS but the first time I 'got' it was after reading the papers here. Modern expositions of NFS only show the algorithm as its currently implemented, and at times certain things are far from obvious. Now this book, being a historical account of NFS, shows how it progressed starting from John Pollard's initial work on SNFS, and things that looked out of place start to make sense. It's particularly enlightening to understand the initial formulation of SNFS, without the use of character columns.

*NOTE:* this has been reprinted and is available from [bn.com](https://www.barnesandnoble.com), at least `-JP`


### A Course In Computational Algebraic Number Theory

As usual, a very algebraic and deep exposition can be found in Henri Cohen's book 'A Course In Computational Algebraic Number Theory'. Certainly not for the faint of heart though. It's quite dated as well, e.g. the SNFS section is based on the 'old' (without character columns) SNFS, but explores a lot of the underlying algebra.


### The Theory of Algebraic Numbers

In order to comprehend NFS, lots of background on algebra and algebraic number theory is necessary.

I found a nice little introductory book on algebraic number theory, 'The Theory of Algebraic Numbers' by Harry Pollard and Harold Diamond. It's an old book, not contaminated by the excess of abstraction found on modern books. It helped me a lot to get a grasp on the algebraic concepts. Cohen's book is hard on the novice but surprisingly useful as one advances on the subject, and the algorithmic touches certainly help.


### Solving Sparse Linear Equations Over Finite Fields

As for papers: 'Solving Sparse Linear Equations Over Finite Fields'by Douglas Wiedemann presents an alternate method for the matrix step. Block Lanczos is probably better, but perhaps Wiedemann's method has some use, e.g. to develop an embarassingly parallel algorithm for linear algebra (which, in my opinion, is the current holy grail of NFS research).

[jasonp.site]: https://web.archive.org/web/20160315221418/http://boo.net/~jasonp/

[msieve.sf]: https://sourceforge.net/p/msieve/code/HEAD/tree/trunk/
[msieve.readme]: https://github.com/upiter/msieve/blob/master/README.md
[msieve.makefile]: https://github.com/upiter/msieve/blob/master/makefile
[msieve.header]: https://github.com/upiter/msieve/blob/master/include/msieve.h

[cado.nfs.web]: https://web.archive.org/web/20210504015721/http://cado-nfs.gforge.inria.fr/
[cado.abstracts.web]: https://web.archive.org/web/20110106163028/cado.gforge.inria.fr/workshop/abstracts.html

[ggnfs.sf]: https://sourceforge.net/p/ggnfs/code/HEAD/tree/trunk/
[factor-by-gnfs.sf]: https://sourceforge.net/p/factor-by-gnfs/factor-by-gnfs/ci/master/tree/gnfs/

[mersenneforum.factoring]: https://mersenneforum.org/forumdisplay.php?f=66
[mersenneforum.programs]: https://mersenneforum.org/showthread.php?t=3255
[mersenneforum.projects]: https://mersenneforum.org/showthread.php?t=9611

[jeff.guide]: https://gilchrist.great-site.net/jeff/factoring/nfs_beginners_guide.html
[rsals.web]: https://web.archive.org/web/20150212024501/http://boinc.unsads.com/rsals/
[nfs@home.web]: https://escatter11.fullerton.edu/nfs/
[boinc.web]: https://boinc.berkeley.edu
[wcnt2011.web]: https://event.cwi.nl/wcnt2011/program.html
