# Full, restarted and weighted GMRES using the MGS and Householder orthogonalization method

***
Matlab programs written by Nora Heinrich, Matrikelnr. 331577
References: Matlab gmres implementation, Algorithm 11.4.2 in Golub and van Loans „Matrix computations“ and Walkers Householder Arnoldi method    
***

The main routine 
	[resvec_m, resvec_house] = gmresM(A, m, b, weight, tol, maxit)
computes a solution of a linear system Ax=b using both MGS and Householder orthogonalization. The variable m is the restart parameter, tol the tolerance and maxit the maximum number of cycles (outer iterations).

For full GMRES, use m=0.
For restarted GMRES, specify the restart parameter and weight = 'e'.
For weighted GMRES following Essai, specify the restart parameter and weight = 'w1'. (Further weighting strategies following Embree et al. and Najafi and Zareamoghaddam are integrated.)

There is four subroutines, depending on the chosen method. These are 
	MGS - for restarted MGS GMRES
	WMGS - for weighted MGS GMRES
	House - for restarted Householder GMRES
	WHouse - for weighted Householder GMRES.

The default output is the relative residual development of both MGS and Householder GMRES.