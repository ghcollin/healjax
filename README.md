Some healpix routines which run with JAX. 

These routines are ports of the BSD licensed healpix library https://github.com/astropy/astropy-healpix

Some notes:

Only contains routines for computing healpix bins from vec/RA-dec/theta-phi coordinates so far.

Tested to agree with astropy-healpix up to 1e-15 away from boundary in 64bit mode. Tests in 32bit mode predictably fail, as astropy-healpix computes in 64bit mode, meaning that we can't expect the binning results to be reproduced by 32bit calculations. 

Currently, there is an issue with running this on NVIDIA GPUs. Tests pass on ARM and x86 CPUs.