# Healjax

Some healpix routines which run with JAX. 

These routines are ports of the BSD licensed healpix library https://github.com/astropy/astropy-healpix

# Installation

```
pip install git+https://github.com/ghcollin/healjax.git
```

You will need to have installed JAX yourself.

# Some notes

All functions take a `scheme` argument that can be one of
 - `'ring'` The ring indexing healpix convention.
 - `'nest'` The nested indexing healpix convetion.
 - `'xy'` The xy convention used in the astropy healpix library. All calculations are done in this scheme, the other schemes are provided by transforming to and from xy.

All functions also take an `nside` argument that should be an integer. In typical JAX style, these functions are only defined to operate on single bin indices/vectors/angle-pairs, use `jax.vmap` to operate on arrays.

The following functions are used to convert to healpix bin indices. They all take an `out_dtype` option that defaults to `int`. This option can be used if you wish to perform calculations in a smaller dtype than the default JAX integer type.

 - `vec2pix(scheme, nside, x, y, z, out_dtype=None)` Takes an xyz vector and returns the healpix bin that it lies in.
 - `ang2pix_radec(scheme, nside, ra, dec, out_dtype=None)` Takes a right ascension and declination and returns a healpix bin.
 - `ang2pix(scheme, nside, theta, phi, out_dtype=None)` Takes a theta and phi pair of angles in the healpy convention and returns a healpix bin.

The following functions are used to convert from healpix bin indices. They all take a `dx` and `dy` option that is used to offset the output within the requested bin. The default for both is 0.5, which returns a vector/angle-pair that lies in the center of the bin.

 - `pix2vec(scheme, nside, hp, dx=None, dy=None)` Takes a healpix index and returns the xyz vector that it corresponds to.
 - `pix2ang_radec(scheme, nside, hp, dx=None, dy=None)` Takes a healpix index and returns the right ascension and declination that it corresponds to.
 - `pix2ang_colatlong(scheme, nside, hp, dx=None, dy=None)` Takes a healpix index and returns theta and phi in the healpy colatitude longitude convention.
 - `pix2ang(scheme, nside, hp, dx=None, dy=None)` Same as `pix2ang_colonglat` but the conversion is done through xyz vectors as in the astropy healpix library.

There are in addition two functions for finding neighbouring healpix bins.
 - `get_neighbours(scheme, nside, hp)` This is intended to have the same API as the healpy neighbour function (as tested against the astropy healpix library). It returns 8 neighbours, with -1 denoting a non-existent neighbour.
 - `get_patch(scheme, nside, hp)` This is used to implement `get_neighbours` and returns a 3x3 array of the neighbours and the input index (located in the central element). A -1 denotes that the neighbour in that position does not exist.

The following functions can be used to convert between coordinate systems:
 - `ang2vec_radec(ra, dec)` converts from right ascension and declination to the x, y, z vector system.
 - `ang2vec(theta, phi)` converts from the healpy colatitude and longitude system to the x, y, z vector system.
 - `vec2ang_radec(x, y, z)` converts from the x, y, z vector system to right ascension and declination.
 - `vec2ang(x, y, z)` converts from the x, y, z vector system to the healpy colatitude longitude system, returning theta, phi.

Finally, the `convert_map(nside, in_scheme, out_scheme, map)` function can be used to convert a map to and from the various schemes detailed above.

# Compatibility

Angle-pair to healpix functions tested to agree with astropy-healpix up to 1e-15 away from boundary in 64bit mode. Tests in 32bit mode predictably fail, as astropy-healpix computes in 64bit mode, meaning that we can't expect the binning results to be reproduced by 32bit calculations. 

Healpix to angle-pair functions current exhibit a relative error that grows with the nside. For nsides up to 1024, this stays within a factor of 1000 times the floating point epsilon. This may be due to an implementation error that I haven't been able to track down.

The neighbours and convert map functions have been tested to agree with astropy-healpix.

Currently, there is an issue with running this on NVIDIA GPUs. Vmapping over certain, medium-sized arrays of inputs can cause JAX to crash. If you run into this, you can try increasing or decreasing the size of the array you vmap over to work around it. This bug has been reported to the JAX team.
Tests pass on ARM and x86 CPUs.