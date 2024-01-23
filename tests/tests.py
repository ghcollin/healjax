"""
Copyright (c) 2023 ghcollin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import unittest
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healjax
import astropy_healpix
import astropy_healpix.healpy
import healpy
import numpy
from functools import partial

n_test = 1000000
test_nsides = [  2**(i+1) for i in range(10) ]
edge_eps = 1e-15

class TestIndexingMethods(unittest.TestCase):

    def test_north_pole(self):
        phis = numpy.array(jnp.linspace(edge_eps, 2*jnp.pi-edge_eps, 10))
        norths = numpy.array(jnp.full_like(phis, 0) + edge_eps)

        for nside in test_nsides:
            true_north_ring = astropy_healpix.healpy.ang2pix(nside, norths, phis, nest=False)
            test_north_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(norths, phis)
            self.assertTrue((true_north_ring == test_north_ring).all(), (true_north_ring, test_north_ring))

        for nside in test_nsides:
            true_north_nest = astropy_healpix.healpy.ang2pix(nside, norths, phis, nest=True)
            test_north_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(norths, phis)
            self.assertTrue((true_north_nest == test_north_nest).all(), (true_north_nest, test_north_nest))

    def test_south_pole(self):
        phis = numpy.array(jnp.linspace(edge_eps, 2*jnp.pi-edge_eps, 10))
        souths = numpy.array(jnp.full_like(phis, jnp.pi) - edge_eps)

        for nside in test_nsides:
            true_south_ring = astropy_healpix.healpy.ang2pix(nside, souths, phis, nest=False)
            test_south_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(souths, phis)
            self.assertTrue((true_south_ring == test_south_ring).all(), (true_south_ring, test_south_ring))

        for nside in test_nsides:
            true_south_nest = astropy_healpix.healpy.ang2pix(nside, souths, phis, nest=True)
            test_south_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(souths, phis)
            self.assertTrue((true_south_nest == test_south_nest).all(), (true_south_nest, test_south_nest))

    def test_seam(self):
        thetas_half = jnp.linspace(0+edge_eps, jnp.pi-edge_eps, 10)
        phis = numpy.array(jnp.concatenate([jnp.full_like(thetas_half, edge_eps), jnp.full_like(thetas_half, 2*jnp.pi - edge_eps)]))
        thetas = numpy.array(jnp.concatenate([thetas_half, thetas_half]))

        for nside in test_nsides:
            true_ring = astropy_healpix.healpy.ang2pix(nside, thetas, phis, nest=False)
            test_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(thetas, phis)
            self.assertTrue((true_ring == test_ring).all(), (true_ring, test_ring))

        for nside in test_nsides:
            true_nest = astropy_healpix.healpy.ang2pix(nside, thetas, phis, nest=True)
            test_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(thetas, phis)
            self.assertTrue((true_nest == test_nest).all(), (true_nest, test_nest))

    def test_bulk(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        test_phis = numpy.array(jax.random.uniform(key1, minval=edge_eps, maxval=2*jnp.pi-edge_eps, shape=(n_test,)))
        test_thetas = numpy.array(jax.random.uniform(key2, minval=edge_eps, maxval=jnp.pi-edge_eps, shape=(n_test,)))

        for nside in test_nsides:
            true_hp_idxs_ring = astropy_healpix.healpy.ang2pix(nside, test_thetas, test_phis, nest=False)
            test_hp_idxs_ring = jax.jit(jax.vmap(partial(healjax.ang2pix, 'ring', nside)))(test_thetas, test_phis)
            self.assertTrue((true_hp_idxs_ring == test_hp_idxs_ring).all(), (true_hp_idxs_ring, test_hp_idxs_ring))
        
        for nside in test_nsides:
            true_hp_idxs_nest = astropy_healpix.healpy.ang2pix(nside, test_thetas, test_phis, nest=True)
            test_hp_idxs_nest = jax.jit(jax.vmap(partial(healjax.ang2pix, 'nest', nside)))(test_thetas, test_phis)
            self.assertTrue((true_hp_idxs_nest == test_hp_idxs_nest).all(), (true_hp_idxs_nest, test_hp_idxs_nest))

    def test_indices(self):
        error_tol = 1000
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_ring = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=False))
            test_angs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang, 'ring', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_ring - test_angs_ring) <= error_tol*jnp.finfo(test_angs_ring.dtype).eps * jnp.abs(true_angs_ring)).all(), (nside, true_angs_ring, test_angs_ring, numpy.max(jnp.abs((true_angs_ring - test_angs_ring)))))

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_nest = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=True))
            test_angs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang, 'nest', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_nest - test_angs_nest) <= error_tol*jnp.finfo(test_angs_nest.dtype).eps * jnp.abs(true_angs_nest)).all(), (nside, true_angs_nest, test_angs_nest, numpy.max(jnp.abs((true_angs_nest - test_angs_nest)))))

    def test_indices_sharp(self):
        error_tol = 1000
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_ring = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=False))
            test_angs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang_colatlong, 'ring', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_ring - test_angs_ring) <= error_tol*jnp.finfo(test_angs_ring.dtype).eps * jnp.abs(true_angs_ring)).all(), (nside, true_angs_ring, test_angs_ring, numpy.max(jnp.abs((true_angs_ring - test_angs_ring)))))

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_angs_nest = numpy.array(astropy_healpix.healpy.pix2ang(nside, hp_idxs, nest=True))
            test_angs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.pix2ang_colatlong, 'nest', nside)))(hp_idxs))
            self.assertTrue((jnp.abs(true_angs_nest - test_angs_nest) <= error_tol*jnp.finfo(test_angs_nest.dtype).eps * jnp.abs(true_angs_nest)).all(), (nside, true_angs_nest, test_angs_nest, numpy.max(jnp.abs((true_angs_nest - test_angs_nest)))))

    def test_neighbours(self):
        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_neighs_ring = astropy_healpix.neighbours(hp_idxs, nside, order='ring').T
            test_neighs_ring = numpy.array(jax.jit(jax.vmap(partial(healjax.get_neighbours, 'ring', nside)))(hp_idxs))
            self.assertTrue((true_neighs_ring  == test_neighs_ring ).all(), (true_neighs_ring , test_neighs_ring, true_neighs_ring - test_neighs_ring ))

        for nside in test_nsides:
            hp_idxs = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_neighs_nest = astropy_healpix.neighbours(hp_idxs, nside, order='nested').T
            test_neighs_nest = numpy.array(jax.jit(jax.vmap(partial(healjax.get_neighbours, 'nest', nside)))(hp_idxs))
            self.assertTrue((true_neighs_nest  == test_neighs_nest ).all(), (true_neighs_nest , test_neighs_nest, true_neighs_nest - test_neighs_nest ))

    def test_convert(self):
        for nside in test_nsides:
            test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_convert_to_nest = healpy.reorder(test_map, r2n=True)
            test_convert_to_nest = jax.jit(partial(healjax.convert_map, nside, 'ring', 'nest'))(test_map)
            self.assertTrue((true_convert_to_nest == test_convert_to_nest).all(), (true_convert_to_nest, test_convert_to_nest))

        for nside in test_nsides:
            test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
            true_convert_to_ring = healpy.reorder(test_map, n2r=True)
            test_convert_to_ring = jax.jit(partial(healjax.convert_map, nside, 'nest', 'ring'))(test_map)
            self.assertTrue((true_convert_to_ring == test_convert_to_ring).all(), (true_convert_to_ring, test_convert_to_ring))

    def test_convert_identity(self):
        for scheme in ['ring', 'nest', 'xy']:
            for nside in test_nsides:
                test_map = numpy.arange(astropy_healpix.nside_to_npix(nside))
                id_map = jax.jit(partial(healjax.convert_map, nside, scheme, scheme))(test_map)
                self.assertTrue((test_map == id_map).all(), (scheme, test_map, id_map))

    def test_convert_from_xy(self):
        error_tol = 1e7
        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True, nest=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs
            test_map = jax.jit(partial(healjax.convert_map, nside, 'xy', 'nest'))(test_map_xy)

            self.assertTrue((numpy.abs(truth_map - test_map) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)).all(), (truth_map, test_map, test_map_xy, numpy.max(numpy.abs(truth_map - test_map)/(numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)))))

        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs
            test_map = jax.jit(partial(healjax.convert_map, nside, 'xy', 'ring'))(test_map_xy)

            self.assertTrue((numpy.abs(truth_map - test_map) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)).all(), (truth_map, test_map, test_map_xy, numpy.max(numpy.abs(truth_map - test_map)/(numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map)))))

    def test_convert_to_xy(self):
        error_tol = 1e7
        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True, nest=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)
            truth_map_xy = jax.jit(partial(healjax.convert_map, nside, 'nest', 'xy'))(truth_map)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs

            self.assertTrue((numpy.abs(truth_map_xy - test_map_xy) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map_xy)).all(), (truth_map_xy, test_map_xy, truth_map))

        for nside in test_nsides:
            truth_ras, truth_decs = astropy_healpix.healpy.pix2ang(nside, numpy.arange(astropy_healpix.nside_to_npix(nside)), lonlat=True)
            truth_map = (truth_ras + 2*numpy.pi*truth_decs)*(numpy.pi/180)
            truth_map_xy = jax.jit(partial(healjax.convert_map, nside, 'ring', 'xy'))(truth_map)

            test_ras, test_decs = jax.jit(jax.vmap(partial(healjax.pix2ang_radec, 'xy', nside)))(numpy.arange(astropy_healpix.nside_to_npix(nside)))
            test_map_xy = test_ras + 2*jnp.pi*test_decs

            self.assertTrue((numpy.abs(truth_map_xy - test_map_xy) <= error_tol * numpy.finfo(truth_map.dtype).eps * numpy.abs(truth_map_xy)).all(), (truth_map_xy, test_map_xy, truth_map))


if __name__ == '__main__':
    unittest.main()