import unittest
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healjax
import astropy_healpix
import astropy_healpix.healpy
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


if __name__ == '__main__':
    unittest.main()