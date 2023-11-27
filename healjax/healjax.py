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

import jax
import jax.numpy as jnp
from functools import partial

def atan2_2pi(y, x):
    val = jnp.arctan2(y, x)
    #return val + (val < 0)*2*jnp.pi
    return val + jnp.where(val < 0, 2*jnp.pi, 0.0)

def xyz_to_hp_polar(out_dtype, nside, vx, vy, vz_raw):
    
    phi = atan2_2pi(vy, vx)
    phi_t = jnp.fmod(phi, jnp.pi/2)
    #jax.debug.print("hj vx={}, vy={}, vz={}, phi={}, phi_t={}", vx, vy, vz_raw, phi, phi_t)

    north, vz = (vz_raw > 2.0/3.0), jnp.abs(vz_raw)

    coz = jnp.hypot(vx, vy)

    kx = (coz/jnp.sqrt(1.0 + vz)) * jnp.sqrt(3) * jnp.fabs(nside * (2.0*phi_t - jnp.pi)/jnp.pi)

    ky = (coz/jnp.sqrt(1.0 + vz)) * jnp.sqrt(3) * nside * 2.0 * phi_t / jnp.pi

    xx, yy = jax.lax.cond(north, lambda: (nside - kx, nside - ky), lambda: (ky, kx))

    x = jnp.minimum(nside - 1, jnp.floor(xx).astype(out_dtype))
    y = jnp.minimum(nside - 1, jnp.floor(yy).astype(out_dtype))

    #dx = xx - x
    #dy = yy - y

    sector = (phi - phi_t)*2/jnp.pi
    offset_raw = jnp.round(sector).astype(out_dtype)
    offset = jnp.fmod(jnp.fmod(offset_raw, 4) + 4, 4) # c '%' operators has same sematics as fmod. fmod should return int (despite name)
    column = offset
    return jnp.where(north, 0, 8) + column, x, y

def xyz_to_hp_equator(out_dtype, nside, vx, vy, vz):

    phi = atan2_2pi(vy, vx)
    phi_t = jnp.fmod(phi, jnp.pi/2.0)

    # project into the unit square z=[-2/3, 2/3], phi=[0, pi/2]
    #zunits = (vz + 2.0/3.0) * 3.0/4.0
    #zunits = (vz * 3.0/4.0 + 0.5)
    zunits = (vz + (2.0 / 3.0)) / (4.0 / 3.0)
    phiunits = phi_t * 2/jnp.pi
    # convert into diagonal units
    # (add 1 to u2 so that they both cover the range [0,2])
    u1 = zunits + phiunits
    u2 = zunits - phiunits + 1.0
    # 1.0 - phiunits = 1 - phi_t * 2/jnp.pi = (pi/2 - phi_t) 2/pi = (pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2)) 2/pi
    #       = (pi/2 - jnp.fmod(atan2(vy, vx), jnp.pi/2)) 2/pi = (jnp.fmod(pi/2 - atan2(vy, vx), jnp.pi/2)) 2/pi
    #       = (jnp.fmod(atan2(vx, vy), jnp.pi/2)) 2/pi
    #u2 = zunits + jnp.fmod(jnp.arctan2(vx, vy), jnp.pi/2) * 2/jnp.pi
    # x is the northeast direction, y is the northwest.
    xx_full = u1 * nside
    yy_full = u2 * nside

    # 1.0 - phiunits = 1 - phi_t * 2/jnp.pi = (pi/2 - phi_t) 2/pi = (pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2)) 2/pi = (pi/2 - atan(abs(vy/vx))) 2/pi = (acot(abs(vy/vx))) 2/pi
    # yy_full = nside ((3 vz/4 + 1.5) pi/2 - jnp.fmod(atan2_2pi(vy, vx), jnp.pi/2) ) 2/pi

    # now compute which big healpix it's in.
    # (note that we subtract off the modded portion used to
    # compute the position within the healpix, so this should be
    # very close to one of the boundaries.)
    sector = (phi - phi_t)*2/jnp.pi
    offset_raw = jnp.round(sector).astype(out_dtype)
    offset = jnp.fmod(jnp.fmod(offset_raw, 4) + 4, 4) # c '%' operators has same sematics as fmod. fmod should return int (despite name)
    
    # we're looking at a square in z,phi space with an X dividing it.
    # we want to know which section we're in.
    # xx ranges from 0 in the bottom-left to 2Nside in the top-right.
    # yy ranges from 0 in the bottom-right to 2Nside in the top-left.
    # (of the phi,z unit box)

    upper_xx = xx_full >= nside
    upper_yy = yy_full >= nside
    xx, yy = jnp.where(upper_xx, xx_full - nside, xx_full), jnp.where(upper_yy, yy_full - nside, yy_full)
    basehp = jnp.array([
        [ 8 + offset                    , offset + 4    ],
        [ jnp.fmod(offset + 1, 4) + 4   , offset        ]
    ])[upper_xx.astype(int)][upper_yy.astype(int)]

    x = jnp.maximum(0, jnp.minimum(nside-1, jnp.floor(xx).astype(out_dtype)))
    #dx = xx - x 
    y = jnp.maximum(0, jnp.minimum(nside-1, jnp.floor(yy).astype(out_dtype)))
    #dy = yy - y
    #jax.debug.print("{} {} {} {} {} {} {} {} {} {} {}", vx, vy, vz, phi, phi_t, zunits, phiunits, u2, basehp, x, y)

    return basehp, x, y

def xyz_to_hp(nside, vx, vy, vz, out_dtype=None):
    out_dtype = int if out_dtype is None else out_dtype
    return jax.lax.cond(jnp.abs(vz) >= 2.0/3.0, 
                        partial(xyz_to_hp_polar, out_dtype), 
                        partial(xyz_to_hp_equator, out_dtype), 
                        nside, vx, vy, vz)

def radec2x(r, d):
    return jnp.cos(d)*jnp.cos(r)

def radec2y(r, d):
    return jnp.cos(d)*jnp.sin(r)

def radec2z(r, d):
    return jnp.sin(d)

def radec2xyz(r, d):
    return radec2x(r, d), radec2y(r, d), radec2z(r, d)

#def radec_to_healpix(nside, ra, dec):
#    return healpixl_compose_xy(nside, *xyz_to_hp(nside, *radec2xyz(ra, dec)))

def healpixl_xy_to_composed_xy(nside, hpxy, x, y):
    return ((nside * hpxy) + x) * nside + y

def healpixl_xy_to_nested(nside, hpxy, x, y):
    # We construct the index called p_n' in the healpix paper, whose bits
    # are taken from the bits of x and y:
    #    x = ... b4 b2 b0
    #    y = ... b5 b3 b1
    # We go through the bits of x,y, building up "index":

    def loop_body(i, carry):
        index, xc, yc = carry
        new_index = jnp.bitwise_or(index, jnp.left_shift(jnp.bitwise_or(jnp.left_shift(jnp.bitwise_and(yc, 1), 1), jnp.bitwise_and(xc, 1)), 2*i))
        new_y = jnp.right_shift(yc, 1)
        new_x = jnp.right_shift(xc, 1)
        #update_index = jax.lax.cond(jnp.logical_or(new_x != 0, new_y != 0), )
        return (new_index, new_x, new_y)
    
    final_index, _, _ = jax.lax.fori_loop(0, 8*x.dtype.itemsize//2, loop_body, (0, x, y))

    return final_index + hpxy.astype(int) * (nside * nside)

def healpixl_xy_to_ring(nside, hpxy, x, y):
    frow = hpxy // 4
    F1 = frow + 2
    v = x + y

    # "ring" starts from 1 at the north pole and goes to 4Nside-1 at
    # the south pole; the pixels in each ring have the same latitude.
    ring = F1*nside - v - 1

    def north_pole(ring):
        # north polar.
        # left-to-right coordinate within this healpix
        index0 = (nside - 1 - y)
        # offset from the other big healpixes
        index1 = index0 + jnp.fmod(hpxy, 4) * ring
        # offset from the other rings
        index2 = index1+  ring*(ring-1)*2
        return index2
    
    def south_pole(ring):
        # south polar.
        # Here I first flip everything so that we label the pixels
        # at zero starting in the southeast corner, increasing to the
        # west and north, then subtract that from the total number of
        # healpixels.
        ri = 4*nside - ring;
        # index within this healpix
        index0 = (ri-1) - x
        # big healpixes
        index1 = index0 + ((3-jnp.fmod(hpxy, 4)) * ri)
        # other rings
        index2 = index1 + ri*(ri-1)*2
        # flip!
        index3 = 12*nside*nside - 1 - index2
        return index3
    
    def equatorial(ring):
        # equatorial.
        s = jnp.fmod((ring - nside), 2)
        F2 = 2 * jnp.fmod(hpxy, 4) - jnp.fmod(frow, 2) + 1
        h = x - y

        index0 = ((F2 * nside + h + s) / 2).astype(h.dtype)
        # offset from the north polar region:
        index1 = index0 + nside * (nside - 1) * 2
        # offset within the equatorial region:
        index2 = index1 + nside * 4 * (ring - nside)
        # handle healpix #4 wrap-around
        index3 = index2 + jnp.where(jnp.logical_and(hpxy == 4, y > x), (4 * nside - 1), 0)
        #jax.debug.print("hj {} {} {} frow={}, F1={}, v={}, ringind={}, s={}, F2={}, h={}, longind={}.", hpxy, x, y, frow, F1, v, ring, s, F2, h, index0)
        return index3
    
    return jnp.piecewise(ring, [ring <= nside, ring >= 3*nside], [north_pole, south_pole, equatorial])

def ang2pix_vec(scheme, nside, x, y, z, out_dtype=None):
    scheme_funcs = {
        'xy': healpixl_xy_to_composed_xy,
        'nest': healpixl_xy_to_nested,
        'ring': healpixl_xy_to_ring
    }

    return scheme_funcs[scheme](nside, *xyz_to_hp(nside, x, y, z, out_dtype=out_dtype))

def ang2pix_radec(scheme, nside, ra, dec, out_dtype=None):
    return ang2pix_vec(scheme, nside, *radec2xyz(ra, dec), out_dtype=out_dtype)

def ang2pix(scheme, nside, theta, phi, out_dtype=None):
    ra = phi
    dec = jnp.pi/2 - theta
    return ang2pix_radec(scheme, nside, ra, dec, out_dtype=out_dtype)