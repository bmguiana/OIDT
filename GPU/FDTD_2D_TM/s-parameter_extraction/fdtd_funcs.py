"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, as part of
                 the Applied and Computational Electromagnetics Signal and
                 Power Integrity (ACEM-SPI) Lab while working toward the Ph.D.
                 in Electrical Engineering at the University of Idaho, Moscow,
                 Idaho, USA. This project was funded, in part, by the National
                 Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
from numba import cuda, njit, prange, f4, f8, i4, void
from aux_funcs import mkdir, check_discretization, dict_from_module
import sys
import pickle

args = sys.argv
try:
    uf = open('user_dict_{}.pkl'.format(args[1]), 'rb')
except:
    print('loading default configuration')
    import user_config as user_module
    user_dict = dict_from_module(user_module)
    dict_file = open('user_dict.pkl', 'wb')
    pickle.dump(user_dict, dict_file)
    dict_file.close()
    uf = open('user_dict.pkl', 'rb')

usr = pickle.load(uf)

if usr['precision'] == np.float32:
    fa = f4
elif usr['precision'] == np.float64:
    fa = f8


#                          Ez     , Hx     , Hy     , kxe  , kye  , cb     , ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa[:,:], i4, i4, i4, i4))
def update_ez(Ez, Hx, Hy, kxe, kye, cb, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j >= jb and j < je:
            Ez[i, j] = Ez[i, j] + cb[i, j] * ( ( Hy[i+1, j] - Hy[i, j] ) / kxe[i] - ( Hx[i, j+1] - Hx[i, j] ) / kye[j] )


#                          Ez     , Hx     , kyh  , db, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:], fa, i4, i4, i4, i4))
def update_hx(Ez, Hx, kyh, db, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j > jb and j < je:
            Hx[i, j] = Hx[i, j] + db * ( Ez[i, j-1] - Ez[i, j] ) / kyh[j]


#                          Ez     , Hy     , kxh  , db, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:], fa, i4, i4, i4, i4))
def update_hy(Ez, Hy, kxh, db, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j >= jb and j < je:
            Hy[i, j] = Hy[i, j] + db * ( Ez[i, j] - Ez[i-1, j] ) / kxh[i]


#                          Field  , Js, xb, yb, xe, ye
@cuda.jit(func_or_sig=void(fa[:,:], fa, i4, i4, i4, i4))
def update_source(Field, Source, sxb, syb, sxe, sye):
    i, j = cuda.grid(2)
    if i >= sxb and i < sxe:
        if j >= syb and j < sye:
            Field[i, j] = Field[i, j] - Source


#                          PEz_xl , PEz_xh , Ez     , Hy     , be   , ce   , cb     , dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa[:,:], fa, i4, i4, i4, i4, i4))
def update_pml_ez_xinc(Psi_Ez_xlo, Psi_Ez_xhi, Ez, Hy, be, ce, cb, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i < pmle:
        if j >= jb and j < je:
            Psi_Ez_xlo[i, j] = be[-(i+1)] * Psi_Ez_xlo[i, j] + ce[-(i+1)] * ( Hy[ib+i+1, j] - Hy[ib+i, j] ) / dx
            Ez[ib+i, j] = Ez[ib+i, j] + cb[ib+i, j] * Psi_Ez_xlo[i, j]
            Psi_Ez_xhi[i, j] = be[i] * Psi_Ez_xhi[i, j] + ce[i] * ( Hy[ie-pmle+i+1, j] - Hy[ie-pmle+i, j] ) / dx
            Ez[ie-pmle+i, j] = Ez[ie-pmle+i, j] + cb[ie-pmle+i, j] * Psi_Ez_xhi[i, j]


#                          PEz_yl , PEz_yh , Ez     , Hx     , be   , ce   , cb     , dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa[:,:], fa, i4, i4, i4, i4, i4))
def update_pml_ez_yinc(Psi_Ez_ylo, Psi_Ez_yhi, Ez, Hx, be, ce, cb, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j < pmle:
            Psi_Ez_ylo[i, j] = be[-(j+1)] * Psi_Ez_ylo[i, j] + ce[-(j+1)] * ( Hx[i, jb+j+1] - Hx[i, jb+j] ) / dy
            Ez[i, jb+j] = Ez[i, jb+j] - cb[i, jb+j] * Psi_Ez_ylo[i, j]
            Psi_Ez_yhi[i, j] = be[j] * Psi_Ez_yhi[i, j] + ce[j] * ( Hx[i, je-pmle+j+1] - Hx[i, je-pmle+j] ) / dy
            Ez[i, je-pmle+j] = Ez[i, je-pmle+j] - cb[i, je-pmle+j] * Psi_Ez_yhi[i, j]


#                          PHx_yl , PHx_yh , Ez     , Hx     , bh   , ch   , db, dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa, fa, i4, i4, i4, i4, i4))
def update_pml_hx_yinc(Psi_Hx_ylo, Psi_Hx_yhi, Ez, Hx, bh, ch, db, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j > 0 and j < pmle:
            Psi_Hx_ylo[i, j] = bh[-j] * Psi_Hx_ylo[i, j] + ch[-j] * ( Ez[i, jb+j] - Ez[i, jb+j-1] ) / dy
            Hx[i, jb+j] = Hx[i, jb+j] - db * Psi_Hx_ylo[i, j]
            Psi_Hx_yhi[i, j] = bh[j] * Psi_Hx_yhi[i, j] + ch[j] * ( Ez[i, je-pmle+j] - Ez[i, je-pmle+j-1] ) / dy
            Hx[i, je-pmle+j] = Hx[i, je-pmle+j] - db * Psi_Hx_yhi[i, j]


#                          PHy_xl , PHy_xh , Ez     , Hy     , bh   , ch   , db, dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa, fa, i4, i4, i4, i4, i4))
def update_pml_hy_xinc(Psi_Hy_xlo, Psi_Hy_xhi, Ez, Hy, bh, ch, db, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > 0 and i < pmle:
        if j >= jb and j < je:
            Psi_Hy_xlo[i, j] = bh[-i] * Psi_Hy_xlo[i, j] + ch[-i] * ( Ez[ib+i, j] - Ez[ib+i-1, j] ) / dx
            Hy[ib+i, j] = Hy[ib+i, j] + db * Psi_Hy_xlo[i, j]
            Psi_Hy_xhi[i, j] = bh[i] * Psi_Hy_xhi[i, j] + ch[i] * ( Ez[ie-pmle+i, j] - Ez[ie-pmle+i-1, j] ) / dx
            Hy[ie-pmle+i, j] = Hy[ie-pmle+i, j] + db * Psi_Hy_xhi[i, j]


#                          Rz     , Iz     , Ez     , cs, sn, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa, fa, i4, i4, i4, i4))
def simul_fft_efield(Rz, Iz, Ez, cos, sin, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rz[i, j] = Rz[i, j] + 0.5 * (Ez[i, j] + Ez[i, j]) * cos
            Iz[i, j] = Iz[i, j] - 0.5 * (Ez[i, j] + Ez[i, j]) * sin


#                          Rx     , Ry     , Ix     , Iy     , Hx     , Hy     , cs, sn, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa, fa, i4, i4, i4, i4))
def simul_fft_hfield(Rx, Ry, Ix, Iy, Hx, Hy, cos, sin, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rx[i, j] = Rx[i, j] + 0.5 * (Hx[i, j] + Hx[i, j+1]) * cos
            Ix[i, j] = Ix[i, j] - 0.5 * (Hx[i, j] + Hx[i, j+1]) * sin
            Ry[i, j] = Ry[i, j] + 0.5 * (Hy[i, j] + Hy[i+1, j]) * cos
            Iy[i, j] = Iy[i, j] - 0.5 * (Hy[i, j] + Hy[i+1, j]) * sin


def make_profile(length_eff, rough_std, rough_acl, dx, stol, atol, mtol):
    from pyspeckle import create_exp_1D
    i = 0
    while(True):
        if i > 300000:
            raise Exception('No valid profile for current settings after 300,000 profile iterations')
        test_array = create_exp_1D(length_eff, 0.0, rough_std, rough_acl)
        sigma, acl, mm = check_discretization(test_array, dx)
        if i % 1000 == 0:
            print('checked {:} profile iterations'.format(i))
        if (1-stol)*rough_std*dx < sigma < (1+stol)*rough_std*dx and (1-atol)*rough_acl*dx < acl < (1+atol)*rough_acl*dx and -mtol < mm < mtol:
            print('Valid profile found after {:} iterations'.format(i))
            print('Parameters for this profile are: s={:.2f} nm, Lc={:.2f} nm\n\n'.format(sigma*1e9, acl*1e9))
            break
        i += 1
    return test_array


def generate_fg_mask(num_x, num_y, bord_x, bord_y, wg_height, pitch, num_lines, settling_range, end_range, rough_std, rough_acl, dx, mode='gen', out_dir='rough_profiles', correlation=3, upper_name='', lower_name='', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y], dtype=bool)
    num_x_eff = num_x - settling_range - end_range
    if mode == 'gen':
        mkdir(out_dir)
        print('Searching for upper profile')
        var_wrt_len_upper = make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        if correlation == 1:
            var_wrt_len_lower = var_wrt_len_upper.copy()
        elif correlation == 2:
            var_wrt_len_lower = -1*var_wrt_len_upper
        elif correlation == 3:
            print('Searching for lower profile')
            var_wrt_len_lower = make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        else:
            raise Exception('Invalid correlation type selected')
    elif mode =='load':
        var_wrt_len_upper = np.load(upper_name)
        var_wrt_len_lower = np.load(lower_name)
    elif mode == 'smooth':
        var_wrt_len_upper = np.zeros(num_x_eff)
        var_wrt_len_lower = np.zeros(num_x_eff)
    else:
        raise Exception('Choose a valid roughness profile mode')

    mkdir(out_dir)
    np.save(upper_name, var_wrt_len_upper)
    np.save(lower_name, var_wrt_len_lower)

    var_wrt_len_upper = np.append(np.zeros(settling_range), var_wrt_len_upper)
    var_wrt_len_upper = np.append(var_wrt_len_upper, np.zeros(end_range))
    var_wrt_len_lower = np.append(np.zeros(settling_range), var_wrt_len_lower)
    var_wrt_len_lower = np.append(var_wrt_len_lower, np.zeros(end_range))

    apply_mask(fg_mask, num_x, num_y, bord_x, bord_y, wg_height, pitch, num_lines, var_wrt_len_lower, var_wrt_len_upper)
    return fg_mask


@njit(parallel=True)
def apply_mask(fg_mask, num_x, num_y, bord_x, bord_y, wg_height, pitch, num_lines, var_wrt_len_lower, var_wrt_len_upper):
    for l in range(num_lines):
        offset = l * pitch
        for j in prange(num_y):
            for i in prange(num_x):
                if j >= (bord_y + offset + var_wrt_len_lower[i]) and j < (bord_y + offset + wg_height + var_wrt_len_upper[i]):
                    fg_mask[i, j] = True


#              Ezz      , Ez   , zs, ic, je
@cuda.jit(void(fa[:,:], fa[:,:], i4, i4, i4))
def map_efield_yline(Ezz, Ez, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Ezz[zwstep, j] = Ez[icut, j]


#              Hxz    , Hyz    , Hx     , Hy     , zs, ic, je
@cuda.jit(void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], i4, i4, i4))
def map_hfield_yline(Hxz, Hyz, Hx, Hy, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Hxz[zwstep, j] = 0.5 * (Hx[icut, j] + Hx[icut, j+1])
        Hyz[zwstep, j] = 0.5 * (Hy[icut, j] + Hy[icut+1, j])
