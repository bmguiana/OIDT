"""
Author: B. Guiana

Description:


Acknowledgement: This project was completed as part of research conducted with
                 my major professor and advisor, Prof. Ata Zadehgol, at the
                 Applied and Computational Electromagnetics Signal and Power
                 Integrity (ACEM-SPI) Lab while working toward the Ph.D. in
                 Electrical Engineering at the University of Idaho, Moscow,
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


#                          Ex     , Hz     , kye  , cb     , ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:], fa[:,:], i4, i4, i4, i4))
def update_ex(Ex, Hz, kye, cb, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j >= jb and j < je:
            Ex[i, j] = Ex[i, j] + cb[i, j] * ( Hz[i, j+1] - Hz[i, j] ) / kye[j]


#                          Ey     , Hz     , kxe  , cb     , ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:], fa[:,:], i4, i4, i4, i4))
def update_ey(Ey, Hz, kxe, cb, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j > jb and j < je:
            Ey[i, j] = Ey[i, j] + cb[i, j] * ( Hz[i, j] - Hz[i+1, j] ) / kxe[i]


#                          Ex     , Ey     , Hz     , kxh  , kyh  , db, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa, i4, i4, i4, i4))
def update_hz(Ex, Ey, Hz, kxh, kyh, db, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j > jb and j < je:
            Hz[i, j] = Hz[i, j] + db * ( ( Ex[i, j] - Ex[i, j-1] ) / kyh[j] - ( Ey[i, j] - Ey[i-1, j] ) / kxh[i] )


#                          Field  , Sc, sx, sy, sx, sy
@cuda.jit(func_or_sig=void(fa[:,:], fa, i4, i4, i4, i4))
def update_source(Field, Source, sxb, syb, sxe, sye):
    i, j = cuda.grid(2)
    if i >= sxb and i < sxe:
        if j >= syb and j < sye:
            Field[i, j] = Field[i, j] - Source


#                          PEx_yl , PEx_yh , Ex     , Hz     , be   , ce   , cb     , dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa[:,:], fa, i4, i4, i4, i4, i4))
def update_pml_ex_yinc(Psi_Ex_ylo, Psi_Ex_yhi, Ex, Hz, be, ce, cb, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j < pmle:
            Psi_Ex_ylo[i, j] = be[-(j+1)] * Psi_Ex_ylo[i, j] + ce[-(j+1)] * ( Hz[i, jb+j+1] - Hz[i, jb+j] ) / dy
            Ex[i, jb+j] = Ex[i, jb+j] + cb[i, jb+j] * Psi_Ex_ylo[i, j]
            Psi_Ex_yhi[i, j] = be[j] * Psi_Ex_yhi[i, j] + ce[j] * ( Hz[i, je-pmle+j+1] - Hz[i, je-pmle+j] ) / dy
            Ex[i, je-pmle+j] = Ex[i, je-pmle+j] + cb[i, je-pmle+j] * Psi_Ex_yhi[i, j]


#                          PEy_xl , PEy_xh , Ey     , Hz     , be   , ce   , cb     , dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa[:,:], fa, i4, i4, i4, i4, i4))
def update_pml_ey_xinc(Psi_Ey_xlo, Psi_Ey_xhi, Ey, Hz, be, ce, cb, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i < pmle:
        if j > jb and j < je:
            Psi_Ey_xlo[i, j] = be[-(i+1)] * Psi_Ey_xlo[i, j] + ce[-(i+1)] * ( Hz[ib+i+1, j] - Hz[ib+i, j] ) / dx
            Ey[ib+i, j] = Ey[ib+i, j] - cb[ib+i, j] * Psi_Ey_xlo[i, j]
            Psi_Ey_xhi[i, j] = be[i] * Psi_Ey_xhi[i, j] + ce[i] * ( Hz[ie-pmle+i+1, j] - Hz[ie-pmle+i, j] ) / dx
            Ey[ie-pmle+i, j] = Ey[ie-pmle+i, j] - cb[ie-pmle+i, j] * Psi_Ey_xhi[i, j]


#                          PHz_xl , PHz_xh , Ey     , Hz     , bh   , ch   , db, dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa, fa, i4, i4, i4, i4, i4))
def update_pml_hz_xinc(Psi_Hz_xlo, Psi_Hz_xhi, Ey, Hz, bh, ch, db, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > 0 and i < pmle:
        if j > jb and j < je:
            Psi_Hz_xlo[i, j] = bh[-i] * Psi_Hz_xlo[i, j] + ch[-i] * ( Ey[ib+i, j] - Ey[ib+i-1, j] ) / dx
            Hz[ib+i, j] = Hz[ib+i, j] - db * Psi_Hz_xlo[i, j]
            Psi_Hz_xhi[i, j] = bh[i] * Psi_Hz_xhi[i, j] + ch[i] * ( Ey[ie-pmle+i, j] - Ey[ie-pmle+i-1, j] ) / dx
            Hz[ie-pmle+i, j] = Hz[ie-pmle+i, j] - db * Psi_Hz_xhi[i, j]


#                          PHz_yl , PHz_yh , Ex     , Hz     , bh   , ch   , db, dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:], fa[:], fa, fa, i4, i4, i4, i4, i4))
def update_pml_hz_yinc(Psi_Hz_ylo, Psi_Hz_yhi, Ex, Hz, bh, ch, db, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j > 0 and j < pmle:
            Psi_Hz_ylo[i, j] = bh[-j] * Psi_Hz_ylo[i, j] + ch[-j] * ( Ex[i, jb+j] - Ex[i, jb+j-1] ) / dy
            Hz[i, jb+j] = Hz[i, jb+j] + db * Psi_Hz_ylo[i, j]
            Psi_Hz_yhi[i, j] = bh[j] * Psi_Hz_yhi[i, j] + ch[j] * ( Ex[i, je-pmle+j] - Ex[i, je-pmle+j-1] ) / dy
            Hz[i, je-pmle+j] = Hz[i, je-pmle+j] + db * Psi_Hz_yhi[i, j]


#                          Rx     , Ry     , Ix     , Iy     , Ex     , Ey     , cs, sn, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa[:,:], fa, fa, i4, i4, i4, i4))
def simul_fft_efield(Rx, Ry, Ix, Iy, Ex, Ey, cos, sin, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rx[i, j] = Rx[i, j] + 0.5 * (Ex[i, j] + Ex[i+1, j]) * cos
            Ix[i, j] = Ix[i, j] - 0.5 * (Ex[i, j] + Ex[i+1, j]) * sin
            Ry[i, j] = Ry[i, j] + 0.5 * (Ey[i, j] + Ey[i, j+1]) * cos
            Iy[i, j] = Iy[i, j] - 0.5 * (Ey[i, j] + Ey[i, j+1]) * sin


#                          Rz     , Iz     , Hz     , cs, sn, ib, jb, ie, je
@cuda.jit(func_or_sig=void(fa[:,:], fa[:,:], fa[:,:], fa, fa, i4, i4, i4, i4))
def simul_fft_hfield(Rz, Iz, Hz, cos, sin, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rz[i, j] = Rz[i, j] + 0.25 * (Hz[i, j] + Hz[i+1, j] + Hz[i, j+1] + Hz[i+1, j+1]) * cos
            Iz[i, j] = Iz[i, j] - 0.25 * (Hz[i, j] + Hz[i+1, j] + Hz[i, j+1] + Hz[i+1, j+1]) * sin


@njit(parallel=True)
def average_materials(Mask, X, Y, ie, je):
    for j in prange(1, je):
        for i in prange(1, ie):
            X[i, j] = 0.5 * (Mask[i, j] + Mask[i-1, j])
            Y[i, j] = 0.5 * (Mask[i, j] + Mask[i, j-1])


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


def generate_fg_mask(num_x, num_y, bord_y, wg_height, pitch, num_lines, settling_range, end_range, rough_std, rough_acl, dx, mode='gen', out_dir='rough_profiles', correlation=3, upper_name='', lower_name='', atol=0.1, stol=0.1, mtol=0.01):
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

    apply_mask(fg_mask, num_x, num_y, bord_y, wg_height, pitch, num_lines, var_wrt_len_lower, var_wrt_len_upper)
    return fg_mask


@njit(parallel=True)
def apply_mask(fg_mask, num_x, num_y, bord_y, wg_height, pitch, num_lines, var_wrt_len_lower, var_wrt_len_upper):
    for l in range(num_lines):
        offset = l * pitch
        for j in prange(num_y):
            for i in prange(num_x):
                if j >= (bord_y + offset + var_wrt_len_lower[i]) and j < (bord_y + offset + wg_height + var_wrt_len_upper[i]):
                    fg_mask[i, j] = True


#              Exz    , Eyz    , Ex     , Ey     , zs, ic, je
@cuda.jit(void(fa[:,:], fa[:,:], fa[:,:], fa[:,:], i4, i4, i4))
def map_efield_yline(Exz, Eyz, Ex, Ey, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Exz[zwstep, j] = 0.5 * (Ex[icut, j] + Ex[icut+1, j])
        Eyz[zwstep, j] = 0.5 * (Ey[icut, j] + Ey[icut, j+1])


#              Hzz    , Hz     , zs, ic, je
@cuda.jit(void(fa[:,:], fa[:,:], i4, i4, i4))
def map_hfield_yline(Hzz, Hz, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Hzz[zwstep, j] = 0.25 * (Hz[icut, j] + Hz[icut+1, j] + Hz[icut, j+1] + Hz[icut+1, j+1])
