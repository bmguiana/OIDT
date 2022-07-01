"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
from numba import cuda, f4, i4, void
from pyspeckle import create_exp_1D
import aux_funcs as aux
from math import cos, sin


def make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol):
    i = 0
    while(True):
        if i > 100000:
            raise Exception('No valid profile for current settings after 300,000 profile iterations')
        test_array = create_exp_1D(num_x_eff, 0.0, rough_std, rough_acl).astype(np.float32)
        sigma, acl, mm = aux.check_discretization(test_array, dx)
        if i % 100 == 0:
            print('checked {:} profile iterations'.format(i))
        if ((1-stol)*rough_std*dx < sigma < (1+stol)*rough_std*dx) and ((1-atol)*rough_acl*dx < acl < (1+atol)*rough_acl*dx) and (-mtol < mm < mtol):
            if not(np.isnan(sigma) or np.isnan(acl) or np.isnan(mm)) or sigma==0 or acl==0:
                print('Valid discretized profile found after {:} iterations'.format(i))
                break
        i += 1
    return test_array


def gen_fg_mask_sgl(num_x, num_y, bord_y, wg_width, settling_range, end_range, rough_std, rough_acl, dx, mode='gen', correlation=1, upper_path='', lower_path='', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y], dtype=bool)
    num_x_eff = num_x - settling_range - end_range
    if mode == 'gen':
        wid_wrt_len_upper = make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        if correlation == 1:
            wid_wrt_len_lower = wid_wrt_len_upper.copy()
        elif correlation == 2:
            wid_wrt_len_lower = -1*wid_wrt_len_upper
        elif correlation == 3:
            wid_wrt_len_lower = make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        else:
            raise Exception('Choose a valid correlation type')
    elif mode == 'smooth':
        wid_wrt_len_upper = np.zeros(num_x_eff)
        wid_wrt_len_lower = np.zeros(num_x_eff)
    elif mode == 'load':
        if correlation == 1:
            wid_wrt_len_upper = np.load(upper_path+'.npy')
            wid_wrt_len_lower = np.load(upper_path+'.npy')
        elif correlation == 2:
            wid_wrt_len_upper = np.load(upper_path+'.npy')
            wid_wrt_len_lower = -1*np.load(upper_path+'.npy')
        elif correlation == 3:
            wid_wrt_len_upper = np.load(upper_path+'.npy')
            wid_wrt_len_lower = np.load(upper_path+'.npy')
    else:
        raise Exception('Choose a valid roughness profile mode')

    aux.mkdir('rough_profiles')
    np.save(upper_path, wid_wrt_len_upper)
    np.save(lower_path, wid_wrt_len_lower)

    wid_wrt_len_upper = np.append(np.zeros(settling_range), wid_wrt_len_upper)
    wid_wrt_len_upper = np.append(wid_wrt_len_upper, np.zeros(end_range))
    wid_wrt_len_lower = np.append(np.zeros(settling_range), wid_wrt_len_lower)
    wid_wrt_len_lower = np.append(wid_wrt_len_lower, np.zeros(end_range))

    for j in range(num_y):
        for i in range(num_x):
            if j >= (bord_y + wid_wrt_len_lower[i]) and j < (bord_y + wg_width + wid_wrt_len_upper[i]):
                fg_mask[i, j] = True
    return fg_mask


#                          Ex     , Hz     , kye  , cb     , ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:], f4[:,:], i4, i4, i4, i4))
def update_ex(Ex, Hz, kye, cb, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j >= jb and j < je:
            Ex[i, j] = Ex[i, j] + cb[i, j] * (Hz[i, j+1] - Hz[i, j]) / kye[j]


#                          Ey     , Hz     , kxe  , cb     , ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:], f4[:,:], i4, i4, i4, i4))
def update_ey(Ey, Hz, kxe, cb, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i >= ib and i < ie:
        if j > jb and j < je:
            Ey[i, j] = Ey[i, j] + cb[i, j] * (Hz[i, j] - Hz[i+1, j]) / kxe[i]


#                          Ex     , Ey     , Hz     , kxh  , kyh  , db, ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, i4, i4, i4, i4))
def update_hz(Ex, Ey, Hz, kxh, kyh, db, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j > jb and j < je:
            Hz[i, j] = Hz[i, j] + db * ( (Ex[i, j] - Ex[i, j-1]) / kyh[j] - (Ey[i, j] - Ey[i-1, j]) / kxh[i] )


#                          F      , S    , sx, sm, , step
@cuda.jit(func_or_sig=void(f4[:,:], f4[:], i4, i4, i4, i4))
def update_source(Field, Source, sx, sy_min, sy_max, step):
    i, j = cuda.grid(2)
    if i == sx and j >= sy_min and j < sy_max:
        Field[i, j] = Field[i, j] + Source[step]

#                          Re     , Im     , F      , freq, step, dt, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4, f4, f4, i4, i4))
def simul_fft(Re, Im, F, freq, step, dt, ie, je):
    i, j = cuda.grid(2)
    if i < ie:
        if j < je:
            Re[i, j] = Re[i, j] + F[i, j] * cos( 2 * np.pi * freq * step * dt )
            Im[i, j] = Im[i, j] + F[i, j] * sin( 2 * np.pi * freq * step * dt )


#                          PEx_yl , PEx_yh , Ex     , Hz     , be   , ce   , cb     , dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4[:,:], f4, i4, i4, i4, i4 ,i4))
def update_pml_ex_yinc(Psi_Ex_ylo, Psi_Ex_yhi, Ex, Hz, be, ce, cb, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j < pmle:
            Psi_Ex_ylo[i, j] = be[-(j+1)]*Psi_Ex_ylo[i, j] + ce[-(j+1)]*(Hz[i, jb+j+1] - Hz[i, jb+j])/dy
            Ex[i, jb+j] = Ex[i, jb+j] + cb[i, jb+j]*Psi_Ex_ylo[i, j]
            Psi_Ex_yhi[i, j] = be[j] * Psi_Ex_yhi[i, j] + ce[j] * (Hz[i, je-pmle+j+1] - Hz[i, je-pmle+j])/dy
            Ex[i, je-pmle+j] = Ex[i, je-pmle+j] + cb[i, je-pmle+j]*Psi_Ex_yhi[i, j]


#                          PEy_xl , PEy_xh , Ey     , Hz     , be   , ce   , cb     , dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4[:,:], f4, i4, i4, i4, i4, i4))
def update_pml_ey_xinc(Psi_Ey_xlo, Psi_Ey_xhi, Ey, Hz, be, ce, cb, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i < pmle:
        if j > jb and j < je and i < pmle:
            Psi_Ey_xlo[i, j] = be[-(i+1)]*Psi_Ey_xlo[i, j] + ce[-(i+1)]*(Hz[ib+i, j] - Hz[ib+i+1, j])/dx
            Ey[ib+i, j] = Ey[ib+i, j] + cb[ib+i, j]*Psi_Ey_xlo[i, j]
            Psi_Ey_xhi[i, j] = be[i] * Psi_Ey_xhi[i, j] + ce[i] * (Hz[ie-pmle+i, j] - Hz[ie-pmle+i+1, j])/dx
            Ey[ie-pmle+i, j] = Ey[ie-pmle+i, j] + cb[ie-pmle+i, j]*Psi_Ey_xhi[i, j]


#                          PHz_xl , PHz_xh , Ey     , Hz     , bh   , ch   , db, dx, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4))
def update_pml_hz_xinc(Psi_Hz_xlo, Psi_Hz_xhi, Ey, Hz, bh, ch, db, dx, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > 0 and i < pmle:
        if j > jb and j < je:
            Psi_Hz_xlo[i, j] = bh[-i]*Psi_Hz_xlo[i, j] + ch[-i]*(Ey[ib+i, j] - Ey[ib+i-1, j])/dx
            Hz[ib+i, j] = Hz[ib+i, j] - db*Psi_Hz_xlo[i, j]
            Psi_Hz_xhi[i, j] = bh[i]*Psi_Hz_xhi[i, j] + ch[i]*(Ey[ie-pmle+i, j] - Ey[ie-pmle+i-1, j])/dx
            Hz[ie-pmle+i, j] = Hz[ie-pmle+i, j] - db*Psi_Hz_xhi[i, j]


#                          PHz_yl , PHz_yh , Ex     , Hz     , bh   , ch   , db, dy, pe, ib, jb, ie, je
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4))
def update_pml_hz_yinc(Psi_Hz_ylo, Psi_Hz_yhi, Ex, Hz, bh, ch, db, dy, pmle, ib, jb, ie, je):
    i, j = cuda.grid(2)
    if i > ib and i < ie:
        if j > 0 and j < pmle:
            Psi_Hz_ylo[i, j] = bh[-j]*Psi_Hz_ylo[i, j] + ch[-j]*(Ex[i, jb+j] - Ex[i, jb+j-1])/dy
            Hz[i, jb+j] = Hz[i, jb+j] + db*Psi_Hz_ylo[i, j]
            Psi_Hz_yhi[i, j] = bh[j]*Psi_Hz_yhi[i, j] + ch[j]*(Ex[i, je-pmle+j] - Ex[i, je-pmle+j-1])/dy
            Hz[i, je-pmle+j] = Hz[i, je-pmle+j] + db*Psi_Hz_yhi[i, j]


#              Exz    , Ex     , zs, ic, je
@cuda.jit(void(f4[:,:], f4[:,:], i4, i4, i4))
def map_efield_zwave(Exz, Ex, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Exz[zwstep, j] = 0.5 * (Ex[icut, j] + Ex[icut+1, j])


#              Hzz    , Hz     , zs, ic, je
@cuda.jit(void(f4[:,:], f4[:,:], i4, i4, i4))
def map_hfield_zwave(Hzz, Hz, zwstep, icut, je):
    j = cuda.grid(1)
    if j < je:
        Hzz[zwstep, j] = 0.25 * (Hz[icut, j] + Hz[icut+1, j] + Hz[icut, j+1] + Hz[icut+1, j+1])
