"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
from numba import cuda, f4, i4, void
from pyspeckle import create_exp_1D
import aux_funcs as aux
from math import sin, cos


def make_profile(num_x_eff, rough_std, rough_acl, dx, stol, atol, mtol):
    i = 0
    while(True):
        if i > 50000:
            raise Exception('No valid profile for current settings after 300,000 profile iterations')
        test_array = create_exp_1D(num_x_eff, 0.0, rough_std, rough_acl)
        sigma, acl, mm = aux.check_discretization(test_array, dx)
        if i % 100 == 0:
            print('checked {:} profile iterations'.format(i))
        if (1-stol)*rough_std*dx < sigma < (1+stol)*rough_std*dx and (1-atol)*rough_acl*dx < acl < (1+atol)*rough_acl*dx and -mtol < mm < mtol:
            print('Valid discretized profile found after {:} iterations'.format(i))
            break
        i += 1
    return test_array


def gen_fg_mask_sgl(num_x, num_y, bord_y, wg_width, port_len, rough_std, rough_acl, dx, mode='gen', correlation=1, upper_path='', lower_path='', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y], dtype=bool)
    num_x_eff = num_x - 2*port_len
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

    wid_wrt_len_upper = np.append(np.zeros(port_len), wid_wrt_len_upper)
    wid_wrt_len_upper = np.append(wid_wrt_len_upper, np.zeros(port_len))
    wid_wrt_len_lower = np.append(np.zeros(port_len), wid_wrt_len_lower)
    wid_wrt_len_lower = np.append(wid_wrt_len_lower, np.zeros(port_len))

    for j in range(num_y):
        for i in range(num_x):
            if j >= (bord_y + wid_wrt_len_lower[i]) and j < (bord_y + wg_width + wid_wrt_len_upper[i]):
                fg_mask[i, j] = True
    return fg_mask


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], i4, i4))
def update_ez(Ez, Hx, Hy, mod_e, curl_h, den_ex, den_ey, ie, je):
    i, j = cuda.grid(2)
    if i > 0 and i < ie-1:
        if j > 0 and j < je-1:
            Ez[i, j] = mod_e[i, j] * Ez[i, j] + curl_h[i, j]*((Hy[i, j] - Hy[i-1, j])/den_ex[i] + (Hx[i, j-1] - Hx[i, j])/den_ey[j])


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4, f4[:], f4[:], i4, i4))
def update_hx_hy(Hx, Hy, Ez, curl_e, den_hx, den_hy, ie, je):
    i, j = cuda.grid(2)
    if i < ie-1:
        if j < je-1:
            Hx[i, j] = Hx[i, j] + curl_e*(Ez[i, j] - Ez[i, j+1])/den_hy[j]
            Hy[i, j] = Hy[i, j] + curl_e*(Ez[i+1, j] - Ez[i, j])/den_hx[i]


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, f4, i4, i4, i4))
def update_ez_cpml_x(Psi_Ez_xlo, Psi_Ez_xhi, Ez, Hx, Hy, curl_h, be, ce, del_x, del_t, eps0, ie, je, cpml_range):
    i, j = cuda.grid(2)
    if i > 0 and i < cpml_range:
        if j > 0 and j < je-1:
            Psi_Ez_xlo[i, j] = be[i]*Psi_Ez_xlo[i, j] + ce[i]/del_x*(Hy[i, j] - Hy[i-1, j])
            Ez[i, j] = Ez[i, j] + curl_h[i, j]*del_x*Psi_Ez_xlo[i, j]
            Psi_Ez_xhi[i, j] = be[i]*Psi_Ez_xhi[i, j] + ce[i]/del_x*(Hy[ie-1-i, j] - Hy[ie-2-i, j])
            Ez[ie-1-i, j] = Ez[ie-1-i, j] + curl_h[ie-1-i, j]*del_x*Psi_Ez_xhi[i, j]


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, f4, i4, i4, i4))
def update_ez_cpml_y(Psi_Ez_ylo, Psi_Ez_yhi, Ez, Hx, Hy, curl_h, be, ce, del_x, del_t, eps0, ie, je, cpml_range):
    i, j = cuda.grid(2)
    if i > 0 and i < ie-1:
        if j > 0 and j < cpml_range:
            Psi_Ez_ylo[i, j] = be[j]*Psi_Ez_ylo[i, j] + ce[j]/del_x*(Hx[i, j-1] - Hx[i, j])
            Ez[i, j] = Ez[i, j] + curl_h[i, j]*del_x*Psi_Ez_ylo[i, j]
            Psi_Ez_yhi[i, j] = be[j]*Psi_Ez_yhi[i, j] + ce[j]/del_x*(Hx[i, je-2-j] - Hx[i, je-1-j])
            Ez[i, je-1-j] = Ez[i, je-1-j] + curl_h[i, je-1-j]*del_x*Psi_Ez_yhi[i, j]


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, f4, i4, i4, i4))
def update_hy_cpml_x(Psi_Hy_xlo, Psi_Hy_xhi, Hy, Ez, bh, ch, del_x, del_t, mu0, ie, je, cpml_range):
    i, j = cuda.grid(2)
    if i > 0 and i < cpml_range:
        if j < je-1:
            Psi_Hy_xlo[i, j] = bh[i]*Psi_Hy_xlo[i, j] + ch[i]/del_x*(Ez[i+1, j] - Ez[i, j])
            Hy[i, j] = Hy[i, j] + del_t/mu0*Psi_Hy_xlo[i, j]
            Psi_Hy_xhi[i, j] = bh[i]*Psi_Hy_xhi[i, j] + ch[i]/del_x*(Ez[ie-1-i, j] - Ez[ie-2-i, j])
            Hy[ie-2-i, j] = Hy[ie-2-i, j] + del_t/mu0*Psi_Hy_xhi[i, j]


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:], f4, f4, f4, i4, i4, i4))
def update_hx_cpml_y(Psi_Hx_ylo, Psi_Hx_yhi, Hx, Ez, bh, ch, del_x, del_t, mu0, ie, je, cpml_range):
    i, j = cuda.grid(2)
    if i < ie-1:
        if j > 0 and j < cpml_range:
            Psi_Hx_ylo[i, j] = bh[j]*Psi_Hx_ylo[i, j] + ch[j]/del_x*(Ez[i, j] - Ez[i, j+1])
            Hx[i, j] = Hx[i, j] + del_t/mu0*Psi_Hx_ylo[i, j]
            Psi_Hx_yhi[i, j] = bh[j]*Psi_Hx_yhi[i, j] + ch[j]/del_x*(Ez[i, je-2-j] - Ez[i, je-1-j])
            Hx[i, je-2-j] = Hx[i, je-2-j] + del_t/mu0*Psi_Hx_yhi[i, j]


@cuda.jit(void(f4[:,:], f4[:,:], f4[:,:], f4, f4, f4, i4, i4))
def simul_fft(Re, Im, F, freq, step, dt, ie, je):
    i, j = cuda.grid(2)
    if i < ie:
        if j < je:
            Re[i, j] = Re[i, j] + F[i, j] * cos( 2 * np.pi * freq * step * dt )
            Im[i, j] = Im[i, j] + F[i, j] * sin( 2 * np.pi * freq * step * dt )


@cuda.jit(void(f4[:,:], f4, i4, i4, i4, i4))
def update_source(Field, Source, sxb, sxe, syb, sye):
    i, j = cuda.grid(2)
    if i >= sxb and i < sxe:
        if j >= syb and j < sye:
            Field[i, j ] = Field[i, j] - Source
