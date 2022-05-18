# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:30:58 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
from numba import njit, prange
from pyspeckle import autocorrelation, create_exp_1D


def check_discretization(test_array, border, dx):
    dist = (test_array).astype(np.int)
    sigma = np.std(dist) * dx
    A = autocorrelation(dist.astype(np.float64))
    f = A - 1 / np.e
    for i in range(len(f)):
        if f[i] < 0:
            break
    acl = f[i-1]/(f[i-1]-f[i]) + i - 1
    acl *= dx
    mm = np.mean(dist)
    return sigma, acl, mm


def gen_fg_mask_sgl(num_x, num_y, bord_y, wg_width, port_len, rough_std, rough_acl, dx, prof_name, mode='gen', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y], dtype=bool)
    num_x_eff = num_x - 2*port_len
    wid_wrt_len = np.zeros(num_x)
    if mode == 'gen':
        i = 0
        while(True):
            if i > 100000:
                raise Exception('No valid profile for current settings after 100,000 profile iterations')
            test_array = create_exp_1D(num_x_eff, 0.0, rough_std, rough_acl)
            sigma, acl, mm = check_discretization(test_array, bord_y, dx)
            if i % 50 == 0:
                print('sigma pass:', (1-stol)*rough_std*dx < sigma < (1+stol)*rough_std*dx)
                print('ACL pass:', (1-atol)*rough_acl*dx < acl < (1+atol)*rough_acl*dx)
                print('mean pass:', -mtol < mm < mtol)
                print('checked {:} profile iterations'.format(i))
            if (1-stol)*rough_std*dx < sigma < (1+stol)*rough_std*dx and (1-atol)*rough_acl*dx < acl < (1+atol)*rough_acl*dx and -mtol < mm < mtol:
                print('Valid discretized profile found after {:} iterations'.format(i))
                break
            i += 1
        wid_wrt_len = test_array
        np.save('./rough_profiles/profile'+prof_name, wid_wrt_len)
        wid_wrt_len = np.append(np.zeros(port_len), wid_wrt_len)
        wid_wrt_len = np.append(wid_wrt_len, np.zeros(port_len))
    elif mode == 'smooth':
        pass
    else:
        wid_wrt_len = np.load(mode)
        wid_wrt_len = np.append(np.zeros(port_len), wid_wrt_len)
        wid_wrt_len = np.append(wid_wrt_len, np.zeros(port_len))

    for j in range(num_y):
        for i in range(num_x):
            if j >= (bord_y + wid_wrt_len[i]) and j < (bord_y + wg_width + wid_wrt_len[i]):
                fg_mask[i, j] = True
    return fg_mask


@njit(parallel=True)
def update_ex(Ex, Hz, mod_e, curl_h, den_ey, ie, je):
    for j in prange(1, je-1):
        for i in prange(1, ie-1):
            Ex[i, j] = mod_e[i, j]*Ex[i, j] + curl_h[i, j]*((Hz[i, j] - Hz[i, j-1])/den_ey[j])
    return Ex


@njit(parallel=True)
def update_ey(Ey, Hz, mod_e, curl_h, den_ex, ie, je):
    for j in prange(1, je-1):
        for i in prange(1, ie-1):
            Ey[i, j] = mod_e[i, j]*Ey[i, j] + curl_h[i, j]*((Hz[i-1, j] - Hz[i, j])/den_ex[i])
    return Ey


@njit(parallel=True)
def update_h(Ex, Ey, Hz, curl_e, den_hx, den_hy, ie, je):
    for j in prange(je-1):
        for i in prange(ie-1):
            Hz[i, j] = Hz[i, j] + curl_e*((Ex[i, j+1] - Ex[i, j])/den_hy[j] + (Ey[i, j] - Ey[i+1, j])/den_hx[i])
    return Hz


@njit(parallel=True)
def update_ex_cpml(Psi_Ex_ylo, Psi_Ex_yhi, Ex, Hz, curl_h, be, ce, del_x, del_t, eps0, ie, je, cpml_range):
    for j in prange(1, cpml_range):
        for i in prange(1, ie-1):
            Psi_Ex_ylo[i, j] = be[j]*Psi_Ex_ylo[i, j] + ce[j]*(Hz[i, j] - Hz[i, j-1])/del_x
            Ex[i, j] = Ex[i, j] + curl_h[i, j]*del_x*Psi_Ex_ylo[i, j]
            Psi_Ex_yhi[i, j] = be[j]*Psi_Ex_yhi[i, j] + ce[j]*(Hz[i, je-1-j] - Hz[i, je-2-j])/del_x
            Ex[i, je-1-j] = Ex[i, je-1-j] + curl_h[i, je-1-j]*del_x*Psi_Ex_yhi[i, j]
    return Psi_Ex_ylo, Psi_Ex_yhi, Ex


@njit(parallel=True)
def update_ey_cpml(Psi_Ey_xlo, Psi_Ey_xhi, Ey, Hz, curl_h, be, ce, del_x, del_t, eps0, ie, je, cpml_range):
    for j in prange(1, je-1):
        for i in prange(1, cpml_range):
            Psi_Ey_xlo[i, j] = be[i]*Psi_Ey_xlo[i, j] + ce[i]*(Hz[i-1, j] - Hz[i, j])/del_x
            Ey[i, j] = Ey[i, j] + curl_h[i, j]*del_x*Psi_Ey_xlo[i, j]
            Psi_Ey_xhi[i, j] = be[i]*Psi_Ey_xhi[i, j] + ce[i]*(Hz[ie-2-i, j] - Hz[ie-1-i, j])/del_x
            Ey[ie-1-i, j] = Ey[ie-1-i, j] + curl_h[ie-1-i, j]*del_x*Psi_Ey_xhi[i, j]
    return Psi_Ey_xlo, Psi_Ey_xhi, Ey


@njit(parallel=True)
def update_hz_cpml(Psi_Hz_xlo, Psi_Hz_xhi, Psi_Hz_ylo, Psi_Hz_yhi, Hz, Ex, Ey, bh, ch, del_x, del_t, mu0, ie, je, cpml_range):
    for j in prange(je-1):
        for i in prange(1, cpml_range):
            Psi_Hz_xlo[i, j] = bh[i]*Psi_Hz_xlo[i, j] + ch[i]*(Ey[i, j] - Ey[i+1, j])/del_x
            Hz[i, j] = Hz[i, j] + del_t/mu0*Psi_Hz_xlo[i, j]
            Psi_Hz_xhi[i, j] = bh[i]*Psi_Hz_xhi[i, j] + ch[i]*(Ey[ie-2-i, j] - Ey[ie-1-i, j])/del_x
            Hz[ie-2-i, j] = Hz[ie-2-i, j] + del_t/mu0*Psi_Hz_xhi[i, j]
    for j in prange(1, cpml_range):
        for i in prange(ie-1):
            Psi_Hz_ylo[i, j] = bh[j]*Psi_Hz_ylo[i, j] + ch[j]*(Ex[i, j+1] - Ex[i, j])/del_x
            Hz[i, j] = Hz[i, j] + del_t/mu0*Psi_Hz_ylo[i, j]
            Psi_Hz_yhi[i, j] = bh[j]*Psi_Hz_yhi[i, j] + ch[j]*(Ex[i, je-1-j] - Ex[i, je-2-j])/del_x
            Hz[i, je-2-j] = Hz[i, je-2-j] + del_t/mu0*Psi_Hz_yhi[i, j]
    return Psi_Hz_xlo, Psi_Hz_xhi, Psi_Hz_ylo, Psi_Hz_yhi, Hz
