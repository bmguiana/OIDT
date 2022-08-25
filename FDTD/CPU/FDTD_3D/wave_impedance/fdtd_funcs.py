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


def generate_fg_mask(num_x, num_y, num_z, bord_x, bord_y, wg_width, wg_height, port_len, rough_std, rough_acl, dx, prof_name, mode='gen', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y, num_z], dtype=bool)
    num_z_eff = num_z - 2*port_len
    wid_wrt_len = np.zeros(num_z)
    hei_wrt_len = np.zeros(num_z)
    for k in prange(num_z):
        for j in prange(num_y):
            for i in prange(num_x):
                if j >= (bord_y + hei_wrt_len[k]) and j < (bord_y + wg_height + hei_wrt_len[k]) and i >= (bord_x + wid_wrt_len[k]) and i < (bord_x + wg_width + wid_wrt_len[k]):
                    fg_mask[i, j, k] = True
    return fg_mask


@njit(parallel=True)
def update_e(Ex, Ey, Ez, Hx, Hy, Hz, mod_e, curl_h, den_ex, den_ey, den_ez, ie, je, ke):
    for k in prange(1, ke-1):
        for j in prange(1, je-1):
            for i in prange(1, ie-1):
                Ex[i, j, k] = mod_e[i, j, k]*Ex[i, j, k] + curl_h[i, j, k]*((Hz[i, j, k] - Hz[i, j-1, k])/den_ey[j] + (Hy[i, j, k-1] - Hy[i, j, k])/den_ez[k])
                Ey[i, j, k] = mod_e[i, j, k]*Ey[i, j, k] + curl_h[i, j, k]*((Hx[i, j, k] - Hx[i, j, k-1])/den_ez[k] + (Hz[i-1, j, k] - Hz[i, j, k])/den_ex[i])
                Ez[i, j, k] = mod_e[i, j, k]*Ez[i, j, k] + curl_h[i, j, k]*((Hy[i, j, k] - Hy[i-1, j, k])/den_ex[i] + (Hx[i, j-1, k] - Hx[i, j, k])/den_ey[j])
    return Ex, Ey, Ez


@njit(parallel=True)
def update_h(Ex, Ey, Ez, Hx, Hy, Hz, curl_e, den_hx, den_hy, den_hz, ie, je, ke):
    for k in prange(ke-1):
        for j in prange(je-1):
            for i in prange(ie-1):
                Hx[i, j, k] = Hx[i, j, k] + curl_e*((Ey[i, j, k+1] - Ey[i, j, k])/den_hz[k] + (Ez[i, j, k] - Ez[i, j+1, k])/den_hy[j])
                Hy[i, j, k] = Hy[i, j, k] + curl_e*((Ez[i+1, j, k] - Ez[i, j, k])/den_hx[i] + (Ex[i, j, k] - Ex[i, j, k+1])/den_hz[k])
                Hz[i, j, k] = Hz[i, j, k] + curl_e*((Ex[i, j+1, k] - Ex[i, j, k])/den_hy[j] + (Ey[i, j, k] - Ey[i+1, j, k])/den_hx[i])
    return Hx, Hy, Hz


@njit(parallel=True)
def update_ex_cpml(Psi_Ex_ylo, Psi_Ex_yhi, Psi_Ex_zlo, Psi_Ex_zhi, Ex, Hy, Hz, curl_h, be, ce, del_x, del_t, eps0, ie, je, ke, cpml_range):
    for k in prange(1, ke-1):
        for j in prange(1, cpml_range):
            for i in prange(1, ie-1):
                Psi_Ex_ylo[i, j, k] = be[j]*Psi_Ex_ylo[i, j, k] + ce[j]*(Hz[i, j, k] - Hz[i, j-1, k])/del_x
                Ex[i, j, k] = Ex[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ex_ylo[i, j, k]
                Psi_Ex_yhi[i, j, k] = be[j]*Psi_Ex_yhi[i, j, k] + ce[j]*(Hz[i, je-1-j, k] - Hz[i, je-2-j, k])/del_x
                Ex[i, je-1-j, k] = Ex[i, je-1-j, k] + curl_h[i, je-1-j, k]*del_x*Psi_Ex_yhi[i, j, k]
    for k in prange(1, cpml_range):
        for j in prange(1, je-1):
            for i in prange(1, ie-1):
                Psi_Ex_zlo[i, j, k] = be[k]*Psi_Ex_zlo[i, j, k] + ce[k]*(Hy[i, j, k-1] - Hy[i, j, k])/del_x
                Ex[i, j, k] = Ex[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ex_zlo[i, j, k]
                Psi_Ex_zhi[i, j, k] = be[k]*Psi_Ex_zhi[i, j, k] + ce[k]*(Hy[i, j, ke-2-k] - Hy[i, j, ke-1-k])/del_x
                Ex[i, j, ke-1-k] = Ex[i, j, ke-1-k] + curl_h[i, j, ke-1-k]*del_x*Psi_Ex_zhi[i, j, k]
    return Psi_Ex_ylo, Psi_Ex_yhi, Psi_Ex_zlo, Psi_Ex_zhi, Ex


@njit(parallel=True)
def update_ey_cpml(Psi_Ey_xlo, Psi_Ey_xhi, Psi_Ey_zlo, Psi_Ey_zhi, Ey, Hx, Hz, curl_h, be, ce, del_x, del_t, eps0, ie, je, ke, cpml_range):
    for k in prange(1, ke-1):
        for j in prange(1, je-1):
            for i in prange(1, cpml_range):
                Psi_Ey_xlo[i, j, k] = be[i]*Psi_Ey_xlo[i, j, k] + ce[i]*(Hz[i-1, j, k] - Hz[i, j, k])/del_x
                Ey[i, j, k] = Ey[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ey_xlo[i, j, k]
                Psi_Ey_xhi[i, j, k] = be[i]*Psi_Ey_xhi[i, j, k] + ce[i]*(Hz[ie-2-i, j, k] - Hz[ie-1-i, j, k])/del_x
                Ey[ie-1-i, j, k] = Ey[ie-1-i, j, k] + curl_h[ie-1-i, j, k]*del_x*Psi_Ey_xhi[i, j, k]
    for k in prange(1, cpml_range):
        for j in prange(1, je-1):
            for i in prange(1, ie-1):
                Psi_Ey_zlo[i, j, k] = be[k]*Psi_Ey_zlo[i, j, k] + ce[k]*(Hx[i, j, k] - Hx[i, j, k-1])/del_x
                Ey[i, j, k] = Ey[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ey_zlo[i, j, k]
                Psi_Ey_zhi[i, j, k] = be[k]*Psi_Ey_zhi[i, j, k] + ce[k]*(Hx[i, j, ke-1-k] - Hx[i, j, ke-2-k])/del_x
                Ey[i, j, ke-1-k] = Ey[i, j, ke-1-k] + curl_h[i, j, ke-1-k]*del_x*Psi_Ey_zhi[i, j, k]
    return Psi_Ey_xlo, Psi_Ey_xhi, Psi_Ey_zlo, Psi_Ey_zhi, Ey


@njit(parallel=True)
def update_ez_cpml(Psi_Ez_xlo, Psi_Ez_xhi, Psi_Ez_ylo, Psi_Ez_yhi, Ez, Hx, Hy, curl_h, be, ce, del_x, del_t, eps0, ie, je, ke, cpml_range):
    for k in prange(1, ke-1):
        for j in prange(1, je-1):
            for i in prange(1, cpml_range):
                Psi_Ez_xlo[i, j, k] = be[i]*Psi_Ez_xlo[i, j, k] + ce[i]*(Hy[i, j, k] - Hy[i-1, j, k])/del_x
                Ez[i, j, k] = Ez[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ez_xlo[i, j, k]
                Psi_Ez_xhi[i, j, k] = be[i]*Psi_Ez_xhi[i, j, k] + ce[i]*(Hy[ie-1-i, j, k] - Hy[ie-2-i, j, k])/del_x
                Ez[ie-1-i, j, k] = Ez[ie-1-i, j, k] + curl_h[ie-1-i, j, k]*del_x*Psi_Ez_xhi[i, j, k]
    for k in prange(1, ke-1):
        for j in prange(1, cpml_range):
            for i in prange(1, ie-1):
                Psi_Ez_ylo[i, j, k] = be[j]*Psi_Ez_ylo[i, j, k] + ce[j]*(Hx[i, j-1, k] - Hx[i, j, k])/del_x
                Ez[i, j, k] = Ez[i, j, k] + curl_h[i, j, k]*del_x*Psi_Ez_ylo[i, j, k]
                Psi_Ez_yhi[i, j, k] = be[j]*Psi_Ez_yhi[i, j, k] + ce[j]*(Hx[i, je-2-j, k] - Hx[i, je-1-j, k])/del_x
                Ez[i, je-1-j, k] = Ez[i, je-1-j, k] + curl_h[i, je-1-j, k]*del_x*Psi_Ez_yhi[i, j, k]
    return Psi_Ez_xlo, Psi_Ez_xhi, Psi_Ez_ylo, Psi_Ez_yhi, Ez


@njit(parallel=True)
def update_hx_cpml(Psi_Hx_ylo, Psi_Hx_yhi, Psi_Hx_zlo, Psi_Hx_zhi, Hx, Ey, Ez, bh, ch, del_x, del_t, mu0, ie, je, ke, cpml_range):
    for k in prange(ke-1):
        for j in prange(1, cpml_range):
            for i in prange(ie-1):
                Psi_Hx_ylo[i, j, k] = bh[j]*Psi_Hx_ylo[i, j, k] + ch[j]*(Ez[i, j, k] - Ez[i, j+1, k])/del_x
                Hx[i, j, k] = Hx[i, j, k] + del_t/mu0*Psi_Hx_ylo[i, j, k]
                Psi_Hx_yhi[i, j, k] = bh[j]*Psi_Hx_yhi[i, j, k] + ch[j]*(Ez[i, je-2-j, k] - Ez[i, je-1-j, k])/del_x
                Hx[i, je-2-j, k] = Hx[i, je-2-j, k] + del_t/mu0*Psi_Hx_yhi[i, j, k]
    for k in prange(1, cpml_range):
        for j in prange(je-1):
            for i in prange(ie-1):
                Psi_Hx_zlo[i, j, k] = bh[k]*Psi_Hx_zlo[i, j, k] + ch[k]*(Ey[i, j, k+1] - Ey[i, j, k])/del_x
                Hx[i, j, k] = Hx[i, j, k] + del_t/mu0*Psi_Hx_zlo[i, j, k]
                Psi_Hx_zhi[i, j, k] = bh[k]*Psi_Hx_zhi[i, j, k] + ch[k]*(Ey[i, j, ke-1-k] - Ey[i, j, ke-2-k])/del_x
                Hx[i, j, ke-2-k] = Hx[i, j, ke-2-k] + del_t/mu0*Psi_Hx_zhi[i, j, k]
    return Psi_Hx_ylo, Psi_Hx_yhi, Psi_Hx_zlo, Psi_Hx_zhi, Hx


@njit(parallel=True)
def update_hy_cpml(Psi_Hy_xlo, Psi_Hy_xhi, Psi_Hy_zlo, Psi_Hy_zhi, Hy, Ex, Ez, bh, ch, del_x, del_t, mu0, ie, je, ke, cpml_range):
    for k in prange(ke-1):
        for j in prange(je-1):
            for i in prange(1, cpml_range):
                Psi_Hy_xlo[i, j, k] = bh[i]*Psi_Hy_xlo[i, j, k] + ch[i]*(Ez[i+1, j, k] - Ez[i, j, k])/del_x
                Hy[i, j, k] = Hy[i, j, k] + del_t/mu0*Psi_Hy_xlo[i, j, k]
                Psi_Hy_xhi[i, j, k] = bh[i]*Psi_Hy_xhi[i, j, k] + ch[i]*(Ez[ie-1-i, j, k] - Ez[ie-2-i, j, k])/del_x
                Hy[ie-2-i, j, k] = Hy[ie-2-i, j, k] + del_t/mu0*Psi_Hy_xhi[i, j, k]
    for k in prange(1, cpml_range):
        for j in prange(je-1):
            for i in prange(ie-1):
                Psi_Hy_zlo[i, j, k] = bh[k]*Psi_Hy_zlo[i, j, k] + ch[k]*(Ex[i, j, k] - Ex[i, j, k+1])/del_x
                Hy[i, j, k] = Hy[i, j, k] + del_t/mu0*Psi_Hy_zlo[i, j, k]
                Psi_Hy_zhi[i, j, k] = bh[k]*Psi_Hy_zhi[i, j, k] + ch[k]*(Ex[i, j, ke-2-k] - Ex[i, j, ke-1-k])/del_x
                Hy[i, j, ke-2-k] = Hy[i, j, ke-2-k] + del_t/mu0*Psi_Hy_zhi[i, j, k]
    return Psi_Hy_xlo, Psi_Hy_xhi, Psi_Hy_zlo, Psi_Hy_zhi, Hy


@njit(parallel=True)
def update_hz_cpml(Psi_Hz_xlo, Psi_Hz_xhi, Psi_Hz_ylo, Psi_Hz_yhi, Hz, Ex, Ey, bh, ch, del_x, del_t, mu0, ie, je, ke, cpml_range):
    for k in prange(ke-1):
        for j in prange(je-1):
            for i in prange(1, cpml_range):
                Psi_Hz_xlo[i, j, k] = bh[i]*Psi_Hz_xlo[i, j, k] + ch[i]*(Ey[i, j, k] - Ey[i+1, j, k])/del_x
                Hz[i, j, k] = Hz[i, j, k] + del_t/mu0*Psi_Hz_xlo[i, j, k]
                Psi_Hz_xhi[i, j, k] = bh[i]*Psi_Hz_xhi[i, j, k] + ch[i]*(Ey[ie-2-i, j, k] - Ey[ie-1-i, j, k])/del_x
                Hz[ie-2-i, j, k] = Hz[ie-2-i, j, k] + del_t/mu0*Psi_Hz_xhi[i, j, k]
    for k in prange(ke-1):
        for j in prange(1, cpml_range):
            for i in prange(ie-1):
                Psi_Hz_ylo[i, j, k] = bh[j]*Psi_Hz_ylo[i, j, k] + ch[j]*(Ex[i, j+1, k] - Ex[i, j, k])/del_x
                Hz[i, j, k] = Hz[i, j, k] + del_t/mu0*Psi_Hz_ylo[i, j, k]
                Psi_Hz_yhi[i, j, k] = bh[j]*Psi_Hz_yhi[i, j, k] + ch[j]*(Ex[i, je-1-j, k] - Ex[i, je-2-j, k])/del_x
                Hz[i, je-2-j, k] = Hz[i, je-2-j, k] + del_t/mu0*Psi_Hz_yhi[i, j, k]
    return Psi_Hz_xlo, Psi_Hz_xhi, Psi_Hz_ylo, Psi_Hz_yhi, Hz
