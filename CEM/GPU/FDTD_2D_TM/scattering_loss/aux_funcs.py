"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from pyspeckle import autocorrelation
import os


def find_neff(n1, n2, d, k0, mode='te'):
    if mode == 'te':
        def func(neff, n1, n2, d, k0):
            bte = (neff**2 - n2**2)/(n1**2-n2**2)
            V = k0*2*d*np.sqrt(n1**2-n2**2)
            f = 2*np.arctan(np.sqrt(bte/(1-bte))) - V*np.sqrt(1-bte)
            return f
    elif mode == 'tm':
        def func(neff, n1, n2, d, k0):
            qs = neff**2/n1**2 + neff**2/n2**2 - 1
            btm = n1**2 * (neff**2 - n2**2)/(n2**2 * qs * (n1**2 - n2**2))
            V = k0*2*d*np.sqrt(n1**2 - n2**2)
            f = 2*np.arctan(np.sqrt(btm/(1-btm))) - V*(n1/n2)*np.sqrt(qs * (1 - btm))
            return f
    neff = fsolve(func, x0=n2, args=(n1, n2, d, k0))
    return neff[0]


def check_discretization(test_array, dx):
    dist = (test_array).astype(int)
    sigma = np.std(dist) * dx
    A = autocorrelation(dist.astype(float))
    f = A - 1 / np.e
    for i in range(len(f)):
        if f[i] < 0:
            break
    acl = f[i-1]/(f[i-1]-f[i]) + i - 1
    acl *= dx
    mm = np.mean(dist)
    return sigma, acl, mm


def mkdir(mypath):
    if os.path.isdir(mypath):
        pass
    else:
        os.makedirs(mypath)
    return


def get_scattering_loss_e2p_fd(A_fd, B_fd, n1, n2, f, dt, dx, d, ell, buffer_size):
    """
    alpha is returned in Np/m
    """
    mu = 4e-7*np.pi
    k0 = 2*np.pi*f/3e8
    neff = find_neff(n1, n2, d, k0, mode='te')
    beta = neff * k0

    start = buffer_size
    stop = len(A_fd) - start + 1
    P1 = 0
    P2 = 0
    for j in range(start, stop):
        P1 += 0.5 * (abs(A_fd[j])**2 * beta / (2*np.pi*f*mu)) * dx
        P2 += 0.5 * (abs(B_fd[j])**2 * beta / (2*np.pi*f*mu)) * dx

    alpha = -np.log(P2/P1)/ell
    return alpha
