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
from scipy.optimize import fsolve
from pyspeckle import autocorrelation
import os
import user_config as usr


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
    A = autocorrelation(dist.astype(usr.precision))
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


def dict_from_module(module):
    context = {}
    for item in dir(module):
        it = type(getattr(module, item))
        if it is float or it is str or it is int or it is type or it is bool:
            if item[0] != '_':
                context[item] = getattr(module, item)
    return context
