#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:56:20 2021

@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


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
