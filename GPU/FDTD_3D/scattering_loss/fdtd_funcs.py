"""
@author: Brian Guiana

Acknowledgement
This project was completed as part of research conducted with my major professor and advisor, Prof. Ata Zadehgol, as part of the Applied and Computational Electromagnetics Signal and Power Integrity (ACEM-SPI) Lab while working toward the Ph.D. in Electrical Engineering at the University of Idaho, Moscow, Idaho, USA. This project was funded, in part, by the National Science Foundation (NSF); award #1816542 [1].

"""

import numpy as np
from numba import cuda, njit, prange, f4, i4, void
from aux_funcs import mkdir, check_discretization


#                          Ex       , Hy       , Hz       , kye  , kze  , cb       , ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], i4, i4, i4, i4, i4, i4))
def update_ex(Ex, Hy, Hz, kye, kze, cb, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i > ib and i < ie:
        if j >= jb and j < je:
            if k >= kb and k < ke:
                Ex[i, j, k] = Ex[i, j, k] + cb[i, j, k] * ( ( Hz[i, j+1, k] - Hz[i, j, k] ) / kye[j] - ( Hy[i, j, k+1] - Hy[i, j, k] ) / kze[k] )


#                          Ey       , Hx       , Hz       , kxe  , kze  , cb       , ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], i4, i4, i4, i4, i4, i4))
def update_ey(Ey, Hx, Hz, kxe, kze, cb, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i >= ib and i < ie:
        if j > jb and j < je:
            if k >= kb and k < ke:
                Ey[i, j, k] = Ey[i, j, k] + cb[i, j, k] * ( ( Hx[i, j, k+1] - Hx[i, j, k] ) / kze[k] - ( Hz[i+1, j, k] - Hz[i, j, k] ) / kxe[i] )


#                          Ez       , Hx       , Hy       , kxe  , kye  , cb       , ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], i4, i4, i4, i4, i4, i4))
def update_ez(Ez, Hx, Hy, kxe, kye, cb, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i >= ib and i < ie:
        if j >= jb and j < je:
            if k > kb and k < ke:
                Ez[i, j, k] = Ez[i, j, k] + cb[i, j, k] * ( ( Hy[i+1, j, k] - Hy[i, j, k] ) / kxe[i] - ( Hx[i, j+1, k] - Hx[i, j, k] ) / kye[j] )


#                          Ey       , Ez       , Hx       , kyh  , kzh  , db, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, i4, i4, i4, i4, i4, i4))
def update_hx(Ey, Ez, Hx, kyh, kzh, db, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i >= ib and i < ie:
        if j > jb and j < je:
            if k > kb and k < ke:
                Hx[i, j, k] = Hx[i, j, k] + db * ( ( Ey[i, j, k] - Ey[i, j, k-1] ) / kzh[k] - ( Ez[i, j, k] - Ez[i, j-1, k] ) / kyh[j] )


#                          Ex       , Ez       , Hy       , kxh  , kzh  , db, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, i4, i4, i4, i4, i4, i4))
def update_hy(Ex, Ez, Hy, kxh, kzh, db, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i > ib and i < ie:
        if j >= jb and j < je:
            if k > kb and k < ke:
                Hy[i, j, k] = Hy[i, j, k] + db * ( ( Ez[i, j, k] - Ez[i-1, j, k] ) / kxh[i] - ( Ex[i, j, k] - Ex[i, j, k-1] ) / kzh[k] )


#                          Ex       , Ey       , Hz       , kxh  , kyh  , db, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, i4, i4, i4, i4, i4, i4))
def update_hz(Ex, Ey, Hz, kxh, kyh, db, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if i > ib and i < ie:
        if j > jb and j < je:
            if k >= kb and k < ke:
                Hz[i, j, k] = Hz[i, j, k] + db * ( ( Ex[i, j, k] - Ex[i, j-1, k] ) / kyh[j] - ( Ey[i, j, k] - Ey[i-1, j, k] ) / kxh[i] )


#                          Field    , Src  , sx, sy, sz, sx, sy, sz, step
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:], i4, i4, i4, i4, i4, i4, i4))
def update_source(Field, Source, sxb, syb, szb, sxe, sye, sze, step):
    i, j, k = cuda.grid(3)
    if i >= sxb and i < sxe:
        if j >= syb and j < sye:
            if k >= szb and k < sze:
                Field[i, j, k] = Field[i, j, k] - Source[step]


#                          PEx_yl   , PEx_yh   , Ex       , Hz       , be   , ce   , cb       , dy, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ex_yinc(Psi_Ex_ylo, Psi_Ex_yhi, Ex, Hz, be, ce, cb, dy, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k >= kb and k < ke and j < pmle and i > ib and i < ie:
        Psi_Ex_ylo[i, j, k] = be[-(j+1)] * Psi_Ex_ylo[i, j, k] + ce[-(j+1)] * ( Hz[i, jb+j+1, k] - Hz[i, jb+j, k] ) / dy
        Ex[i, jb+j, k] = Ex[i, jb+j, k] + cb[i, jb+j, k] * Psi_Ex_ylo[i, j, k]
        Psi_Ex_yhi[i, j, k] = be[j] * Psi_Ex_yhi[i, j, k] + ce[j] * ( Hz[i, je-pmle+j+1, k] - Hz[i, je-pmle+j, k] ) / dy
        Ex[i, je-pmle+j, k] = Ex[i, je-pmle+j, k] + cb[i, je-pmle+j, k] * Psi_Ex_yhi[i, j, k]


#                          PEx_zl   , PEx_zh   , Ex       , Hy       , be   , ce   , cb       , dz, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ex_zinc(Psi_Ex_zlo, Psi_Ex_zhi, Ex, Hy, be, ce, cb, dz, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k < pmle and j >= jb and j < je and i > ib and i < ie:
        Psi_Ex_zlo[i, j, k] = be[-(k+1)] * Psi_Ex_zlo[i, j, k] + ce[-(k+1)] * ( Hy[i, j, kb+k+1] - Hy[i, j, kb+k] ) / dz
        Ex[i, j, kb+k] = Ex[i, j, kb+k] - cb[i, j, kb+k] * Psi_Ex_zlo[i, j, k]
        Psi_Ex_zhi[i, j, k] = be[k] * Psi_Ex_zhi[i, j, k] + ce[k] * ( Hy[i, j, ke-pmle+k+1] - Hy[i, j, ke-pmle+k] ) / dz
        Ex[i, j, ke-pmle+k] = Ex[i, j, ke-pmle+k] - cb[i, j, ke-pmle+k] * Psi_Ex_zhi[i, j, k]


#                          PEy_xl   , PEy_xh   , Ey       , Hz       , be   , ce   , cb       , dx, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ey_xinc(Psi_Ey_xlo, Psi_Ey_xhi, Ey, Hz, be, ce, cb, dx, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k >= kb and k < ke and j > jb and j < je and i < pmle:
        Psi_Ey_xlo[i, j, k] = be[-(i+1)] * Psi_Ey_xlo[i, j, k] + ce[-(i+1)] * ( Hz[ib+i+1, j, k] - Hz[ib+i, j, k] ) / dx
        Ey[ib+i, j, k] = Ey[ib+i, j, k] - cb[ib+i, j, k] * Psi_Ey_xlo[i, j, k]
        Psi_Ey_xhi[i, j, k] = be[i] * Psi_Ey_xhi[i, j, k] + ce[i] * ( Hz[ie-pmle+i+1, j, k] - Hz[ie-pmle+i, j, k] ) / dx
        Ey[ie-pmle+i, j, k] = Ey[ie-pmle+i, j, k] - cb[ie-pmle+i, j, k] * Psi_Ey_xhi[i, j, k]


#                          PEy_zl   , PEy_zh   , Ey       , Hx       , be   , ce   , cb       , dz, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ey_zinc(Psi_Ey_zlo, Psi_Ey_zhi, Ey, Hx, be, ce, cb, dz, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k < pmle and j > jb and j < je and i >= ib and i < ie:
        Psi_Ey_zlo[i, j, k] = be[-(k+1)] * Psi_Ey_zlo[i, j, k] + ce[-(k+1)] * ( Hx[i, j, kb+k+1] - Hx[i, j, kb+k] ) / dz
        Ey[i, j, kb+k] = Ey[i, j, kb+k] + cb[i, j, kb+k] * Psi_Ey_zlo[i, j, k]
        Psi_Ey_zhi[i, j, k] = be[k] * Psi_Ey_zhi[i, j, k] + ce[k] * ( Hx[i, j, ke-pmle+k+1] - Hx[i, j, ke-pmle+k] ) / dz
        Ey[i, j, ke-pmle+k] = Ey[i, j, ke-pmle+k] + cb[i, j, ke-pmle+k] * Psi_Ey_zhi[i, j, k]


#                          PEz_xl   , PEz_xh   , Ez       , Hy       , be   , ce   , cb       , dx, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ez_xinc(Psi_Ez_xlo, Psi_Ez_xhi, Ez, Hy, be, ce, cb, dx, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > kb and k < ke and j >= jb and j < je and i < pmle:
        Psi_Ez_xlo[i, j, k] = be[-(i+1)] * Psi_Ez_xlo[i, j, k] + ce[-(i+1)] * ( Hy[ib+i+1, j, k] - Hy[ib+i, j, k] ) / dx
        Ez[ib+i, j, k] = Ez[ib+i, j, k] + cb[ib+i, j, k] * Psi_Ez_xlo[i, j, k]
        Psi_Ez_xhi[i, j, k] = be[i] * Psi_Ez_xhi[i, j, k] + ce[i] * ( Hy[ie-pmle+i+1, j, k] - Hy[ie-pmle+i, j, k] ) / dx
        Ez[ie-pmle+i, j, k] = Ez[ie-pmle+i, j, k] + cb[ie-pmle+i, j, k] * Psi_Ez_xhi[i, j, k]


#                          PEz_yl   , PEz_yh   , Ez       , Hx       , be   , ce   , cb       , dy, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4[:,:,:], f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_ez_yinc(Psi_Ez_ylo, Psi_Ez_yhi, Ez, Hx, be, ce, cb, dy, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > kb and k < ke and j < pmle and i >= ib and i < ie:
        Psi_Ez_ylo[i, j, k] = be[-(j+1)] * Psi_Ez_ylo[i, j, k] + ce[-(j+1)] * ( Hx[i, jb+j+1, k] - Hx[i, jb+j, k] ) / dy
        Ez[i, jb+j, k] = Ez[i, jb+j, k] - cb[i, jb+j, k] * Psi_Ez_ylo[i, j, k]
        Psi_Ez_yhi[i, j, k] = be[j] * Psi_Ez_yhi[i, j, k] + ce[j] * ( Hx[i, je-pmle+j+1, k] - Hx[i, je-pmle+j, k] ) / dy
        Ez[i, je-pmle+j, k] = Ez[i, je-pmle+j, k] - cb[i, je-pmle+j, k] * Psi_Ez_yhi[i, j, k]


#                          PHx_yl   , PHx_yh   , Ez       , Hx       , bh   , ch   , db, dy, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hx_yinc(Psi_Hx_ylo, Psi_Hx_yhi, Ez, Hx, bh, ch, db, dy, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > kb and k < ke and j > 0 and j < pmle and i >= ib and i < ie:
        Psi_Hx_ylo[i, j, k] = bh[-j] * Psi_Hx_ylo[i, j, k] + ch[-j] * ( Ez[i, jb+j, k] - Ez[i, jb+j-1, k] ) / dy
        Hx[i, jb+j, k] = Hx[i, jb+j, k] - db * Psi_Hx_ylo[i, j, k]
        Psi_Hx_yhi[i, j, k] = bh[j] * Psi_Hx_yhi[i, j, k] + ch[j] * ( Ez[i, je-pmle+j, k] - Ez[i, je-pmle+j-1, k] ) / dy
        Hx[i, je-pmle+j, k] = Hx[i, je-pmle+j, k] - db * Psi_Hx_yhi[i, j, k]


#                          PHx_zl   , PHx_zh   , Ey       , Hx       , bh   , ch   , db, dz, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hx_zinc(Psi_Hx_zlo, Psi_Hx_zhi, Ey, Hx, bh, ch, db, dz, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > 0 and k < pmle and j > jb and j < je and i >= ib and i < ie:
        Psi_Hx_zlo[i, j, k] = bh[-k] * Psi_Hx_zlo[i, j, k] + ch[-k] * ( Ey[i, j, kb+k] - Ey[i, j, kb+k-1] ) / dz
        Hx[i, j, kb+k] = Hx[i, j, kb+k] + db * Psi_Hx_zlo[i, j, k]
        Psi_Hx_zhi[i, j, k] = bh[k] * Psi_Hx_zhi[i, j, k] + ch[k] * ( Ey[i, j, ke-pmle+k] - Ey[i, j, ke-pmle+k-1] ) / dz
        Hx[i, j, ke-pmle+k] = Hx[i, j, ke-pmle+k] + db * Psi_Hx_zhi[i, j, k]


#                          PHy_xl   , PHy_xh   , Ez       , Hy       , bh   , ch   , db, dx, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hy_xinc(Psi_Hy_xlo, Psi_Hy_xhi, Ez, Hy, bh, ch, db, dx, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > kb and k < ke and j >= jb and j < je and i > 0 and i < pmle:
        Psi_Hy_xlo[i, j, k] = bh[-i] * Psi_Hy_xlo[i, j, k] + ch[-i] * ( Ez[ib+i, j, k] - Ez[ib+i-1, j, k] ) / dx
        Hy[ib+i, j, k] = Hy[ib+i, j, k] + db * Psi_Hy_xlo[i, j, k]
        Psi_Hy_xhi[i, j, k] = bh[i] * Psi_Hy_xhi[i, j, k] + ch[i] * ( Ez[ie-pmle+i, j, k] - Ez[ie-pmle+i-1, j, k] ) / dx
        Hy[ie-pmle+i, j, k] = Hy[ie-pmle+i, j, k] + db * Psi_Hy_xhi[i, j, k]


#                          PHy_zl   , PHy_zh   , Ex       , Hy       , bh   , ch   , db, dz, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hy_zinc(Psi_Hy_zlo, Psi_Hy_zhi, Ex, Hy, bh, ch, db, dz, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k > 0 and k < pmle and j >= jb and j < je and i > ib and i < ie:
        Psi_Hy_zlo[i, j, k] = bh[-k] * Psi_Hy_zlo[i, j, k] + ch[-k] * ( Ex[i, j, kb+k] - Ex[i, j, kb+k-1] ) / dz
        Hy[i, j, kb+k] = Hy[i, j, kb+k] - db * Psi_Hy_zlo[i, j, k]
        Psi_Hy_zhi[i, j, k] = bh[k] * Psi_Hy_zhi[i, j, k] + ch[k] * ( Ex[i, j, ke-pmle+k] - Ex[i, j, ke-pmle+k-1] ) / dz
        Hy[i, j, ke-pmle+k] = Hy[i, j, ke-pmle+k] - db * Psi_Hy_zhi[i, j, k]


#                          PHz_xl   , PHz_xh   , Ey       , Hz       , bh   , ch   , db, dx, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hz_xinc(Psi_Hz_xlo, Psi_Hz_xhi, Ey, Hz, bh, ch, db, dx, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k >= kb and k < ke and j > jb and j < je and i > 0 and i < pmle:
        Psi_Hz_xlo[i, j, k] = bh[-i] * Psi_Hz_xlo[i, j, k] + ch[-i] * ( Ey[ib+i, j, k] - Ey[ib+i-1, j, k] ) / dx
        Hz[ib+i, j, k] = Hz[ib+i, j, k] - db * Psi_Hz_xlo[i, j, k]
        Psi_Hz_xhi[i, j, k] = bh[i] * Psi_Hz_xhi[i, j, k] + ch[i] * ( Ey[ie-pmle+i, j, k] - Ey[ie-pmle+i-1, j, k] ) / dx
        Hz[ie-pmle+i, j, k] = Hz[ie-pmle+i, j, k] - db * Psi_Hz_xhi[i, j, k]


#                          PHz_yl   , PHz_yh   , Ex       , Hz       , bh   , ch   , db, dy, pe, ib, jb, kb, ie, je, ke
@cuda.jit(func_or_sig=void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:], f4[:], f4, f4, i4, i4, i4, i4, i4, i4, i4))
def update_pml_hz_yinc(Psi_Hz_ylo, Psi_Hz_yhi, Ex, Hz, bh, ch, db, dy, pmle, ib, jb, kb, ie, je, ke):
    i, j, k = cuda.grid(3)
    if k >= kb and k < ke and j > 0 and j < pmle and i > ib and i < ie:
        Psi_Hz_ylo[i, j, k] = bh[-j] * Psi_Hz_ylo[i, j, k] + ch[-j] * ( Ex[i, jb+j, k] - Ex[i, jb+j-1, k] ) / dy
        Hz[i, jb+j, k] = Hz[i, jb+j, k] + db * Psi_Hz_ylo[i, j, k]
        Psi_Hz_yhi[i, j, k] = bh[j] * Psi_Hz_yhi[i, j, k] + ch[j] * ( Ex[i, je-pmle+j, k] - Ex[i, je-pmle+j-1, k] ) / dy
        Hz[i, je-pmle+j, k] = Hz[i, je-pmle+j, k] + db * Psi_Hz_yhi[i, j, k]


#                          Rx     , Ry     , Rz     , Ix     , Iy     , Iz     , Ex       , Ey       , Ez       , cs, sn, ib, jb, ie, je, ct
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4, f4, i4, i4, i4, i4, i4))
def simul_fft_efield(Rx, Ry, Rz, Ix, Iy, Iz, Ex, Ey, Ez, cos, sin, ib, jb, ie, je, cut):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rx[i, j] = Rx[i, j] + 0.5 * (Ex[i, j, cut] + Ex[i+1, j, cut]) * cos
            Ix[i, j] = Ix[i, j] + 0.5 * (Ex[i, j, cut] + Ex[i+1, j, cut]) * sin
            Ry[i, j] = Ry[i, j] + 0.5 * (Ey[i, j, cut] + Ey[i, j+1, cut]) * cos
            Iy[i, j] = Iy[i, j] + 0.5 * (Ey[i, j, cut] + Ey[i, j+1, cut]) * sin
            Rz[i, j] = Rz[i, j] + 0.5 * (Ez[i, j, cut] + Ez[i, j, cut+1]) * cos
            Iz[i, j] = Iz[i, j] + 0.5 * (Ez[i, j, cut] + Ez[i, j, cut+1]) * sin


#                          Rx     , Ry     , Rz     , Ix     , Iy     , Iz     , Hx       , Hy       , Hz       , cs, sn, ib, jb, ie, je, ct
@cuda.jit(func_or_sig=void(f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4, f4, i4, i4, i4, i4, i4))
def simul_fft_hfield(Rx, Ry, Rz, Ix, Iy, Iz, Hx, Hy, Hz, cos, sin, ib, jb, ie, je, cut):
    i, j = cuda.grid(2)
    if ib <= i < ie:
        if jb <= j < je:
            Rx[i, j] = Rx[i, j] + 0.25 * (Hx[i, j, cut] + Hx[i, j+1, cut] + Hx[i, j, cut+1] + Hx[i, j+1, cut+1]) * cos
            Ix[i, j] = Ix[i, j] + 0.25 * (Hx[i, j, cut] + Hx[i, j+1, cut] + Hx[i, j, cut+1] + Hx[i, j+1, cut+1]) * sin
            Ry[i, j] = Ry[i, j] + 0.25 * (Hy[i, j, cut] + Hy[i+1, j, cut] + Hy[i, j, cut+1] + Hy[i+1, j, cut+1]) * cos
            Iy[i, j] = Iy[i, j] + 0.25 * (Hy[i, j, cut] + Hy[i+1, j, cut] + Hy[i, j, cut+1] + Hy[i+1, j, cut+1]) * sin
            Rz[i, j] = Rz[i, j] + 0.25 * (Hz[i, j, cut] + Hz[i+1, j, cut] + Hz[i, j+1, cut] + Hz[i+1, j+1, cut]) * cos
            Iz[i, j] = Iz[i, j] + 0.25 * (Hz[i, j, cut] + Hz[i+1, j, cut] + Hz[i, j+1, cut] + Hz[i+1, j+1, cut]) * sin


@njit(parallel=True)
def average_materials(Mask, X, Y, Z, ie, je, ke):
    for k in prange(1, ke):
        for j in prange(1, je):
            for i in prange(1, ie):
                X[i, j, k] = 0.5 * (Mask[i, j, k] + Mask[i-1, j, k])
                Y[i, j, k] = 0.5 * (Mask[i, j, k] + Mask[i, j-1, k])
                Z[i, j, k] = 0.5 * (Mask[i, j, k] + Mask[i, j, k-1])


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


def generate_fg_mask(num_x, num_y, num_z, bord_x, bord_y, wg_height, wg_width, settling_range, end_range, rough_std, rough_acl, dx, mode='gen', correlation=3, upper_name='', lower_name='', atol=0.1, stol=0.1, mtol=0.01):
    fg_mask = np.zeros([num_x, num_y, num_z], dtype=bool)
    num_z_eff = num_z - settling_range - end_range
    if mode == 'gen':
        mkdir('rough_profiles')
        print('Searching for upper profile')
        var_wrt_len_upper = make_profile(num_z_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        if correlation == 1:
            var_wrt_len_lower = var_wrt_len_upper.copy()
        elif correlation == 2:
            var_wrt_len_lower = -1*var_wrt_len_upper
        elif correlation == 3:
            print('Searching for lower profile')
            var_wrt_len_lower = make_profile(num_z_eff, rough_std, rough_acl, dx, stol, atol, mtol)
        else:
            raise Exception('Invalid correlation type selected')
    elif mode =='load':
        print('roughness not ready yet, making smooth mask')
        var_wrt_len_upper = np.load(upper_name)
        var_wrt_len_lower = np.load(lower_name)
    elif mode == 'smooth':
        var_wrt_len_upper = np.zeros(num_z_eff)
        var_wrt_len_lower = np.zeros(num_z_eff)
    else:
        raise Exception('Choose a valid roughness profile mode')

    mkdir('rough_profiles')
    np.save(upper_name, var_wrt_len_upper)
    np.save(lower_name, var_wrt_len_lower)

    var_wrt_len_upper = np.append(np.zeros(settling_range), var_wrt_len_upper)
    var_wrt_len_upper = np.append(var_wrt_len_upper, np.zeros(end_range))
    var_wrt_len_lower = np.append(np.zeros(settling_range), var_wrt_len_lower)
    var_wrt_len_lower = np.append(var_wrt_len_lower, np.zeros(end_range))

    apply_mask(fg_mask, num_x, num_y, num_z, bord_x, bord_y, wg_height, wg_width, var_wrt_len_lower, var_wrt_len_upper)
    return fg_mask


@njit(parallel=True)
def apply_mask(fg_mask, num_x, num_y, num_z, bord_x, bord_y, wg_height, wg_width, var_wrt_len_lower, var_wrt_len_upper):
    for k in prange(num_z):
        for j in prange(num_y):
            for i in prange(num_x):
                if j >= (bord_y) and j < (bord_y + wg_width) and i >= (bord_x + var_wrt_len_lower[k]) and i < (bord_x + wg_height + var_wrt_len_upper[k]):
                # if i >= (bord_x + var_wrt_len_lower[k]) and i < (bord_x + wg_width + var_wrt_len_upper[k]):
                    fg_mask[i, j, k] = True


#              Exz      , Eyz      , Ezz      , Ex       , Ey       , Ez       , zs, ct, ie, je
@cuda.jit(void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], i4, i4, i4, i4))
def map_efield_zwave(Exz, Eyz, Ezz, Ex, Ey, Ez, zwstep, cut, ie, je):
    i, j, k = cuda.grid(3)
    if i < ie:
        if j < je:
            if k == cut:
                Exz[zwstep, i, j] = 0.5 * (Ex[i, j, k] + Ex[i+1, j, k])
                Eyz[zwstep, i, j] = 0.5 * (Ey[i, j, k] + Ey[i, j+1, k])
                Ezz[zwstep, i, j] = 0.5 * (Ez[i, j, k] + Ez[i, j, k+1])


#              Hxz      , Hyz      , Hzz      , Hx       , Hy       , Hz       , zs, ct, ie, je
@cuda.jit(void(f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], f4[:,:,:], i4, i4, i4, i4))
def map_hfield_zwave(Hxz, Hyz, Hzz, Hx, Hy, Hz, zwstep, cut, ie, je):
    i, j, k = cuda.grid(3)
    if i < ie:
        if j < je:
            if k == cut:
                Hxz[zwstep, i, j] = 0.25 * (Hx[i, j, k] + Hx[i, j+1, k] + Hx[i, j, k+1] + Hx[i, j+1, k+1])
                Hyz[zwstep, i, j] = 0.25 * (Hy[i, j, k] + Hy[i+1, j, k] + Hy[i, j, k+1] + Hy[i+1, j, k+1])
                Hzz[zwstep, i, j] = 0.25 * (Hz[i, j, k] + Hz[i+1, j, k] + Hz[i, j+1, k] + Hz[i+1, j+1, k])
