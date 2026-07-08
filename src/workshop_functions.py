#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boiler-plate functions for fast humid heat computations
"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import numba as nb
from numba import prange
import sys 
import math
import time
from scipy.special import lambertw


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Some constants
EPS = 0.622
ONE_MINUS_EPS = 1.0 - EPS
CPD = 1005.7   # J kg-1 K-1
CPV = 1850.0   # J kg-1 K-1

@nb.njit(fastmath=True, inline="always")
def _satvp_huang_scalar(tc):
    """
    Saturation vapour pressure (Pa).

    Parameters
    ----------
    tc : float
        Temperature [°C].
    """
    if tc > 0.0:
        return np.exp(34.494 - 4924.99/(tc + 237.1)) / (tc + 105.0)**1.57
    else:
        return np.exp(43.494 - 6545.8/(tc + 278.0)) / (tc + 868.0)**2.0

@nb.njit(fastmath=True, parallel=True)
def satvp_huang(tc):
    """
    Saturation vapour pressure (Pa).

    Parameters
    ----------
    tc : ndarray
        Temperature [°C].
    """
    vp = np.empty_like(tc)

    for i in prange(tc.size):
        vp[i] = _satvp_huang_scalar(tc[i])

    return vp

@nb.njit(fastmath=True, inline="always")
def _satQ_huang_scalar(T, p):
    """
    Saturation specific humidity (kg kg-1).

    Parameters
    ----------
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].
    """
    es = _satvp_huang_scalar(T - 273.15)

    return EPS * es / (p - ONE_MINUS_EPS * es)

@nb.njit(fastmath=True, parallel=True)
def satQ_huang(T, p):
    """
    Saturation specific humidity (kg kg-1).

    Parameters
    ----------
    T : ndarray
        Temperature [K].
    p : ndarray
        Pressure [Pa].
    """
    q = np.empty_like(T)

    for i in prange(T.size):
        q[i] = _satQ_huang_scalar(T[i], p[i])

    return q


@nb.njit(fastmath=True)
def _moist_properties(T, q):
    """
    Moist thermodynamic properties from temperature and specific humidity.

    Parameters
    ----------
    T : float
        Air temperature [K].
    q : float
        Specific humidity [kg kg-1], i.e. mass vapour / mass moist air.

    Returns
    -------
    Lv : float
        Latent heat of vaporisation [J kg-1].
    Cp : float
        Specific heat capacity of moist air [J kg-1 K-1],
        per unit mass of moist air.
    """
    
    CPD = 1005.7   # J kg-1 K-1, dry air
    CPV = 1850.0   # J kg-1 K-1, water vapour
    
    Lv = 1.918e6 * (T / (T - 33.91)) ** 2
    Cp = CPD * (1.0 - q) + CPV * q

    return Lv, Cp

@nb.njit(fastmath=True, inline="always")
def _me_te_from_tq_scalar(T, q):
    Lv, Cp = _moist_properties(T, q)
    me = Cp * T + Lv * q
    te = me / Cp
    return me, te

    
@nb.njit(fastmath={"nnan": False}, parallel=True)
def me_te(T, q):
    """
    Vector moist enthalpy and equivalent temperature from T and q.

    T : ndarray
        Air temperature [K].
    q : ndarray
        Specific humidity [kg kg-1].

    Returns
    -------
    Me : ndarray
        Moist enthalpy [J kg-1].
    Te : ndarray
        Equivalent temperature [K].
    """
    Me = np.empty_like(T)
    Te = np.empty_like(T)

    for i in prange(T.size):
        if np.isnan(T[i]) or np.isnan(q[i]):
            Me[i] = np.nan
            Te[i] = np.nan
        else:
            Me[i], Te[i] = _me_te_from_tq_scalar(T[i], q[i])

    return Me, Te
    

@nb.njit(fastmath=True, inline="always")
def _lv_vap(T):
    """
    Latent heat of vaporisation [J kg-1].
    Bolton-style approximation.
    """
    return 1.918e6 * (T / (T - 33.91)) ** 2


@nb.njit(fastmath=True, inline="always")
def _cp_moist(q):
    """
    Moist-air cp [J kg-1 K-1], per kg moist air.
    q is specific humidity [kg kg-1].
    """
    return CPD * (1.0 - q) + CPV * q


import numba as nb
import numpy as np

@nb.njit(parallel=True)
def delta_te(rh, te, rh_ref, te_ref):
    # Shape
    nt, nr, nc = rh.shape

    # Flatten query arrays
    rh_query = rh.ravel()
    te_query = te.ravel()

    # Allocate output
    n = rh_query.size
    out = np.empty(n, dtype=np.float64)

    # Ensure reference RH is ascending
    if rh_ref[-1] < rh_ref[0]:
        rh_ref_use = rh_ref[::-1]
        te_ref_use = te_ref[::-1]
    else:
        rh_ref_use = rh_ref
        te_ref_use = te_ref

    # Loop over all points
    for i in nb.prange(n):
        crit_te = np.interp(
            rh_query[i],
            rh_ref_use,
            te_ref_use,
            left=te_ref_use[0],
            right=te_ref_use[-1],
        )

        out[i] = crit_te - te_query[i]

    return out.reshape(nt, nr, nc)


