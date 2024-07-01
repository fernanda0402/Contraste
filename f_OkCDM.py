#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:49:20 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parâmetros:
H0 = 67.4
O_m0 = 0.315
O_r0 = 0
O_k0 = 0.0007          # r = 8 Mpc/h


# Modelo Ok-CDN:
def fg_k(t, y):
    fg_k  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_k = O_k0*t**(-2)
    O_L = 1 - O_m0 - O_k0

    H_RG_k = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG_k = - (H_RG_k/t) - 0.5*(H0/t)*(H0/H_RG_k)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_k**2)

    dfg_k = - ((dH_RG_k/H_RG_k) + ((2+fg_k)/t))*fg_k + (3/2)*((H0/H_RG_k)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_k


# Integração:
t_span =[0.17, 1]
t = np.linspace(0.17, 1, 1000)


# Condições iniciais:
ti = 0.17
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
fgi = ((H0/Hi)**2 * O_mi)**(6/11)
y0 = [fgi]


# Solução RG:
sol_k = solve_ivp(fg_k, t_span, y0, t_eval=t, method='LSODA')
fg_k = sol_k.y[0]


# definindo o redshift
z = 1/sol_k.t - 1




plt.plot(z, fg_k, color='green', linewidth = 2, label='$\Omega_k$-CDM')
plt.legend()
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('fg(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()
