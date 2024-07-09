#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:32:43 2024

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
O_L0 = 0.7
w0 = -0.957
wa = -0.29


# Modelo w0waCDM:
def fgw0wa(t, y):
    fg_w0wa  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = O_L0*(t**(-3*(1+w0 + wa*(1-t)) ) )

    H_RG_w = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG_w = - (H_RG_w/t) - 0.5*(H0/t)*(H0/H_RG_w)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w**2)

    df_w = - ((dH_RG_w/H_RG_w) + ((2+fg_w0wa)/t))*fg_w0wa + (3/2)*((H0/H_RG_w)**2)*(O_m/t)  # equação diferencial para o f
    return df_w


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
sol_w = solve_ivp(fgw0wa, t_span, y0, t_eval=t, method='LSODA')
fg_w0wa = sol_w.y[0]


# definindo o redshift
z = 1/sol_w.t - 1




plt.plot(z, fg_w0wa, color='red', linewidth = 2, label='$\omega_0 \omega_a$CDM')
plt.legend()
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('fg(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()