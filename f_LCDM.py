#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:00:34 2024

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
b = 2
mu = 10**(-7)
k = 0.125              # r = 8 Mpc/h


# Modelo LCDM

# Modelo LCDM:
def fgRG(t, y):
    fg_RG  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    dfg_RG = - ((dH_RG/H_RG) + ((2+fg_RG)/t))*fg_RG + (3/2)*((H0/H_RG)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_RG


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
solRG = solve_ivp(fgRG, t_span, y0, t_eval=t, method='LSODA')
fg_RG = solRG.y[0]


# definindo o redshift
z = 1/solRG.t - 1




plt.plot(z, fg_RG, color='blue', linewidth = 2, label='$\Lambda$CDM')
plt.legend()
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('fg(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()
