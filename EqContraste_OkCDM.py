#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:22:24 2024

@author: usuario
"""


# Bibliotecas:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp


# constantes
H0 = 67.4
O_m0 = 0.3485
O_k0 = -0.011

# Modelo O_k-CDM:
def Densidade_k(t, y):    # t é o fator de escala e y é o delta (contraste)
    D_RG_k  = y[0]     # contraste
    dD_RG_k = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)  
    O_k = O_k0*t**(-2)
    O_L = 1 - O_m0 - O_k0

    H_RG_k = H0*np.sqrt(O_m + O_L + O_k)    # H(z)
    dH_RG_k = - (H_RG_k/t) - 0.5*(H0/t)*(H0/H_RG_k)*(O_m - 2*O_L)       # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_k**2)

    ddD_RG_k = - ((3/t) + (dH_RG_k/H_RG_k))*dD_RG_k + (faux1/faux2)*O_m*D_RG_k   #  eq. da derivada segunda do contraste
    return [dD_RG_k, ddD_RG_k]


# Espaço de integração:
t_span = [0.17, 1]   # intervalo de integração do fator de escala
t = np.linspace(0.17, 1, 1000)

# Condições iniciais:
y0 = [0.17, 1] 


# Solução:
sol = solve_ivp(Densidade_k, t_span, y0, t_eval=t, method='LSODA') # função que quero resolver, o intervalo de integração, as condições iniciais, o linspace e o método
D_RG_k = sol.y[0]
dD_RG_k = sol.y[1]


# definindo o redshift
z = 1/sol.t - 1


# plotando o contraste x z
plt.plot(z, D_RG_k, color='green', linewidth = 2, label='$\Omega_k$-CDM')
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta$')
#plt.savefig('delta(z)_OkLCDM.png', dpi=520, format='png', bbox_inches='tight')
plt.show()







