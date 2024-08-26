#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:19:10 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Parâmetros:
#H0 = 67.4
#O_m0 = 0.315
O_r0 = 0

ti = 0.17
t = np.linspace(ti, 1, 1000)
z = (1/t) - 1

#M = 2.0331
#H_RG = H0*np.sqrt(O_m0*t**(-3) + 1 - O_m0 )


# Modelo f(Q) - 3
def Densidade_Q3(t, y, H0, O_m0, m):
    D_Q3 = y[0]
    dD_Q3 = y[1]

    O_m = O_m0*t**(-3)
    O_L = 1 - O_m0 - O_r0

    M = m*H0
       
    H_Q3 = H0*np.sqrt(O_m + O_L)
    dH_Q3 = -(3/2)*H0*(H0/H_Q3)*(O_m/t) # derivada de H com relação ao fator de escala
    
    Q = 6*(H_Q3)**2
    
    dfQ_3 = 1 + M*(Q**(-1/2)) / 2
    
    f1 = 3*(H0**2)*O_m
    f2 = 2*(t**2)*(H_Q3**2)*dfQ_3

    ddD_Q3 = - ((3/t) + (dH_Q3/H_Q3))*dD_Q3 + (f1/f2)*D_Q3  # equação do contraste
    return [dD_Q3, ddD_Q3]


def solD_fQ3(H0, O_m0, m):
    # Espaço de integração:
    t_span = [ti, 1]
    t = np.linspace(ti, 1, 1000)

    # Condições iniciais:
    y0 = [ti, 1]

    # Solução:
    sol_Q3 = solve_ivp(Densidade_Q3, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0, m))
    D_Q3 = sol_Q3.y[0]
    return D_Q3


plt.plot(z, solD_fQ3(70, 0.3, 2.0331), color='maroon', label='$f_3(Q)$, M = 2.0331', linewidth = 2)
plt.plot(z, solD_fQ3(70, 0.3, -2), color='peru', label='$f_3(Q)$, M = - 2', linewidth = 2)
plt.legend()
plt.xlabel('$z$')
plt.ylabel('$\delta(z)$')
#plt.savefig('delta(z)_fQ3.pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()






