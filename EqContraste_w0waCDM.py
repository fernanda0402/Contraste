#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:23:37 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp


# constantes
H0 = 67.4
O_m0 = 0.315
O_L0 = 0.7
w0 = -0.957
wa = -0.29



# Modelo wCDM:
def Densidade(t, y):    # t é o fator de escala e y é o delta (contraste)
    D_RG  = y[0]     # contraste
    dD_RG = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)  
    O_L = O_L0*(t**(-3*(1+ w0 + wa*(1-t)) ) )

    H_RG = H0*np.sqrt(O_m + O_L)    # H(z)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m - 2*O_L)        # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    ddD_RG = - ((3/t) + (dH_RG/H_RG))*dD_RG + (faux1/faux2)*O_m*D_RG   #  eq. da derivada segunda do contraste
    return [dD_RG, ddD_RG]


# Espaço de integração:
t_span = [0.17, 1]   # intervalo de integração do fator de escala
t = np.linspace(0.17, 1, 1000)

# Condições iniciais:
y0 = [0.17, 1] 


# Solução:
sol = solve_ivp(Densidade, t_span, y0, t_eval=t, method='LSODA') # função que quero resolver, o intervalo de integração, as condições iniciais, o linspace e o método
D_RG = sol.y[0]
dD_RG = sol.y[1]


# definindo o redshift
z = 1/sol.t - 1


# plotando o contraste x z
plt.plot(z, D_RG, color='red', linewidth = 2, label='$\omega_0 \omega_a$CDM')
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta$')
#plt.savefig('delta(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()



# vamos plotar a função G = dln(delta/a)/dln(a)

G = ((sol.t)*dD_RG)/D_RG - 1

#plt.figure(figsize=(10, 8))
#plt.plot(z, G, color='deeppink', linewidth = 2, label='$\omega$CDM')
#plt.xlabel('z')
#plt.ylabel('G(a)')
#plt.legend()
#plt.show()