#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:56:07 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams['text.usetex'] = True

# Parâmetros:
H0 = 67.4
O_m0 = 0.315
O_r0 = 0
O_k0 = -0.2


# Modelo LCDM

# Modelo LCDM:
def fgRG(t, y):
    fg_RG  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    dfg_RG = - ((dH_RG/H_RG) + ((2+fg_RG)/t))*fg_RG + (3/2)*((H0/H_RG)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_RG


# Integração:
t_span =[0.16666, 1]
t = np.linspace(0.16666, 1, 1000)


# Condições iniciais:
ti = 0.16666
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




#plt.plot(z, fg_RG, color='blue', linewidth = 2, label='$\Lambda$CDM')
#plt.legend()
#plt.xlabel('z')
#plt.ylabel('$f(z)$')
#plt.savefig('fg(z).png', dpi=520, format='png', bbox_inches='tight')
#plt.show()



########## LCDM COM Om DO LINDER E CAHN

z_LC = np.linspace(0, 5, 1000) 

a = 1 / (1 + z_LC)

O_L = 1 - O_m0 - O_k0
 
H_LC = H0*np.sqrt(O_m0*(a**(-3)) + O_L + O_k0*(a**(-2)))

Om = O_m0*(a**(-3)) / ( (H_LC/H0)**2 )

f_LC = Om**0.55


plt.plot(z, fg_RG, color='blue', linewidth = 2, label='Differential Eq.')
plt.plot(z_LC, f_LC, color='purple', label='Linder e Cahn')
plt.title('$\Omega_{k0} = - 0.2 $')
plt.legend(loc='lower right')
plt.xlabel('$z$')
plt.ylabel('$f(z)$')
#plt.savefig('f_LC_Ok=-02.png', dpi=520, format='png', bbox_inches='tight')
plt.show()











