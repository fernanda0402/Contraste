#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:50:27 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parâmetros:
H0 = 67.4
O_m0 = 0.315
O_r0 = 0

t = np.linspace(0.2, 1, 1000)
z = (1/t) - 1

m2 = 20*H0**2
H_RG = H0*np.sqrt(O_m0*t**(-3) + 1 - O_m0 )
H_Q1 = np.sqrt(m2*(1 - np.sqrt( 1 - (H0**2 * O_m0 * (t**(-3))) / (2*m2) )))



# Modelo f(Q) - 1
def Densidade_Q1(t, y):
    D_Q1 = y[0]
    dD_Q1 = y[1]

    O_m = O_m0*t**(-3)
    O_L = 1 - O_m0 - O_r0

    G = 6.67*(10**(-11))
    #Mpl = 2.176*(10**(-8))
    m2 = 20*H0**2
    #rho = ( 3*(H0**2) / (8*np.pi*G) ) * (O_m + O_r + O_L)

    H_Q1 = np.sqrt( m2*(1 - np.sqrt( 1 - (H0**2 * O_m0 * (t**(-3))) / (2*m2) )) )
    dH_Q1 = - (3/2)*m2*(H0**2)*O_m0*(t**(-3)) / ((m2 - H_Q1**2)*t*H_Q1) # derivada de H com relação ao fator de escala

    Q = 6*(H_Q1)**2
    #G = 6.67*(10**(-11))
    #m2 = (H_Q1**2)*(O_m + O_r + O_L)

    #fQ1 = Q/(8*np.pi*G) - 1/(288*np.pi*G)*(Q**2/m2)  # é a f(Q)

    #rho = ( 3*(H_Q1**2) / (8*np.pi*G) ) * (O_m + O_r + O_L)

    #dfQ = 1/(8*np.pi*G) - 1/(144*np.pi*G*m2)  # derivada da f(Q)
    dfQ = 1 - (Q/(18*m2))

    aux = (t**2) * (H_Q1**2)
    ddD_Q1 = - ((3/t) + dH_Q1/H_Q1)*dD_Q1 + ( (3/2)*(1/aux)*(H0**2)*O_m*D_Q1 / (2*(1+ dfQ)) )  # equação do contraste
    return [dD_Q1, ddD_Q1]




t_span = [0.2, 1]
# Condições iniciais:
y0 = [0.2, 1]



# Solução2
sol_Q1 = solve_ivp(Densidade_Q1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6))
D_Q1 = sol_Q1.y[0]



#print(D_Q1)

#plt.plot(z, H_Q1, color='red', linewidth = 2, label='$f(Q)$')
#plt.plot(z, H_RG, color='blue', linewidth = 2, label='$\Lambda$CDM')
plt.plot(z, D_Q1, color='saddlebrown', label='$f_1(Q)$', linewidth = 2)
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta(z)$')
#plt.savefig('delta(z)_fQ1.png', dpi=520, format='png', bbox_inches='tight')
plt.show()








