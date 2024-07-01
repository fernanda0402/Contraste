#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:51:42 2024

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

M4 = 20*H0**2
H_RG = H0*np.sqrt(O_m0*t**(-3) + 1 - O_m0 )



# Modelo f(Q) - 2
def Densidade_Q2(t, y):
    D_Q2 = y[0]
    dD_Q2 = y[1]

    O_m = O_m0*t**(-3)
    O_L = 1 - O_m0 - O_r0

    G = 6.67*(10**(-11))
    #Mpl = 2.176*(10**(-8))
    M4 = 20*H0**2
    #rho = ( 3*(H0**2) / (8*np.pi*G) ) * (O_m + O_r + O_L)

    H_Q2 = np.sqrt( ( (H0**2)*O_m / 2 )*(1 + np.sqrt(1 + 3*M4/( 9*(H0**2)*(O_m**2) ) ) ) )
    dH_Q2 = (- 18*O_m*(H0**2)*(H_Q2**2) / ( t*(M4 + 12*(H_Q2**4)) )  ) # derivada de H com relação ao fator de escala

    Q = 6*(H_Q2)**2
    #G = 6.67*(10**(-11))
    #m2 = (H_Q1**2)*(O_m + O_r + O_L)

    #fQ1 = Q/(8*np.pi*G) - 1/(288*np.pi*G)*(Q**2/m2)  # é a f(Q)

    #rho = ( 3*(H_Q1**2) / (8*np.pi*G) ) * (O_m + O_r + O_L)

    #dfQ = 1/(8*np.pi*G) - 1/(144*np.pi*G*m2)  # derivada da f(Q)
    dfQ_2 = 1 - (M4/Q**2)

    aux = (t**2) * (H_Q2**2)
    ddD_Q2 = - ((3/t) + dH_Q2/H_Q2)*dD_Q2 + ( (3/2)*(1/aux)*(H0**2)*O_m*D_Q2 / (2*(1+ dfQ_2)) )  # equação do contraste
    return [dD_Q2, ddD_Q2]




t_span = [0.2, 1]
# Condições iniciais:
y0 = [0.2, 1]



# Solução2
sol_Q2 = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6))
D_Q2 = sol_Q2.y[0]



#print(D_Q1)

#plt.plot(z, H_Q1, color='red', linewidth = 2, label='$f(Q)$')
#plt.plot(z, H_RG, color='blue', linewidth = 2, label='$\Lambda$CDM')
plt.plot(z, D_Q2, color='slategray', label='$f_2(Q)$', linewidth = 2)
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta(z)$')
#plt.savefig('delta(z)_fQ2.png', dpi=520, format='png', bbox_inches='tight')
plt.show()

