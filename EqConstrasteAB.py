#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:11:05 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


# Parâmetros:

# Condições iniciais
ti = 0.2
O_r0 = 0
mu = 10**(-7)
k = 0.125

# t é o fator de escala
t = np.linspace(ti, 1, 1000)

# z é o redshift
z = 1/t - 1


# Modelo LCDM:
def Den_RG(t, y, H0, O_m0):
    D_RG  = y[0]
    dD_RG = y[1]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    ddD_RG = - ((3/t) + (dH_RG/H_RG))*dD_RG + (faux1/faux2)*O_m*D_RG
    return [dD_RG, ddD_RG]

def solD(H0, O_m0):
    # Espaço de integração:
    t_span = [ti, 1]
    t = np.linspace(ti, 1, 1000)

    # Condições iniciais:
    y0 = [ti, 1]

    # Solução:
    sol = solve_ivp(Den_RG, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D = sol.y[0]
    return D

#print(solD(70, 0.3))

# Modelo AB-f(R):
def Den_AB(t, y, H0, O_m0, b):
    H_AB   = y[0]
    dH_AB  = y[1]
    ddH_AB = y[2]
    D_AB   = y[3]
    dD_AB  = y[4]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    R_vac = 12*H0**2  # R_vac abaixo da eq 32
    e_AB = R_vac/np.log(1 + np.exp(2*b))  # eq 32
    M2 = e_AB/mu

    R_AB = 6*H_AB*(2*H_AB + t*dH_AB)
    alpha = (R_AB/e_AB) - b  # argumento do cosh

    if abs(alpha) < 25:
      s = 1/np.cosh(alpha)
    else:
      s = 0

    A = (H0**2)*(3*O_m + 4*O_r)
    B = (2*t*H_AB*dH_AB*R_AB)/(3*M2)
    C = t*H_AB*dH_AB*(np.tanh(alpha) + 1)

    c1 = 6*(t**2)*(H_AB**2)/e_AB
    c2 = ddH_AB + ((dH_AB**2)/H_AB) + (5*dH_AB/t)

    if abs(alpha) < 15:
      D = ((c1*c2*s)**2)*np.tanh(alpha)
    else:
      D = 0

    if abs(alpha) < 15:
      E = 6*(t**3)*(H_AB**3)*((1/(3*M2)) + (0.5*s*s/e_AB))
    else:
      E = 6*(t**3)*(H_AB**3)*(1/(3*M2))
    F = (11*dH_AB**2/(t*H_AB)) + (dH_AB**3/(H_AB**2)) + (6*ddH_AB/t) + (4*dH_AB*ddH_AB/H_AB)
    dddH_AB = - ((A + B + C - D)/E) - F

    fR = 0.5*(1 + np.tanh(alpha)) + R_AB/(3*M2)
    fRR= (0.5/e_AB)*s**2 + 1/(3*M2)

    R_0 = R_vac
    alpha_0 = (R_0/e_AB) - b

    fR_0 = 0.5*(1 + np.tanh(alpha_0)) + (R_0/(3*M2))
    fRR_0 = (0.5/e_AB)*((1/np.cosh(alpha_0))**2) + (1/3*M2)

    gaux3 = 9e6*(k*k/(t*t))*(fRR/fR)*(fR_0/fRR_0)
    #gaux3 = (fRR/fR)*(k/t)**2
    Geffn = (1/fR)*(1 + 4*gaux3)/(1 + 3*gaux3)

    fAB1 = 3*Geffn*H0**2
    fAB2 = 2*t*t*H_AB*H_AB

    ddD_AB = - ((3/t) + (dH_AB/H_AB))*dD_AB + (fAB1/fAB2)*O_m*D_AB
    return [dH_AB, ddH_AB, dddH_AB, dD_AB, ddD_AB]

def sol_DAB(H0, O_m0, b):
    #Integração:
    t_span = [ti, 1]
    t = np.linspace(ti, 1, 1000)

    # Condições iniciais:
    O_mi = O_m0*ti**(-3)
    O_ri = O_r0*ti**(-4)
    O_L = 1 - O_m0
    Hi = H0*np.sqrt(O_mi + O_ri + O_L)
    dHi = - ((H0**2)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
    ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
    y0 = [Hi, dHi, ddHi, ti, 1]

    # Solução:
    sol = solve_ivp(Den_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
    D_AB  = sol.y[3]
    return D_AB

#print(Sol_DAB(70, 0.3, 2))



plt.plot(z, solD(70, 0.3), color='blue', linewidth = 2, label='$\Lambda$CDM')
plt.plot(z, sol_DAB(70, 0.3, 1.6), color='black', linewidth = 2, label='$R^2$_AB model', linestyle = '--')
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta(z)$')
#plt.savefig('fsig8(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()





