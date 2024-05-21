#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:06:20 2024

@author: usuario
"""

# Bibliotecas:
import pandas as pd
import numpy as np
import scipy as sp
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from multiprocessing import Pool
#import emcee
import time
#from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

# Parâmetros:

O_r0 = 0
mu = 10**(-11)
k = 0.125

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1


# Modelo Hu-Sawicki:
def Den_HS(t, y, H0, O_m0, c2, n):
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    D_HS = y[2]
    dD_HS = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    O_L = 1 - O_m0

    c1 = 6*c2*(O_L/O_m0)

    R = 3*ms*(4*yH + t*YH + t**(-3))

    xn1 = (R/ms)**n
    xn2 = c1/(ms**n)
    xn3 = (n+1)*(c2/ms)*R**(2*(n-1))
    xn4 = (n-1)*R**(n-2)
    xn5 = ((c2*xn1) + 1)

    f = R - ms*((c1*xn1)/((c2*xn1)  + 1))
    fR = 1 - n*xn2*ms*((R**(n-1))/(xn5**2))
    fRR = n*xn2*ms*( (xn3 - xn4)/(xn5**3))

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)

    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)

    dYH = - J1*YH - J2*yH - J3

    H_HS = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH_HS = (R/(6*t*H_HS)) - (2*H_HS/t)

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*gaux3)/(1 + 3*gaux3)

    fAB1 = 3*Geffn*H0**2
    fAB2 = 2*t*t*H_HS*H_HS

    ddD_HS = - ((3/t) + (dH_HS/H_HS))*dD_HS + (fAB1/fAB2)*O_m*D_HS
    return [YH, dYH, dD_HS, ddD_HS]

def solD_HS(H0, O_m0, c2, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol = solve_ivp(Den_HS, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, c2, n))
    D_HS  = sol.y[2]
    return D_HS

#print(solD_HS(70,0.3,1,1)[999])



plt.plot(z, solD_HS(70,0.3,1,2), color='purple', linewidth = 2, label='Hu-Sawicki (n=2)')
plt.legend()
plt.xlabel('z')
plt.ylabel('$\delta(z)$')
#plt.savefig('fsig8(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()

