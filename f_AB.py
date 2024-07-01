#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:25:00 2024

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




# Modelo AB-f(R):
def fgAB(t, y):
    H = y[0]
    dH = y[1]
    ddH = y[2]
    fg = y[3]

    O_m = O_m0*t**(-3)                           # densidade do material
    O_r = O_r0*t**(-4)

    R_vac = 12*H0**2                             # constantes
    e_AB = R_vac/np.log(1 + np.exp(2*b))
    M2 = e_AB/mu
    M3 = 1/(3*M2)

    R = 6*H*(2*H + t*dH)                          # escalar de curvatura
    alpha = (R/e_AB - b)

    tanh = np.tanh(alpha)                         # tangente hiperbólica

    if abs(alpha) < 25:                           # secante hiperbólica, no outro código é representada por s
        sech = 1/np.cosh(alpha)
    else:
        sech = 0

    A = (3*O_m + 4*O_r)*H0*H0                     # funções auxiliares
    B = 2*M3*t*H*dH*R
    C = (tanh + 1)*t*H*dH

    g1 = (6/e_AB)*t*t*H*H
    g2 = ddH + (1/H)*dH*dH + (5/t)*dH   # até aqui tudo igual ao código para resolver delta

    fR = 0.5*(1 + tanh) + R/(3*M2) # no código de delta o fR está depois do if else do alpha
    if abs(alpha) < 15:
        fRR = M3 + (0.5/e_AB)*sech**2  # função E no código de delta
    else:
        fRR = M3

    if abs(alpha) < 15:
        D = tanh*(g1*g2*sech)**2  # função D no código de delta
    else:
        D = 0

    E = 6*fRR*(t**3)*(H**3)

    F =  (11/t)*(dH/H)*dH + (dH/H)*(dH/H)*dH + (6/t)*ddH + (4/H)*dH*ddH   # função F do código de delta

    dddH = - ((A + B + C - D)/E) - F               # EDO

    g3 = (fRR/fR)*(k/t)**2  # segunda função gaux3 do código de delta
    Geffn = (1/fR)*(1 + 4*g3)/(1 + 3*g3)   # igual ao outro código

    fAB1 = 3*Geffn*H0**2   # igual ao outro código
    fAB2 = 2*t*t*H*H   # igual ao outro código

    dfg = - ((dH/H) + ((2 + fg)/t))*fg + (3/2)*(H0/H)**2*Geffn*(O_m/t)
    return [dH, ddH, dddH, dfg]



# Integração:
t_span =[0.17, 1]
t = np.linspace(0.17, 1, 1000)


# Condições iniciais:
ti = 0.17
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
Hi = H0*np.sqrt(O_mi + O_ri + O_L)
dHi = - ((H0*H0)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
fgi = ((H0/Hi)**2 * O_mi)**(6/11)  # no passado, todas as teorias f(R) recaem na RG
y0 = [Hi, dHi, ddHi, fgi]


# Solução AB-f(R):
solAB = solve_ivp(fgAB, t_span, y0, t_eval=t, method='LSODA')
fg_AB = solAB.y[3]


# definindo o redshift
z = 1/solAB.t - 1


plt.plot(z, fg_AB, color='black', linewidth = 2, label='$R^2$_AB model')
plt.legend()
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('fg(z).png', dpi=520, format='png', bbox_inches='tight')
plt.show()





