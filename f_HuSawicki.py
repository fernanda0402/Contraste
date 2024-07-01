#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:04:05 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parâmetros:
H0 = 67.4
O_m0 = 0.315
O_r0 = 0
k = 0.125
c2 = 1  # Valor de exemplo para c2
n = 2     # Valor de exemplo para n

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1

def f_HS(t, y, H0, O_m0, O_r0, c2, n):
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    fg_HS = y[2]

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

    dfg_HS = - ((dH_HS/H_HS) + ((2 + fg_HS)/t))*fg_HS + (3/2)*(H0/H_HS)**2*Geffn*(O_m/t)
    return [YH, dYH, dfg_HS]




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
y0 = [Hi, dHi, fgi]



# Solução:
solHS = solve_ivp(f_HS, t_span, y0, t_eval=t, method='RK45', args=(H0, O_m0, O_r0, c2, n), rtol=1e-3, atol=1e-6)

# Ajustando o intervalo de interpolação
t_interp = np.linspace(solHS.t[0], solHS.t[-1], 1000)

interp_func = interp1d(solHS.t, solHS.y[2], kind='cubic')
fgHS_interp = interp_func(t_interp)



plt.plot(z, fgHS_interp, color='purple', linewidth=2, label='Hu-Sawicki (n=2)')
plt.legend()
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('f_HuSawicki.png', dpi=520, format='png', bbox_inches='tight')
plt.show()










