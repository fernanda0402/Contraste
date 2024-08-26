#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:17:50 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Parâmetros:

O_r0 = 0
mu = 10**(-11)
k = 0.125

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1




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
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol = solve_ivp(Den_RG, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D = sol.y[0]
    return D

def solfs8L(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol = solve_ivp(Den_RG, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_RG = sol.y[1]
    fs8L = sig8*sol.t*(dD_RG/solD(H0, O_m0)[999])
    return fs8L
#print(solfs8L(67.4, 0.315, 0.811))


###################################################################################

# Modelo Appleby-Battye:
def Den_AB(t, y, H0, O_m0, b):
    H_AB   = y[0]
    dH_AB  = y[1]
    ddH_AB = y[2]
    D_AB   = y[3]
    dD_AB  = y[4]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    R_vac = 12*H0**2
    e_AB = R_vac/np.log(1 + np.exp(2*b))
    M2 = e_AB/mu

    R_AB = 6*H_AB*(2*H_AB + t*dH_AB)
    alpha = (R_AB/e_AB) - b

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

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*gaux3)/(1 + 3*gaux3)

    fAB1 = 3*Geffn*H0**2
    fAB2 = 2*t*t*H_AB*H_AB

    ddD_AB = - ((3/t) + (dH_AB/H_AB))*dD_AB + (fAB1/fAB2)*O_m*D_AB
    return [dH_AB, ddH_AB, dddH_AB, dD_AB, ddD_AB]

def solD_AB(H0, O_m0, b):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    ti = p
    O_mi = O_m0*ti**(-3)
    O_ri = O_r0*ti**(-4)
    O_L = 1 - O_m0
    Hi = H0*np.sqrt(O_mi + O_ri + O_L)
    dHi = - ((H0**2)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
    ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
    y0 = [Hi, dHi, ddHi, p, 1]

    # Solução:
    sol = solve_ivp(Den_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
    D_AB  = sol.y[3]
    return D_AB

def solfs8AB(H0, O_m0, sig8, b):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    ti = p
    O_mi = O_m0*ti**(-3)
    O_ri = O_r0*ti**(-4)
    O_L = 1 - O_m0
    Hi = H0*np.sqrt(O_mi + O_ri + O_L)
    dHi = - ((H0**2)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
    ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
    y0 = [Hi, dHi, ddHi, p, 1]

    # Solução:
    sol = solve_ivp(Den_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
    D_AB  = sol.y[3]
    dD_AB = sol.y[4]
    fs8 = sig8*sol.t*(dD_AB/solD_AB(H0, O_m0, b)[999])
    return fs8
#print(solfs8AB(67.4, 0.315, 2))


#######################################################################################

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

    R0 = 6*ms*(2*((1-O_m0)/O_m0 + 1 + chi) - .5*(3 + 4*chi))

    xn10 = (R0/ms)**n
    xn2 = c1/(ms**n)
    xn30 = (n+1)*(c2/ms)*(R0**(2*(n-1)))
    xn40 = (n-1)*(R0**(n-2))
    xn50 = ((c2*xn10) + 1)

    fR_0 = 1 - n*xn2*ms*((R0**(n-1))/(xn50**2))
    fRR_0 = n*xn2*ms*( (xn30 - xn40)/(xn50**3))

    #gaux3 = 9e6*(k*k/(t*t))*(fRR/fR)*(fR_0/fRR_0)
    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/fR)*(1 + 4*gaux3)/(1 + 3*gaux3)

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

def solfs8HS(H0, O_m0, sig8, c2, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol = solve_ivp(Den_HS, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, c2, n))
    D_HS  = sol.y[2]
    dD_HS = sol.y[3]
    fs8 = sig8*sol.t*(dD_HS/solD_HS(H0, O_m0, c2, n)[999])
    return fs8
#print(solfs8HS(70, 0.3, 0.811, 1,1))


#####################################################################################

# Modelo Starobinsky:
def Den_S(t, y, H0, O_m0, lbd, n):
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    D_S = y[2]
    dD_S = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    O_L = 1 - O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    Rs = 6*(H0**2)*(1 - O_m0)/lbd
    star = 1 + (R/Rs)**2

    f = R + lbd*Rs*( (star**(-n)) - 1 )
    fR = 1 - 2*n*lbd*(R/Rs)*((star)**(-(n+1)))
    fRR = (2*n*lbd/Rs)*( 2*(n+1)*((R/Rs)**2)*(star**(-n-2)) - (star**(-n-1)) )

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)

    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)

    dYH = - J1*YH - J2*yH - J3

    H_S = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH_S = (R/(6*t*H_S)) - (2*H_S/t)

    R0 = ms*((12/O_m0) - 9)

    star0 = 1 + (R0/Rs)**2

    #fR0 = 1
    #fRR0 = 1
    fR0 = 1 - 2*n*lbd*(R0/Rs)*((star0)**(-(n+1)))
    fRR0 = (2*n*lbd/Rs)*( 2*(n+1)*((R0/Rs)**2)*(star0**(-n-2)) - (star0**(-n-1)) )

    #gaux3 = 9e6*(k*k/(t*t))*(fRR/fR)*(fR0/fRR0)

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*gaux3)/(1 + 3*gaux3)

    fS1 = 3*Geffn*H0**2
    fS2 = 2*t*t*H_S*H_S

    ddD_S = - ((3/t) + (dH_S/H_S))*dD_S + (fS1/fS2)*O_m*D_S
    return [YH, dYH, dD_S, ddD_S]


def solD_S(H0, O_m0, lbd, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol = solve_ivp(Den_S, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, lbd, n))
    D_S  = sol.y[2]
    return D_S

#print(solD_HS(70,0.3,1,1)[999])

def solfs8S(H0, O_m0, sig8, lbd, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol = solve_ivp(Den_S, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, lbd, n))
    D_S  = sol.y[2]
    dD_S = sol.y[3]
    fs8 = sig8*sol.t*(dD_S/solD_S(H0, O_m0, lbd, n)[999])
    return fs8
#print(solfs8HS(70, 0.3, 0.811, 1,1))


#########################################################################################

plt.plot(z, solfs8AB(70, 0.3, 0.8, 2), color='black', linewidth = 1.5, label='Appleby-Battye')
plt.plot(z, solfs8HS(70, 0.3, 0.8, 1, 1), color='purple', linewidth = 1.5, label='Hu-Sawicki ($n=1$)')
plt.plot(z, solfs8HS(70, 0.3, 0.8, 1, 2), color='magenta', linewidth = 1.5, label='Hu-Sawicki ($n=2$)')
plt.plot(z, solfs8S(70, 0.3, 0.8, 2, 1), color='orange', linewidth = 1.5, label='Starobinsky ($n=1$)', linestyle = '-.')
plt.plot(z, solfs8S(70, 0.3, 0.8, 2, 2), color='deeppink', linewidth = 1.5, label='Starobinsky ($n=2$)', linestyle = '-.')
plt.plot(z, solfs8L(70, 0.3, 0.8), color='blue', linewidth = 1.5, label='$\Lambda$CDM', linestyle = '--')



plt.legend()
plt.xlabel('$z$')
plt.ylabel('$[f\sigma_8](z)$')
#plt.xlim(-0.1, 2)
#plt.ylim(0.315, 0.47)
#plt.savefig('fs8_f(R).pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()


print('Comprimento de z:', len(z))
print('Comprimento de solfs8AB:', len(solfs8AB(70, 0.3, 0.8, 2)))

