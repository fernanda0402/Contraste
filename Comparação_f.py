#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:33:39 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from numpy import loadtxt, savetxt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['text.usetex'] = True
from gapp import gp




# Parâmetros:
#H0 = 67.4
#O_m0 = 0.315
O_r0 = 0
k = 0.125
#c2 = 1  # Valor de exemplo para c2
#n2 = 1     # Valor de exemplo para n
#n3 = 2
O_L0 = 0.7
#w0 = -0.957
#wa = -0.29
#O_k0 = -0.056 
#b = 2.18
mu = 10**(-7)
c = 1
Delta = 10**(-7)


p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1



# LCDM Model:
def contrast_GR(t, y, O_m0):
  D_GR = y[0]
  dD_GR = y[1]
  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)
  O_L = 1 - O_m0
  E_GR = np.sqrt(O_m + O_r + O_L)
  dE_GR = - (E_GR/t) - (0.5/t)*(1/E_GR)*(O_m + 2*O_r - 2*O_L)
  ddD_GR = - ((3/t) + (dE_GR/E_GR))*dD_GR + (3/(2*t**2*E_GR**2))*O_m*D_GR
  return [dD_GR, ddD_GR]

def D_GR(O_m0):
  y0 = [0.17, 1]
  sol_GR = solve_ivp(contrast_GR, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,))
  D_GR = sol_GR.y[0]
  return D_GR

def f_GR(O_m0):
  y0 = [0.17, 1]
  sol_GR = solve_ivp(contrast_GR, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,))
  D_GR = sol_GR.y[0]
  dD_GR = sol_GR.y[1]
  f_GR = t*(dD_GR/D_GR)
  return f_GR

####################################################################################


# wCDM Model:
def contrast_wCDM(t, y, O_m0, w):
  D_wCDM = y[0]
  dD_wCDM = y[1]
  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)
  O_L0 = 1 - O_m0
  O_L = O_L0*t**(-3*(1 + w))
  E_wCDM = np.sqrt(O_m + O_r + O_L)
  dE_wCDM = - (E_wCDM/t) - (0.5/t)*(1/E_wCDM)*(O_m + 2*O_r - 2*O_L)
  ddD_wCDM = - ((3/t) + (dE_wCDM/E_wCDM))*dD_wCDM + (3/(2*t**2*E_wCDM**2))*O_m*D_wCDM
  return [dD_wCDM, ddD_wCDM]

def D_wCDM(O_m0, w):
  y0 = [0.17, 1]
  sol_wCDM = solve_ivp(contrast_wCDM, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,w))
  D_wCDM = sol_wCDM.y[0]
  return D_wCDM

def f_wCDM(O_m0, w):
  y0 = [0.17, 1]
  sol_wCDM = solve_ivp(contrast_wCDM, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,w))
  dD_wCDM = sol_wCDM.y[1]
  f_wCDM = t*(dD_wCDM/D_wCDM(O_m0,w))
  return f_wCDM


#####################################################################################


# w0waCDM model:
def contrast_w0wa(t, y, O_m0, w0, wa):
  D_w0wa = y[0]
  dD_w0wa = y[1]
  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)
  O_L0 = 1 - O_m0
  #w_aux = 1 + (w0 + wa*(1 - t))
  O_L = O_L0*np.exp(-3*wa*(1-t))*(t**(-3*(1 + w0 + wa) ) )
  E_w0wa = np.sqrt(O_m + O_L)
  dE_w0wa = - (E_w0wa/t) - (0.5/t)*(1/E_w0wa)*(O_m + 2*O_r - 2*O_L)
  ddD_w0wa = - ((3/t) + (dE_w0wa/E_w0wa))*dD_w0wa + (3/(2*t**2*E_w0wa**2))*O_m*D_w0wa
  return [dD_w0wa, ddD_w0wa]

def D_w0wa(O_m0, w0, wa):
  y0 = [0.17, 1]
  sol_w0wa = solve_ivp(contrast_w0wa, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,w0,wa))
  D_w0wa = sol_w0wa.y[0]
  return D_w0wa

def f_w0wa(O_m0, w0, wa):
  y0 = [0.17, 1]
  sol_w0wa = solve_ivp(contrast_w0wa, t_span, y0, t_eval=t, method='LSODA', args=(O_m0,w0,wa))
  dD_w0wa = sol_w0wa.y[1]
  f_w0wa = t*(dD_w0wa/D_w0wa(O_m0,w0,wa))
  return f_w0wa


####################################################################################



#################################################################################### 


# Modelo AB-f(R):
def fgAB(t, y):
    H = y[0]
    dH = y[1]
    ddH = y[2]
    fg = y[3]
    
    H0 = 67.3
    O_m0 = 0.332
    b = 1.98

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
    Geffn = (1/(fR))*(1 + 4*(c**2)*g3)/(1 + 3*(c**2)*g3)   # igual ao outro código

    fAB1 = 3*Geffn*H0**2   # igual ao outro código
    fAB2 = 2*t*t*H*H   # igual ao outro código

    dfg = - ((dH/H) + ((2 + fg)/t))*fg + (3/2)*(H0/H)**2*Geffn*(O_m/t)
    return [dH, ddH, dddH, dfg]



# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
H0 = 67.3
O_m0 = 0.332
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


##################################################################################


# Modelo Starobinski n=1
def contrast_S1(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]
    D_S = y[2]
    dD_S = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3 * H0**2 * (1 - O_m0)
    Rs = 2*Lbd*mu
    star = 1 + (R/Rs)**2

    lmu2 = Lbd * mu**2

    f = R - 2*Lbd*(1 - (star**(-1)))
    fR = 1 - (R/lmu2)*(star**(-2))
    fRR = (1/lmu2)*(4 * (R/Rs)**2 * star**(-1) - 1) * star**(-2)

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3

    H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH = (R/(6*t*H)) - (2*H/t)

    c = 1

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*(c**2)*gaux3)/(1 + 3*(c**2)*gaux3)
    faux1 = 3*Geffn*H0**2
    faux2 = 2*t*t*H*H

    ddD_S = - ((3/t) + (dH/H))*dD_S + (faux1/faux2)*O_m*D_S
    return [YH, dYH, dD_S, ddD_S]

def D_S1(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S1 = solve_ivp(contrast_S1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    D1_S1 = sol_S1.y[2]
    return D1_S1

def f_S1(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S1 = solve_ivp(contrast_S1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    dD_S1 = sol_S1.y[3]
    f_S1 = sol_S1.t*(dD_S1/D_S1(H0, O_m0, mu))
    return f_S1


#D_Q3 = sol.y[0]
#dD_Q3 = sol.y[1]
#f_Q3 = sol.t*(dD_Q3/D_Q3)

#######################################################################################


 # Modelo Starobinski n=2
def contrast_S2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]
    D_S = y[2]
    dD_S = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3 * H0**2 * (1 - O_m0)
    Rs = 2*Lbd*mu
    star = 1 + (R/(2*Lbd*mu))**2

    M2 = Rs/Delta

    lmu2 = Lbd * (mu**2)
    star2 = (2*Lbd*mu)**2 + R**2

    f = R - 2*Lbd*(1 - (star**(-2))) + ((R**2)/(6*M2))
    fR = 1 - (2*R/lmu2)*(star**(-3)) + (R/(3*M2))
    fRR = ((2**7)*(Lbd**5)*(mu**4)/(star2**3))*(6*((R**2)/star2)- 1) + (1/(3*M2))

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3

    H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH = (R/(6*t*H)) - (2*H/t)

    c = 1

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*(c**2)*gaux3)/(1 + 3*(c**2)*gaux3)
    faux1 = 3*Geffn*H0**2
    faux2 = 2*t*t*H*H

    ddD_S = - ((3/t) + (dH/H))*dD_S + (faux1/faux2)*O_m*D_S
    return [YH, dYH, dD_S, ddD_S]

def D_S2(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S2 = solve_ivp(contrast_S2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    D1_S2 = sol_S2.y[2]
    return D1_S2

def f_S2(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S2 = solve_ivp(contrast_S2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    dD_S2 = sol_S2.y[3]
    f_S2 = sol_S2.t*(dD_S2/D_S2(H0, O_m0, mu))
    return f_S2



####################################################################################

# Modelo Hu-Sawicki n=1
def contrast_HS1(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]
    D_HS = y[2]
    dD_HS = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3*(H0**2)*(1 - O_m0)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    n = 1
    A = (R**n) + (mu**(2*n))
    B = (R**n)
    C = (n+1)*(R**(2*n - 2)) - (n- 1)*(mu**(2*n))*(R**(n-2))
    D = A**3

    f = R - 2*Lbd*(B/A)
    fR = 1 - 2*n*Lbd*(mu**(2*n))*(B/A)*(1/(R*A))
    fRR = 2*n*Lbd*(mu**(2*n))*(C/D)

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3

    H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH = (R/(6*t*H)) - (2*H/t)

    c = 1

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*(c**2)*gaux3)/(1 + 3*(c**2)*gaux3)
    faux1 = 3*Geffn*H0**2
    faux2 = 2*t*t*H*H

    ddD_HS = - ((3/t) + (dH/H))*dD_HS + (faux1/faux2)*O_m*D_HS
    return [YH, dYH, dD_HS, ddD_HS]

def D_HS1(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S1 = solve_ivp(contrast_HS1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    D1_S1 = sol_S1.y[2]
    return D1_S1

def f_HS1(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_HS1 = solve_ivp(contrast_HS1, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    dD_HS1 = sol_HS1.y[3]
    f_HS1 = sol_HS1.t*(dD_HS1/D_HS1(H0, O_m0, mu))
    return f_HS1


###################################################################################

# Modelo Hu-Sawicki n=2
def contrast_HS2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]
    D_HS = y[2]
    dD_HS = y[3]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    R = 3*ms*(4*yH + t*YH + t**(-3))

    Lbd = 3*(H0**2)*(1 - O_m0)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    n = 2
    A = (R**n) + (mu**(2*n))
    B = (R**n)
    C = (n+1)*(R**(2*n - 2)) - (n- 1)*(mu**(2*n))*(R**(n-2))
    D = A**3

    f = R - 2*Lbd*(B/A)
    fR = 1 - 2*n*Lbd*(mu**(2*n))*(B/A)*(1/(R*A))
    fRR = 2*n*Lbd*(mu**(2*n))*(C/D)

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)
    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)
    dYH = - J1*YH - J2*yH - J3

    H = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    dH = (R/(6*t*H)) - (2*H/t)

    c = 1

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*(c**2)*gaux3)/(1 + 3*(c**2)*gaux3)
    faux1 = 3*Geffn*H0**2
    faux2 = 2*t*t*H*H

    ddD_HS = - ((3/t) + (dH/H))*dD_HS + (faux1/faux2)*O_m*D_HS
    return [YH, dYH, dD_HS, ddD_HS]

def D_HS2(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_S2 = solve_ivp(contrast_HS2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    D1_S2 = sol_S2.y[2]
    return D1_S2

def f_HS2(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0, 0.17, 1]
    t_span = [0.17, 1]
    sol_HS2 = solve_ivp(contrast_HS2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    dD_HS2 = sol_HS2.y[3]
    f_HS2 = sol_HS2.t*(dD_HS2/D_HS2(H0, O_m0, mu))
    return f_HS2


######################################################################################


# O_KCDM model:
def contrast_OKCDM(t, y, O_m0, O_K0):
  D_OK = y[0]
  dD_OK = y[1]
  O_K = O_K0*t**(-2)
  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)
  O_L = 1 - O_m0 - O_r0
  E_OK = np.sqrt(O_K + O_m + O_r + O_L)
  dE_OK = - (E_OK/t) - (0.5/t)*(1/E_OK)*(O_m + 2*O_r - 2*O_L)
  ddD_OK = - ((3/t) + (dE_OK/E_OK))*dD_OK + (3/(2*t**2*E_OK**2))*O_m*D_OK
  return [dD_OK, ddD_OK]

def D_OKCDM(O_m0, O_K0):
  y0 = [0.17, 1]
  sol_OK = solve_ivp(contrast_OKCDM, t_span, y0, t_eval=t, method='LSODA', args=(O_m0, O_K0))
  D_OK = sol_OK.y[0]
  return D_OK

def f_OKCDM(O_m0, O_K0):
  y0 = [0.17, 1]
  sol_OK = solve_ivp(contrast_OKCDM, t_span, y0, t_eval=t, method='LSODA', args=(O_m0, O_K0))
  dD_OK = sol_OK.y[1]
  f_OK = t*(dD_OK/D_OKCDM(O_m0, O_K0))
  return f_OK


################### CURVA SOMBREADA #########################################

data_fz = np.genfromtxt('/home/usuario/Documentos/Dados/fz_data.csv', delimiter=', ')

z_gapp = data_fz[:,0]
fz = data_fz[:,1]
sig_fz = data_fz[:,2]


# nomeando
x_gapp = z_gapp
y_gapp = fz
e = sig_fz

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 5.0
nstar = 1000

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [2.0, 2.0]

# initialization of the Gaussian Process
g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma


######################################################################



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


#Solução do delta
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


# Solução para f(z)
def solf_fQ3(H0, O_m0, m):
    t_span = [ti, 1]
    y0 = [ti, 1]
    sol = solve_ivp(Densidade_Q3, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, m))
    D_Q3 = sol.y[0]
    dD_Q3 = sol.y[1]
    f_Q3 = sol.t*(dD_Q3/D_Q3)
    return f_Q3






# Modelo f(Q) - 2
def Densidade_Q2(t, y, H0, O_m0, m):
    D_Q2 = y[0]
    dD_Q2 = y[1]

    O_m = O_m0*t**(-3)

    M = m*H0
    M4 = M**4

    H_Q2 = np.sqrt( (H0**2)*O_m / 2 + (1/6)*np.sqrt(9*H0**4*O_m**2 + 3*M4) ) # eq de Friedmann minha
    dH_Q2 = ( (M**4 - 12*H_Q2**4)*3*H_Q2 ) / (2*t*(M**4 + 12*H_Q2**4) ) # obtida com a 1 eq de Friedmann minha

    Q = 6*(H_Q2**2)

    dfQ_2 = 1 - (M**4/Q**2)  # primeira derivada da f(Q)

    f1 = 3*(H0**2)*O_m
    f2 = 2*(t**2)*(H_Q2**2)*dfQ_2

    ddD_Q2 = - ((3/t) + dH_Q2/H_Q2)*dD_Q2 + ( (f1/f2)*D_Q2 )  # equação do contraste
    return [dD_Q2, ddD_Q2]

def solD_fQ2(H0, O_m0, m):
    # Espaço de integração:
    t_span = [ti, 1]
    t = np.linspace(ti, 1, 1000)

    # Condições iniciais:
    y0 = [ti, 1]

    # Solução:
    sol_Q2 = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0, m))
    D_Q2 = sol_Q2.y[0]
    return D_Q2


# Solução para f(z)
def solf_fQ2(H0, O_m0, m):
    t_span = [ti, 1]
    y0 = [ti, 1]
    sol = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, m))
    D_Q2 = sol.y[0]
    dD_Q2 = sol.y[1]
    f_Q2 = sol.t*(dD_Q2/D_Q2)
    return f_Q2



# plote

plt.plot(z,f_GR(0.339), color='blue', linewidth = 2, linestyle='--', label='$\Lambda$CDM')
#plt.plot(z, f_wCDM(0.306, -0.887), color='red', linewidth = 2, label='$\omega$CDM')
#plt.plot(z, f_w0wa(0.320, -0.876, -0.07), color='darkgoldenrod', linewidth = 2, label='$\omega_0 \omega_a$CDM')
#plt.plot(z, f_OKCDM(0.317, 0.2), color='green', linewidth = 2, label='$\Omega_k$-CDM')


#plt.plot(z, f_S1(67.1, 0.328, 0.589), color='orange', linewidth=2, label='Starobinski (n=1)', linestyle = '-.')
plt.plot(z, f_S2(66.4, 0.350, 1.047), color='green', linewidth=2, label='Starobinski (n=2)', linestyle = '-.')
plt.plot(z, f_HS1(69.7, 0.269, 82), color='purple', linewidth=2, label='Hu-Sawicki (n=1)')
plt.plot(z, f_HS2(66.8, 0.327, 104), color='red', linewidth=2, label='Hu-Sawicki (n=2)')
plt.plot(z, fg_AB, color='black', linewidth = 2, label='$R^2$_AB')

#plt.plot(z, solf_fQ3(66.9, 0.347, 2.0), color='maroon', label='$F_1(Q)$', linewidth = 2)
#plt.plot(z, solf_fQ2(67.4, 0.315, 2.0331), color='slategray', label='$F_2(Q)$', linewidth = 2)

plt.plot(xi, y_pred, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")


plt.fill(np.concatenate([xi, xi[::-1]]),
        np.concatenate([y_pred - 1.9100 * sigma,
                      (y_pred + 1.9100 * sigma)[::-1]]),
        alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
        np.concatenate([y_pred - 1.00 * sigma,
                       (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')

#plt.fill(np.concatenate([xi, xi[::-1]]),
#        np.concatenate([y_pred - 2.997 * sigma,
#                       (y_pred + 2.997 * sigma)[::-1]]),
#         alpha=.5, color = 'lightblue', ec='None')


plt.ylim(0.4,1) 
plt.xlim(0,1)   
plt.legend(prop={'size':9.5}, loc='lower right')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$f(z)$', fontsize=15)
#plt.savefig('f(z)_comparação_manoel.pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()



###################################################################################




#plt.plot(z, f_S1(67.1, 0.328, 0.589), color='orange', linewidth=2, label='Starobinski (n=1)', linestyle = '-.')
#plt.plot(z, f_S2(66.4, 0.350, 1.047), color='deeppink', linewidth=2, label='Starobinski (n=2)', linestyle = '-.')
#plt.plot(z, f_HS1(69.7, 0.269, 82), color='purple', linewidth=2, label='Hu-Sawicki (n=1)')
#plt.plot(z, f_HS2(66.8, 0.327, 104), color='magenta', linewidth=2, label='Hu-Sawicki (n=2)')
#plt.plot(z, fg_AB, color='black', linewidth = 2, label='$R^2$_AB')
#plt.plot(z, solf_fQ3(66.9, 0.347, 2.0), color='maroon', label='$F_1(Q)$', linewidth = 2)

plt.plot(xi, y_pred, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")
plt.fill(np.concatenate([xi, xi[::-1]]),
        np.concatenate([y_pred - 1.9100 * sigma,
                      (y_pred + 1.9100 * sigma)[::-1]]),
        alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
        np.concatenate([y_pred - 1.00 * sigma,
                       (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')

#plt.fill(np.concatenate([xi, xi[::-1]]),
#        np.concatenate([y_pred - 2.997 * sigma,
#                       (y_pred + 2.997 * sigma)[::-1]]),
#         alpha=.5, color = 'lightblue', ec='None')


plt.ylim(0.4,1) 
plt.xlim(0,1)   
plt.legend(prop={'size':7.5}, loc='lower right')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$f(z)$', fontsize=15)
#plt.savefig('f(z)_comparação_MG.pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()





