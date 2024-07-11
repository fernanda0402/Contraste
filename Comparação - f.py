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
H0 = 67.4
O_m0 = 0.315
O_r0 = 0
k = 0.125
c2 = 1  # Valor de exemplo para c2
n2 = 1     # Valor de exemplo para n
n3 = 2
O_L0 = 0.7
w0 = -0.957
wa = -0.29
O_k0 = -0.056 
b = 2
mu = 10**(-7)

p = 0.17 #0.049787068
t = np.linspace(p, 1, 1000)
z = (1/t) - 1



# Modelo LCDM

def fgRG(t, y):
    fg_RG  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = 1 - O_m0

    H_RG = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG = - (H_RG/t) - 0.5*(H0/t)*(H0/H_RG)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG**2)

    dfg_RG = - ((dH_RG/H_RG) + ((2+fg_RG)/t))*fg_RG + (3/2)*((H0/H_RG)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_RG


# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
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

####################################################################################


# Modelo wCDM:
def fg_w(t, y):
    fg_w1  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = O_L0*(t**(-3*(1+w0) ) )

    H_RG_w1 = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG_w1 = - (H_RG_w1/t) - 0.5*(H0/t)*(H0/H_RG_w1)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w1**2)

    dfg_w1 = - ((dH_RG_w1/H_RG_w1) + ((2+fg_w1)/t))*fg_w1 + (3/2)*((H0/H_RG_w1)**2)*(O_m/t)  # equação diferencial para o f
    return dfg_w1


# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
fgi = ((H0/Hi)**2 * O_mi)**(6/11)
y0 = [fgi]


# Solução RG:
sol_w1 = solve_ivp(fg_w, t_span, y0, t_eval=t, method='LSODA')
fg_w = sol_w1.y[0]


# definindo o redshift
z = 1/sol_w1.t - 1


#####################################################################################


# Modelo w0waCDM:
def fgw0wa(t, y):
    fg_w0wa  = y[0]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)
    O_L = O_L0*(t**(-3*(1 + w0 + wa*(1-t)) ) )

    H_RG_w = H0*np.sqrt(O_m + O_r + O_L)
    dH_RG_w = - (H_RG_w/t) - 0.5*(H0/t)*(H0/H_RG_w)*(O_m + 2*O_r - 2*O_L)

    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w**2)

    df_w = - ((dH_RG_w/H_RG_w) + ((2+fg_w0wa)/t))*fg_w0wa + (3/2)*((H0/H_RG_w)**2)*(O_m/t)  # equação diferencial para o f
    return df_w


# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
fgi = ((H0/Hi)**2 * O_mi)**(6/11)
y0 = [fgi]


# Solução RG:
sol_w = solve_ivp(fgw0wa, t_span, y0, t_eval=t, method='LSODA')
fg_w0wa = sol_w.y[0]


# definindo o redshift
z = 1/sol_w.t - 1


####################################################################################


# Modelo Ok-CDM:

a = np.linspace(0.17, 1, 1000) 

z_LC = (1/a) - 1

O_L1 = 1 - O_m0 - O_k0
     
H_LC = H0*np.sqrt(O_m0*(a**(-3)) + O_L1 + O_k0*(a**(-2)))

Om = O_m0*(a**(-3)) / ( (H_LC/H0)**2 )

f_LC = Om**0.55




#################################################################################### 


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
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
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


# Modelo Starobinski (n = 1)
def f_S(t, y, lbd, n): # lbd é o parâmetro lambda
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    fg_s = y[2]

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

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*gaux3)/(1 + 3*gaux3)

    fAB1 = 3*Geffn*H0**2
    fAB2 = 2*t*t*H_S*H_S

    dfg_s =  - ((dH_S/H_S) + ((2 + fg_s)/t))*fg_s + (3/2)*(H0/H_S)**2*Geffn*(O_m/t)
    return [YH, dYH, dfg_s]



# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)



# Condições iniciais:
ti = p
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
Hi = H0*np.sqrt(O_mi + O_ri + O_L)
dHi = - ((H0*H0)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
fgi = ((H0/Hi)**2 * O_mi)**(6/11)  # no passado, todas as teorias f(R) recaem na RG
y0 = [Hi, dHi, fgi]


# Solução
lbd = 1 
n = 1
sol_s = solve_ivp(f_S, t_span, y0, t_eval=t, args=(lbd, n), method='LSODA', rtol=1e-6)
fg_S = sol_s.y[2]


#######################################################################################


# Modelo Starobinski (n = 2)
def f_S(t, y, lbd, n1): # lbd é o parâmetro lambda
    yH = y[0]
    YH = y[1]        # YH = dyH/da
    fg_s = y[2]

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    O_L = 1 - O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))

    Rs = 6*(H0**2)*(1 - O_m0)/lbd
    star = 1 + (R/Rs)**2

    f = R + lbd*Rs*( (star**(-n1)) - 1 )
    fR = 1 - 2*n1*lbd*(R/Rs)*((star)**(-(n1+1)))
    fRR = (2*n1*lbd/Rs)*( 2*(n1+1)*((R/Rs)**2)*(star**(-n1-2)) - (star**(-n1-1)) )

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

    gaux3 = (fRR/fR)*(k*(H0/100)/t)**2
    Geffn = (1/(fR))*(1 + 4*gaux3)/(1 + 3*gaux3)

    fAB1 = 3*Geffn*H0**2
    fAB2 = 2*t*t*H_S*H_S

    dfg_s =  - ((dH_S/H_S) + ((2 + fg_s)/t))*fg_s + (3/2)*(H0/H_S)**2*Geffn*(O_m/t)
    return [YH, dYH, dfg_s]



# Integração:
t_span =[p, 1]
t = np.linspace(p, 1, 1000)



# Condições iniciais:
ti = p
Hi = H0*np.sqrt(O_m0*ti**(-3) + (1 - O_m0))
O_mi = O_m0*ti**(-3)
O_ri = O_r0*ti**(-4)
O_L = 1 - O_m0
Hi = H0*np.sqrt(O_mi + O_ri + O_L)
dHi = - ((H0*H0)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
fgi = ((H0/Hi)**2 * O_mi)**(6/11)  # no passado, todas as teorias f(R) recaem na RG
y0 = [Hi, dHi, fgi]


# Solução
lbd = 1 
n1 = 2
sol_s = solve_ivp(f_S, t_span, y0, t_eval=t, args=(lbd, n1), method='LSODA', rtol=1e-6)
fg_S_n1 = sol_s.y[2]


####################################################################################


# Modelo de Hu-Sawicki (n = 1)

def f_HS(t, y, H0, O_m0, O_r0, c2, n2):
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

    xn1 = (R/ms)**n2
    xn2 = c1/(ms**n2)
    xn3 = (n2+1)*(c2/ms)*R**(2*(n2-1))
    xn4 = (n2-1)*R**(n2-2)
    xn5 = ((c2*xn1) + 1)

    f = R - ms*((c1*xn1)/((c2*xn1)  + 1))
    fR = 1 - n2*xn2*ms*((R**(n2-1))/(xn5**2))
    fRR = n2*xn2*ms*( (xn3 - xn4)/(xn5**3))

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
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
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
solHS = solve_ivp(f_HS, t_span, y0, t_eval=t, method='RK45', args=(H0, O_m0, O_r0, c2, n2), rtol=1e-3, atol=1e-6)

# Ajustando o intervalo de interpolação
t_interp = np.linspace(solHS.t[0], solHS.t[-1], 1000)

interp_func = interp1d(solHS.t, solHS.y[2], kind='cubic')
fgHS_interp = interp_func(t_interp)


###################################################################################3


# Modelo de Hu-Sawicki (n = 2)

def f_HS(t, y, H0, O_m0, O_r0, c2, n3):
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

    xn1 = (R/ms)**n3
    xn2 = c1/(ms**n3)
    xn3 = (n3+1)*(c2/ms)*R**(2*(n3-1))
    xn4 = (n3-1)*R**(n3-2)
    xn5 = ((c2*xn1) + 1)

    f = R - ms*((c1*xn1)/((c2*xn1)  + 1))
    fR = 1 - n2*xn2*ms*((R**(n3-1))/(xn5**2))
    fRR = n3*xn2*ms*( (xn3 - xn4)/(xn5**3))

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
t_span =[p, 1]
t = np.linspace(p, 1, 1000)


# Condições iniciais:
ti = p
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
solHS = solve_ivp(f_HS, t_span, y0, t_eval=t, method='RK45', args=(H0, O_m0, O_r0, c2, n3), rtol=1e-3, atol=1e-6)

# Ajustando o intervalo de interpolação
t_interp = np.linspace(solHS.t[0], solHS.t[-1], 1000)

interp_func = interp1d(solHS.t, solHS.y[2], kind='cubic')
fgHS_interp_n3 = interp_func(t_interp)



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




# plote

plt.plot(z, fg_RG, color='blue', linewidth = 5, linestyle='--', label='$\Lambda$CDM')
plt.plot(z, fg_w, color='deeppink', linewidth = 2, label='$\omega$CDM')
plt.plot(z, fg_w0wa, color='red', linewidth = 2, label='$\omega_0 \omega_a$CDM')
plt.plot(z_LC, f_LC, color='green', label='$\Omega_k$-CDM')
plt.plot(z, fg_AB, color='black', linewidth = 2, label='$R^2$_AB model')
plt.plot(z, fg_S, color='orange', linewidth=2, label='Starobinski (n=1)')
plt.plot(z, fg_S_n1, color='darkgoldenrod', linewidth=2, label='Starobinski (n=2)')
plt.plot(z, fgHS_interp, color='purple', linewidth=2, label='Hu-Sawicki (n=1)')
plt.plot(z, fgHS_interp_n3, color='magenta', linewidth=2, label='Hu-Sawicki (n=2)')
#plt.plot(xi, y_pred, color = 'green', label='Prediction', linestyle="-")
plt.fill(np.concatenate([xi, xi[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                       (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')


plt.ylim(0.4,1.15) 
plt.xlim(0,1)   
plt.legend(prop={'size':6.5})
plt.xlabel('z')
plt.ylabel('$f(z)$')
#plt.savefig('f(z)_comparação_z=1.png', dpi=520, format='png', bbox_inches='tight')
plt.show()




