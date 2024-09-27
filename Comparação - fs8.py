#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:26:53 2024

@author: usuario
"""

# Bibliotecas:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from gapp import gp
from scipy.special import gamma
from numpy import exp
from scipy.integrate import cumtrapz


# SOMBREADA

# BAIXANDO O ARQUIVO DE fs8

data = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z_gapp = data[:, 0]
fs8_gapp = data[:, 1]
sig_fs8 = data[:, 2]



# DEFININDO INVGAMMA

def invgamma(x, a, b1):
    x = x[1]
    p = b1**a/gamma(a) * x**(-1 - a) * exp(-b1/x)

    return p


# nomeando
x_gapp = z_gapp[z_gapp<2]
y_gapp = fs8_gapp[z_gapp<2]
e = sig_fs8[z_gapp<2]

# xmin, xmax and nstar are interpreted as two-dimensional vectors
xmin = 0
xmax = 5.0
nstar = 1000

# initial values of the hyperparameters of the squared-exponential covariance function
initheta = [0.5, 0.3]

# initialization of the Gaussian Process
#g = gp.GaussianProcess(x_gapp, y_gapp, e, cXstar=(xmin, xmax, nstar))
g = gp.GaussianProcess(x_gapp,y_gapp,e,cXstar=(xmin, xmax, nstar),
                        prior=invgamma, priorargs=(4, 1.5),
                        grad='False')

# training of the hyperparameters and reconstruction of the function
(rec, theta) = g.gp(theta=initheta)

xi = rec[:, 0]

y_pred = rec[:, 1]
sigma  = rec[:, 2]

y_pred_95_less = y_pred - 1.9600*sigma
y_pred_95_plus = y_pred + 1.9600*sigma





#####################################################################################

# Parâmetros:

O_r0 = 0
O_L0 = 0.7
mu = 10**(-11)
k = 0.125
w0 = -0.957
wa = -0.29
O_k0 = -0.056

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


###################################################################################3

# Modelo wCDM
def Densidade_w1(t, y, H0, O_m0):    # t é o fator de escala e y é o delta (contraste)
    D_RG_w1  = y[0]     # contraste
    dD_RG_w1 = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)
    O_L = O_L0*(t**(-3*(1+w0) ) )

    H_RG_w1 = H0*np.sqrt(O_m + O_L)    # H(z)
    dH_RG_w1 = - (H_RG_w1/t) - 0.5*(H0/t)*(H0/H_RG_w1)*(O_m - 2*O_L)        # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w1**2)

    ddD_RG_w1 = - ((3/t) + (dH_RG_w1/H_RG_w1))*dD_RG_w1 + (faux1/faux2)*O_m*D_RG_w1   #  eq. da derivada segunda do contraste
    return [dD_RG_w1, ddD_RG_w1]

def solD_w1(H0, O_m0):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w1 = solve_ivp(Densidade_w1, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D_w1 = sol_w1.y[0]
    return D_w1

def solfs8L_w1(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w1 = solve_ivp(Densidade_w1, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_w1 = sol_w1.y[1]
    fs8L_w1 = sig8*sol_w1.t*(dD_w1/solD_w1(H0, O_m0)[999])
    return fs8L_w1



###################################################################################33

# Modelo w0waCDM

def Densidade_w(t, y, H0, O_m0): # t é o fator de escala e y é o delta (contraste)
   
    D_RG_w  = y[0]     # contraste
    dD_RG_w = y[1]    # primeira derivada do contraste

    O_m = O_m0*t**(-3)
    O_L = O_L0*(t**(-3*(1 + w0 + wa*(1-t)) ) )

    H_RG_w = H0*np.sqrt(O_m + O_L)    # H(z)
    dH_RG_w = - (H_RG_w/t) - 0.5*(H0/t)*(H0/H_RG_w)*(O_m - 2*O_L)        # derivada do H(z)

# funções definidas somente para facilitar a escrita da derivada segunda do contraste
    faux1 = 3*H0**2
    faux2 = 2*(t**2)*(H_RG_w**2)

    ddD_RG_w = - ((3/t) + (dH_RG_w/H_RG_w))*dD_RG_w + (faux1/faux2)*O_m*D_RG_w   #  eq. da derivada segunda do contraste
    return [dD_RG_w, ddD_RG_w]

def solD_w(H0, O_m0):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w = solve_ivp(Densidade_w, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    D_w = sol_w.y[0]
    return D_w

def solfs8L_w(H0, O_m0, sig8):
    # Espaço de integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_w = solve_ivp(Densidade_w, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0))
    dD_w = sol_w.y[1]
    fs8L_w = sig8*sol_w.t*(dD_w/solD_w(H0, O_m0)[999])
    return fs8L_w



#####################################################################################




#######################################################################################


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
    sol_AB = solve_ivp(Den_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
    D_AB  = sol_AB.y[3]
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
    sol_AB = solve_ivp(Den_AB, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
    D_AB  = sol_AB.y[3]
    dD_AB = sol_AB.y[4]
    fs8_AB = sig8*sol_AB.t*(dD_AB/solD_AB(H0, O_m0, b)[999])
    return fs8_AB
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
    sol_HS = solve_ivp(Den_HS, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, c2, n))
    D_HS  = sol_HS.y[2]
    return D_HS

#print(solD_HS(70,0.3,1,1)[999])

def solfs8HS(H0, O_m0, sig8, c2, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol_HS = solve_ivp(Den_HS, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, c2, n))
    D_HS  = sol_HS.y[2]
    dD_HS = sol_HS.y[3]
    fs8_HS = sig8*sol_HS.t*(dD_HS/solD_HS(H0, O_m0, c2, n)[999])
    return fs8_HS
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
    sol_S = solve_ivp(Den_S, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, lbd, n))
    D_S = sol_S.y[2]
    return D_S

#print(solD_HS(70,0.3,1,1)[999])

def solfs8S(H0, O_m0, sig8, lbd, n):
    #Integração:
    t_span = [p, 1]

    # Condições iniciais:
    y0 = [(1-O_m0)/O_m0, 0, p, 1]

    # Solução:
    sol_S = solve_ivp(Den_S, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, lbd, n))
    D_S  = sol_S.y[2]
    dD_S = sol_S.y[3]
    fs8_S = sig8*sol_S.t*(dD_S/solD_S(H0, O_m0, lbd, n)[999])
    return fs8_S
#print(solfs8HS(70, 0.3, 0.811, 1,1))



#######################################################################################


# Modelo Ok-CDM - arxiv 0903.0001

t = np.linspace(p, 1, 1000)
z = (1/t) - 1

def growth(O_m0, O_K0):
    
  w = -1
  
  O_L = 1 - O_m0 - O_K0
  O_K = O_K0*t / ( O_m0 + O_K0*t + O_L*(t**(-3*w)) )
  O_M = O_m0/( O_m0 + O_K0*t + O_L*(t**(-3*w)) )
  gamma = 0.55
  f_K = O_M**(gamma) + (gamma - 4/7)*O_K
  return f_K


def ln_delta(O_m0, O_K0):
  y = growth(O_m0, O_K0)/t
  ln_delta = cumtrapz(y, t, initial=0)
  return ln_delta

def delta(O_m0, O_K0):
  delta = np.exp(ln_delta(O_m0, O_K0))
  delta = delta/delta[999]
  return delta

def fs8(O_m0, O_K0, sig80):
      fs8 = sig80*growth(O_m0, O_K0)*delta(O_m0, O_K0)
      return fs8

#######################################################################################




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


def solD_fQ3(H0, O_m0, m):
    # Espaço de integração:
    t_span = [p, 1]
    t = np.linspace(p, 1, 1000)

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_Q3 = solve_ivp(Densidade_Q3, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0, m))
    D_Q3 = sol_Q3.y[0]
    return D_Q3


def solfs8_fQ3(H0, O_m0, sig8, m):
    y0 = [p, 1]
    t_span = [p, 1]
    sol = solve_ivp(Densidade_Q3, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, m))
    D = sol.y[0]
    dD = sol.y[1]
    fs8 = sig8*sol.t*(dD/solD_fQ3(H0, O_m0, m)[999])
    return fs8



# Modelo f(Q) - 2
def Densidade_Q2(t, y, H0, O_m0, m):
    D_Q2 = y[0]
    dD_Q2 = y[1]

    O_m = O_m0*t**(-3)

    M = m*H0
    M4 = M**4

    H_Q2 = np.sqrt( (H0**2)*O_m / 2 + (1/6)*np.sqrt(9*H0**4*O_m**2 + M4) ) # eq de Friedmann minha
    dH_Q2 = ( (M**4 - 36*H_Q2**4)*H_Q2 ) / (2*t*(M**4 + 12*H_Q2**4) ) # obtida com a 1 eq de Friedmann minha

    Q = 6*(H_Q2**2)

    dfQ_2 = 1 - (M**4/Q**2)  # primeira derivada da f(Q)

    f1 = 3*(H0**2)*O_m
    f2 = 2*(t**2)*(H_Q2**2)*dfQ_2

    ddD_Q2 = - ((3/t) + dH_Q2/H_Q2)*dD_Q2 + ( (f1/f2)*D_Q2 )  # equação do contraste
    return [dD_Q2, ddD_Q2]

def solD_fQ2(H0, O_m0, m):
    # Espaço de integração:
    t_span = [p, 1]
    t = np.linspace(p, 1, 1000)

    # Condições iniciais:
    y0 = [p, 1]

    # Solução:
    sol_Q2 = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', args=(H0, O_m0, m))
    D_Q2 = sol_Q2.y[0]
    return D_Q2


# Solução para f(z)
def solf_fQ2(H0, O_m0, m):
    t_span = [p, 1]
    y0 = [p, 1]
    sol = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, m))
    D_Q2 = sol.y[0]
    dD_Q2 = sol.y[1]
    f_Q2 = sol.t*(dD_Q2/D_Q2)
    return f_Q2


def solfs8_fQ2(H0, O_m0, sig8, m):
    y0 = [p, 1]
    t_span = [p, 1]
    sol = solve_ivp(Densidade_Q2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, m))
    D = sol.y[0]
    dD = sol.y[1]
    fs8 = sig8*sol.t*(dD/solD_fQ2(H0, O_m0, m)[999])
    return fs8



plt.plot(z, solfs8L(70, 0.3, 0.8), color='blue', linewidth = 2.5, label='$\Lambda$CDM', linestyle = '--')
plt.plot(z, solfs8L_w1(70, 0.3, 0.8), color='red', linewidth = 2, label='$\omega$CDM')
plt.plot(z, solfs8L_w(70, 0.3, 0.8), color='darkgoldenrod', linewidth = 2, label='$\omega_0 \omega_a$CDM')
plt.plot(z, fs8(0.3, -0.056, 0.811), color='green', label='$\Omega_k$-CDM', linewidth = 2)


plt.plot(z, solfs8S(70, 0.3, 0.8, 2, 1), color='orange', linewidth = 2, label='Starobinsky ($n=1$)', linestyle = '-.')
plt.plot(z, solfs8S(70, 0.3, 0.8, 2, 2), color='deeppink', linewidth = 2.8, label='Starobinsky ($n=2$)', linestyle = '-.')
plt.plot(z, solfs8HS(70, 0.3, 0.8, 1, 1), color='purple', linewidth = 2, label='Hu-Sawicki ($n=1$)')
plt.plot(z, solfs8HS(70, 0.3, 0.8, 1, 2), color='magenta', linewidth = 2, label='Hu-Sawicki ($n=2$)')
plt.plot(z, solfs8AB(70, 0.3, 0.8, 2), color='black', linewidth = 2, label='$R^2$_AB')


plt.plot(z, solfs8_fQ3(70, 0.3, 0.8, 2.0331), color='maroon', label='$F_1(Q)$', linewidth = 2)
plt.plot(z, solfs8_fQ2(70, 0.3, 0.8, 2.0331), color='slategray', label='$F_2(Q)$', linewidth = 2)

plt.plot(xi, y_pred, color = 'navy', label='GP', linestyle="dotted")

plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')
plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred - 1.00 * sigma,
                        (y_pred + 1.00 * sigma)[::-1]]),
         alpha=.5, color = 'dodgerblue', ec='None')

plt.legend(prop={'size':5.5}, loc='lower right')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$[f\sigma_8](z)$', fontsize=15)
plt.xlim(0, 1)
plt.ylim(0.3, 0.53)
#plt.savefig('fs8_comparação.pdf', dpi=520, format='pdf', bbox_inches='tight')
plt.show()
















