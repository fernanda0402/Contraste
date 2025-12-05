#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:45:03 2025

@author: felipe
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt, savetxt
from gapp import gp
plt.rcParams['text.usetex'] = True
import pandas as pd
import scipy as sp
from scipy.integrate import solve_ivp
from scipy import integrate
import pyccl as ccl


# 1) BAIXANDO O ARQUIVO

data = np.genfromtxt('/home/usuario/Documentos/Dados/dlc_snia.dat', delimiter='\t')

zCMB = data[:, 0]
dl = data[:, 1]
dlerr = data[:, 2]

######## PROCESSO GAUSSIANO USANDO GAPP #################

# nomeando
x_gapp = zCMB
y_gapp = dl
e = dlerr

#print(max(zCMB))


# ABRINDO OS DADOS RECONSTRUÍDOS

data = np.genfromtxt('/home/usuario/Documentos/Códigos/Estudo_do_contraste/dlz_recon_covariance.csv', delimiter=',')

xi = data[:, 0]
y_pred = data[:, 1]
sigma = data[:, 2]


c = 299792.458


# Modelos tipo-LCDM

def mb(x, Om0, h0, w0, wa, Mb, Ok0):

    cosmo = ccl.Cosmology(
        Omega_c=Om0-0.0494, Omega_b=0.0494, h=h0,
        sigma8=0.8120, n_s=0.9649,wa=wa, w0=w0, Omega_k=Ok0)
    
    mu = ccl.background.distance_modulus(cosmo, 1/(1+x))
    
    return mu + Mb

x = np.linspace(0.0001, 2.27, 1000)

m = mb(x, 0.3, 0.7, -1, 0, -19.416, 0)

def DL(x, Om0, h0, w0, wa, Mb, Ok0):
    
    return 10 ** ((mb(x,Om0,h0, w0,wa,Mb, Ok0)-Mb-25) / 5)

#plt.plot(x, DL(x, 0.3, 0.7, -1, 0, -19.416, 0))



# plote da reconstrução com os dados
fig, ax = plt.subplots()
plt.tick_params(labelsize=14, color='purple')

plt.plot(x, DL(x, 0.339, 0.672, -1, 0, -19.416, 0)*67.2/c, color='blue', linewidth = 2.5, linestyle='--', label='$\Lambda$CDM')
plt.plot(xi, y_pred*73/c, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")
plt.errorbar(x_gapp, y_gapp*73, e*73, fmt='o', color='red', markersize=5, label='Data')

plt.plot()
plt.fill(np.concatenate([xi, xi[::-1]]),
          np.concatenate([y_pred*73/c - 1.9600 * sigma*73/c,
                        (y_pred*73/c + 1.9600 * sigma*73/c)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')

plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred*73/c - 1.00 * sigma*73/c,
                        (y_pred*73/c + 1.00 * sigma*73/c)[::-1]]),
        alpha=.5, color = 'dodgerblue', ec='None')

    

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$D_L(z)H_0/c$', fontsize=15)
plt.legend(loc='upper left', prop={'size':10})
#plt.ylim(0,8000)
plt.ylim(0,5)
plt.xlim(0,2.27)
#plt.savefig('dl_recon_cov.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot the function, the prediction and the 95% confidence interval
fig, ax = plt.subplots()
plt.tick_params(labelsize=14, color='purple')
#plt.errorbar(x_gapp, y_gapp, e, fmt='o', color='red', markersize=5, label='Data')


# Modelos tipo-LCDM

plt.plot(x, DL(x, 0.339, 0.672, -1, 0, -19.416, 0)*67.2/c, color='blue', linewidth = 1, linestyle='--', label='$\Lambda$CDM')
plt.plot(x, DL(x, 0.306, 0.673, -0.887, 0, -19.412, 0)*67.3/c, color='red', linewidth = 1, label='$\omega$CDM')
plt.plot(x, DL(x, 0.320, 0.674, -0.876, -0.07, -19.407, 0)*67.4/c, color='saddlebrown', linewidth = 1, label='$\omega_0 \omega_a$CDM')
plt.plot(x, DL(x, 0.317, 0.613, -1, 0, -19.416, 0.2)*61.3/c, color='green', label='$\Omega_k$-CDM', linewidth = 1)


# Modelo f(Q)
#plt.plot(x, DL(x, 0.347, 0.669, -1, 0, -19.423, 0)/c, color='maroon', label='$F_1(Q)$', linewidth = 2)

plt.plot(xi, y_pred*73/c, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")


plt.plot()
plt.fill(np.concatenate([xi, xi[::-1]]),
          np.concatenate([y_pred*73/c - 1.9600 * sigma*73/c,
                        (y_pred*73/c + 1.9600 * sigma*73/c)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')

plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred*73/c - 1.00 * sigma*73/c,
                        (y_pred*73/c + 1.00 * sigma*73/c)[::-1]]),
        alpha=.5, color = 'dodgerblue', ec='None')


    

# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$D_L(z)H_0/c$', fontsize=15)
plt.legend(loc='upper left', prop={'size':10})
plt.ylim(0,5)
plt.xlim(0,2.27)
plt.savefig('dl_comparação_RG_cov.pdf', format='pdf', bbox_inches='tight')
plt.show()


#diff = (y_pred - DL(x, 0.339, 0.672, -1, 0, -19.416, 0)/c) / y_pred

#plt.plot(x, diff)
#plt.show()


# Modelos f(R)

ti = 0.17
tf = 1.0
p = 1000
O_r0 = 0
Delta = 10**(-7)
t_span = [ti, tf]
t = np.linspace(ti, tf, p)
z = (1/t) - 1



# --------------- Dados SNe --------------- #
df= pd.read_csv('/home/usuario/Downloads/DataRelease-main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat', sep=' ')
#df= pd.read_csv('/content/Pantheon+SH0ES.dat', sep=' ')
df=df.sort_values("zCMB", ascending=False)
df = df.drop_duplicates(subset='zCMB', keep='first')
df['acmb'] = 1/(1+df["zCMB"])
a_sn = df['acmb']
y_sn = df["m_b_corr"]

mcov = np.loadtxt('/home/usuario/Downloads/DataRelease-main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
#mcov = np.loadtxt('/content/Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = np.reshape(mcov, (1701, 1701))
Icov=np.linalg.inv(cov)
Icov



# Starobinsky n = 1


def yH_S(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]
    
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
    return [YH, dYH]


# m_B(z) para o modelo HS:
def solyH1_S(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_S = solve_ivp(yH_S, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    yH1 = sol_yH_S.y[0]
    return yH1

def H1_S(H0, O_m0, mu):
    yH = solyH1_S(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    H_S = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H_S

# condições iniciais de acordo com GR/LCDM
def Int_GR(H0, O_m0):
  ms = H0*np.sqrt(O_m0)
  O_L = 1 - O_m0
  hyper = sp.special.hyp2f1
  O_m = O_m0*t**(-3)
  H_GR = H0*np.sqrt(O_m + O_L)
  
  w = - 1

  x1 = - 1/(6*w)
  x2 = 1 + x1
  x3 = 1 - (1/O_m0)
  x4 = 1 - ((1/O_m)*(H_GR/H0)**2)

  IntGR = (2/ms)*(hyper(1/2,x1,x2,x3) - np.sqrt(t)*hyper(1/2,x1,x2,x4))
  return IntGR

# Integral em MG:
def IntMG(H0, O_m0, mu):
  y = 1/((t**2)*H1_S(H0, O_m0, mu))
  IntMG = Int_GR(H0, O_m0)[0] + integrate.cumulative_trapezoid(y, t, initial=0)
  return IntMG

# Primitiva:
def Int_MG(H0, O_m0, mu):
  Int_MG = IntMG(H0, O_m0, mu)[999] - IntMG(H0, O_m0, mu)
  return Int_MG

# Interpolação com os dados:
def IntMG_interp(H0, O_m0, mu):
  x_interp = t
  y_interp = np.interp(x_interp, t, Int_MG(H0, O_m0, mu))
  return y_interp

# Distância-luminosidade:
def Lumi_MG(H0, O_m0, mu):
  dL_MG = (299792.458/t)*(IntMG_interp(H0, O_m0, mu))
  return dL_MG

# Magnitude relativa:
def mag_MG(a_sn, H0, O_m0, Mb, mu):
  mb_MG = 5*np.log10(Lumi_MG(H0, O_m0, mu)) + 25 + Mb
  return mb_MG


#plt.plot(df["zCMB"], mag_MG(a_sn, 70, 0.3, -19.3, 0.4), 'blue')

def DL_MG(t, H0, O_m0, Mb, mu):
    
    return 10 ** ((mag_MG(t, H0, O_m0, Mb, mu)-Mb-25) / 5)



A6 = z, DL_MG(t, 67.1, 0.328, -19.407, 0.589)/c

#np.savetxt('dl_s1_mcmc_william.csv', np.transpose(A6), delimiter=', ')


# Starobinsky n = 2

def yH_S2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]

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
    return [YH, dYH]


# m_B(z) para o modelo HS:
    
def solyH2_S(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_S2 = solve_ivp(yH_S2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    yH2 = sol_yH_S2.y[0]
    return yH2

def H2_S(H0, O_m0, mu):
    yH2 = solyH2_S(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    H_S2 = np.sqrt(ms*(yH2 + (1/t)**3 + chi*(1/t)**4))
    return H_S2

# condições iniciais de acordo com GR/LCDM
def Int_GR(H0, O_m0):
  ms = H0*np.sqrt(O_m0)
  O_L = 1 - O_m0
  hyper = sp.special.hyp2f1
  O_m = O_m0*t**(-3)
  H_GR = H0*np.sqrt(O_m + O_L)
  
  w=-1

  x1 = - 1/(6*w)
  x2 = 1 + x1
  x3 = 1 - (1/O_m0)
  x4 = 1 - ((1/O_m)*(H_GR/H0)**2)

  IntGR = (2/ms)*(hyper(1/2,x1,x2,x3) - np.sqrt(t)*hyper(1/2,x1,x2,x4))
  return IntGR

# Integral em MG:
def IntMG2(H0, O_m0, mu):
  y = 1/((t**2)*H2_S(H0, O_m0, mu))
  IntMG2 = Int_GR(H0, O_m0)[0] + integrate.cumulative_trapezoid(y, t, initial=0)
  return IntMG2

# Primitiva:
def Int_MG2(H0, O_m0, mu):
  Int_MG2 = IntMG2(H0, O_m0, mu)[999] - IntMG2(H0, O_m0, mu)
  return Int_MG2

# Interpolação com os dados:
def IntMG2_interp(H0, O_m0, mu):
  x_interp = t
  y_interp = np.interp(x_interp, t, Int_MG2(H0, O_m0, mu))
  return y_interp

# Distância-luminosidade:
def Lumi_MG2(H0, O_m0, mu):
  dL_MG2 = (299792.458/t)*(IntMG2_interp(H0, O_m0, mu))
  return dL_MG2

# Magnitude relativa:
def mag_MG2(a_sn, H0, O_m0, Mb, mu):
  mb_MG2 = 5*np.log10(Lumi_MG2(H0, O_m0, mu)) + 25 + Mb
  return mb_MG2


def DL_MG2(t, H0, O_m0, Mb, mu):
    
    return 10 ** ((mag_MG2(t, H0, O_m0, Mb, mu)-Mb-25) / 5)

#plt.plot(df["zCMB"], DL_MG2(t, 70, 0.3, -19.3, 0.4)/c, 'red')


A7 = z, DL_MG2(t, 66.4, 0.350, -19.416, 1.047)/c

#np.savetxt('dl_s2_mcmc_william.csv', np.transpose(A7), delimiter=', ')




# Hu-Sawicki n = 1

def yH_HS(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]        # YH = dyH/da

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

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
    return [YH, dYH]


# m_B(z) para o modelo HS:
def solyH1_HS(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_HS = solve_ivp(yH_HS, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, mu))
    yHS1 = sol_yH_HS.y[0]
    return yHS1

def H1_HS(H0, O_m0, mu):
    yH = solyH1_HS(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    H_HS = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H_HS

# Condições iniciais = GR:
def Int_GR(H0, O_m0):
  ms = H0*np.sqrt(O_m0)
  O_L = 1 - O_m0
  hyper = sp.special.hyp2f1
  O_m = O_m0*t**(-3)
  H_GR = H0*np.sqrt(O_m + O_L)
  
  w=-1

  x1 = - 1/(6*w)
  x2 = 1 + x1
  x3 = 1 - (1/O_m0)
  x4 = 1 - ((1/O_m)*(H_GR/H0)**2)

  IntGR = (2/ms)*(hyper(1/2,x1,x2,x3) - np.sqrt(t)*hyper(1/2,x1,x2,x4))
  return IntGR

# Integral em MG:
def IntMG3(H0, O_m0, mu):
  y = 1/((t**2)*H1_HS(H0, O_m0, mu))
  IntMG3 = Int_GR(H0, O_m0)[0] + integrate.cumulative_trapezoid(y, t, initial=0)
  return IntMG3

# Primitiva:
def Int_MG3(H0, O_m0, mu):
  Int_MG3 = IntMG3(H0, O_m0, mu)[999] - IntMG3(H0, O_m0, mu)
  return Int_MG3

# Interpolação com os dados:
def IntMG3_interp(H0, O_m0, mu):
  x_interp = t
  y_interp = np.interp(x_interp, t, Int_MG3(H0, O_m0, mu))
  return y_interp

# Distância-luminosidade:
def Lumi_MG3(H0, O_m0, mu):
  dL_MG3 = (299792.458/t)*(IntMG3_interp(H0, O_m0, mu))
  return dL_MG3

# Magnitude relativa:
def mag_MG3(a_sn, H0, O_m0, Mb, mu):
  mb_MG3 = 5*np.log10(Lumi_MG3(H0, O_m0, mu)) + 25 + Mb
  return mb_MG3


def DL_MG3(t, H0, O_m0, Mb, mu):
    
    return 10 ** ((mag_MG3(t, H0, O_m0, Mb, mu)-Mb-25) / 5)

#plt.plot(df["zCMB"], DL_MG3(t, 70, 0.3, -19.3, 82)/c, 'green')



A8 = z, DL_MG3(t, 69.7, 0.269, -19.402, 82)/c

#np.savetxt('dl_hs1_mcmc_william.csv', np.transpose(A8), delimiter=', ')



# Hu-Sawicki n = 2

# Modelo HS-f(R):
def yH_HS2(t, y, H0, O_m0, mu):
    yH = y[0]
    YH = y[1]        # YH = dyH/da

    O_m = O_m0*t**(-3)
    O_r = O_r0*t**(-4)

    Lbd = 3*(H0**2)*(1 - O_m0)

    chi = O_r0/O_m0
    ms = (H0**2)*O_m0

    R = 3*ms*(4*yH + t*YH + t**(-3))
    
    n=2

    A = (R**n) + (mu**(2*n))
    B = (R**n)
    C = (n+1)*(R**(2*n - 2)) - (n- 1)*(mu**(2*n))*(R**(n-2))
    D = A**3

    R0 = ms*((12/O_m0) - 9)
    M2 = R0/Delta

    f = R - 2*Lbd*(B/A) + (R**2/(6*M2))
    fR = 1 - 2*n*Lbd*(mu**(2*n))*(B/A)*(1/(R*A)) + (R/(3*M2))
    fRR = 2*n*Lbd*(mu**(2*n))*(C/D) + (1/(3*M2))

    yaux1 = yH + t**(-3) + chi*t**(-4)
    yaux2 = t**(-3) + 2*chi*t**(-4)

    j1 = 4 + (1/yaux1)*(1-fR)/(6*ms*fRR)
    j2 = (1/yaux1)*(2-fR)/(3*ms*fRR)
    j3 = - 3*t**(-3) - (((1-fR)*yaux2 + (R-f)/(3*ms))/yaux1)*(1/(6*ms*fRR))
    J1 = (1/t)*(1 + j1)
    J2 = (1/t)*(j2/t)
    J3 = (1/t)*(j3/t)

    dYH = - J1*YH - J2*yH - J3
    return [YH, dYH]


# m_B(z) para o modelo HS:
def solyH2_HS(H0, O_m0, mu):
    y0 = [(1-O_m0)/O_m0, 0]
    sol_yH_HS2 = solve_ivp(yH_HS2, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-10), args=(H0, O_m0, mu))
    yHS2 = sol_yH_HS2.y[0]
    return yHS2

def H2_HS(H0, O_m0, mu):
    yH = solyH2_HS(H0, O_m0, mu)
    chi = O_r0/O_m0
    ms = (H0**2)*O_m0
    H_HS2 = np.sqrt(ms*(yH + (1/t)**3 + chi*(1/t)**4))
    return H_HS2

# Condições iniciais = GR:
def Int_GR(H0, O_m0):
  ms = H0*np.sqrt(O_m0)
  O_L = 1 - O_m0
  hyper = sp.special.hyp2f1
  O_m = O_m0*t**(-3)
  H_GR = H0*np.sqrt(O_m + O_L)
  
  w=-1

  x1 = - 1/(6*w)
  x2 = 1 + x1
  x3 = 1 - (1/O_m0)
  x4 = 1 - ((1/O_m)*(H_GR/H0)**2)

  IntGR = (2/ms)*(hyper(1/2,x1,x2,x3) - np.sqrt(t)*hyper(1/2,x1,x2,x4))
  return IntGR

# Integral em MG:
def IntMG4(H0, O_m0, mu):
  y = 1/((t**2)*H2_HS(H0, O_m0, mu))
  IntMG4 = Int_GR(H0, O_m0)[0] + integrate.cumulative_trapezoid(y, t, initial=0)
  return IntMG4

# Primitiva:
def Int_MG4(H0, O_m0, mu):
  Int_MG4 = IntMG4(H0, O_m0, mu)[999] - IntMG4(H0, O_m0, mu)
  return Int_MG4

# Interpolação com os dados:
def IntMG4_interp(H0, O_m0, mu):
  x_interp = t
  y_interp = np.interp(x_interp, t, Int_MG4(H0, O_m0, mu))
  return y_interp

# Distância-luminosidade:
def Lumi_MG4(H0, O_m0, mu):
  dL_MG4 = (299792.458/t)*(IntMG4_interp(H0, O_m0, mu))
  return dL_MG4

# Magnitude relativa:
def mag_MG4(a_sn, H0, O_m0, Mb, mu):
  mb_MG4 = 5*np.log10(Lumi_MG4(H0, O_m0, mu)) + 25 + Mb
  return mb_MG4


def DL_MG4(t, H0, O_m0, Mb, mu):
    
    return 10 ** ((mag_MG4(t, H0, O_m0, Mb, mu)-Mb-25) / 5)

#plt.plot(df["zCMB"], DL_MG4(t, 70, 0.3, -19.3, 120)/c, 'brown')


A9 = z, DL_MG4(t, 66.8, 0.327, -19.400, 104)/c

#np.savetxt('dl_hs2_mcmc_william.csv', np.transpose(A9), delimiter=', ')




# Modelo Appleby-Battye


# m_B(z) para o modelo R2-AB:
def Hubble(t, y, H0, O_m0, b):
  H  =  y[0]
  dH  = y[1]
  ddH = y[2]

  O_m = O_m0*t**(-3)
  O_r = O_r0*t**(-4)

  R_vac = 12*H0**2
  e_AB = R_vac/np.log(1 + np.exp(2*b))
  M2 = e_AB/Delta

  R = 6*H*(2*H + t*dH)
  alpha = (R/e_AB) - b
  if abs(alpha) < 15:
    s = 1/np.cosh(alpha)
  else:
    s = 0

  A = H0*H0*(3*O_m + 4*O_r)
  B = t*H*dH*(np.tanh(alpha) + 1)
  C = 4*t*H*H*dH*(t*dH + 2*H)/M2
  if abs(alpha) < 15:
    D = ((6*t*t*H*H)/e_AB)*(((5*dH/t) + (dH*dH/H) +ddH)**2)*np.tanh(alpha)*s*s
    E = 6*(t**3)*(H**3)*((1/(3*M2)) + (0.5*s*s/e_AB))
  else:
    D = 0
    E = 6*(t**3)*(H**3)/(3*M2)

  F = (11*dH*dH/(t*H)) + (dH*dH*dH/(H*H)) + (6*ddH/t) + (4*dH*ddH/H)

  dddH = - ((A + B + C - D)/E) - F
  return [dH, ddH, dddH]

def solution(H0, O_m0, b):
  t_span = [0.17, 1]
  ti = 0.17
  O_mi = O_m0*ti**(-3)
  O_ri = O_r0*ti**(-4)
  O_L = 1 - O_m0
  Hi = H0*np.sqrt(O_mi + O_ri + O_L)
  dHi = - ((H0*H0)/(2*ti*Hi))*(3*O_mi + 4*O_ri)
  ddHi = 0.5*(H0/(ti*Hi))**2*(Hi + ti*dHi)*(3*O_mi + 4*O_ri) + 0.5*(H0/(ti*Hi))**2*Hi*(9*O_mi + 16*O_ri)
  y0 = [Hi, dHi, ddHi]

  sol = solve_ivp(Hubble, t_span, y0, t_eval=t, method='LSODA', rtol = 10**(-6), args=(H0, O_m0, b))
  H = sol.y[0]
  return H



# Condições iniciais = GR:
def Int_GR(H0, O_m0):
  ms = H0*np.sqrt(O_m0)
  O_L = 1 - O_m0
  hyper = sp.special.hyp2f1
  O_m = O_m0*t**(-3)
  H_GR = H0*np.sqrt(O_m + O_L)
  
  w=-1

  x1 = - 1/(6*w)
  x2 = 1 + x1
  x3 = 1 - (1/O_m0)
  x4 = 1 - ((1/O_m)*(H_GR/H0)**2)

  IntGR = (2/ms)*(hyper(1/2,x1,x2,x3) - np.sqrt(t)*hyper(1/2,x1,x2,x4))
  return IntGR

# Integral em MG:
def IntMG5(H0, O_m0, b):
  y = 1/((t**2)*solution(H0, O_m0, b))
  IntMG5 = Int_GR(H0, O_m0)[0] + integrate.cumulative_trapezoid(y, t, initial=0)
  return IntMG5

# Primitiva:
def Int_MG5(H0, O_m0, b):
  Int_MG5 = IntMG5(H0, O_m0, b)[999] - IntMG5(H0, O_m0, b)
  return Int_MG5

# Interpolação com os dados:
def IntMG5_interp(H0, O_m0, b):
  x_interp = t
  y_interp = np.interp(x_interp, t, Int_MG5(H0, O_m0, b))
  return y_interp

# Distância-luminosidade:
def Lumi_MG5(H0, O_m0, b):
  dL_MG5 = (299792.458/t)*(IntMG5_interp(H0, O_m0, b))
  return dL_MG5

# Magnitude relativa:
def mag_MG5(a_sn, H0, O_m0, Mb, b):
  mb_MG5 = 5*np.log10(Lumi_MG5(H0, O_m0, b)) + 25 + Mb
  return mb_MG5



def DL_MG5(t, H0, O_m0, Mb, b):
    
    return 10 ** ((mag_MG5(t, H0, O_m0, Mb, b)-Mb-25) / 5)

#plt.plot(df["zCMB"], DL_MG5(t, 70, 0.3, -19.3, 1.98)/c, 'gray')


A10 = z, DL_MG5(t, 67.3, 0.332, -19.416, 1.98)/c

#np.savetxt('dl_ab_mcmc_william.csv', np.transpose(A10), delimiter=', ')




# modelos de gravidade modificada

fig, ax = plt.subplots()
plt.tick_params(labelsize=14, color='purple')


plt.plot(z, DL_MG(t, 67.1, 0.328, -19.407, 0.589)*67.1/c, color='orange', linewidth = 1, label='Starobinsky ($n=1$)', linestyle = '-.')
plt.plot(z, DL_MG2(t, 66.4, 0.350, -19.416, 1.047)*66.4/c, color='deeppink', linewidth = 1, label='Starobinsky ($n=2$)', linestyle = '-.')
plt.plot(z, DL_MG3(t, 69.7, 0.269, -19.402, 82)*69.7/c, color='purple', linewidth = 1, label='Hu-Sawicki ($n=1$)')
plt.plot(z, DL_MG4(t, 66.8, 0.327, -19.400, 104)*66.8/c, color='magenta', linewidth = 1, label='Hu-Sawicki ($n=2$)')
plt.plot(z, DL_MG5(t, 67.3, 0.332, -19.416, 1.98)*67.3/c, color='black', linewidth = 1, label='$R^2$_AB')
plt.plot(x, DL(x, 0.347, 0.669, -1, 0, -19.423, 0)*66.9/c, color='maroon', label='$F(Q)$', linewidth = 1)


plt.plot(xi, y_pred*73/c, color = 'darkblue', linewidth=2, label='GP', linestyle="dotted")


plt.plot()
plt.fill(np.concatenate([xi, xi[::-1]]),
          np.concatenate([y_pred*73/c - 1.9600 * sigma*73/c,
                        (y_pred*73/c + 1.9600 * sigma*73/c)[::-1]]),
         alpha=.5, color = 'lightblue', ec='None')

plt.fill(np.concatenate([xi, xi[::-1]]),
         np.concatenate([y_pred*73/c - 1.00 * sigma*73/c,
                        (y_pred*73/c + 1.00 * sigma*73/c)[::-1]]),
        alpha=.5, color = 'dodgerblue', ec='None')


# legenda, label e título
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$D_L(z)H_0/c$', fontsize=15)
plt.legend(loc='upper left', prop={'size':10})
plt.ylim(0,5)
plt.xlim(0,2.27)
plt.savefig('dl_comparação_MG_cov.pdf', format='pdf', bbox_inches='tight')
plt.show()

