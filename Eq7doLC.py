#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:42:01 2024

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.2656, Omega_b=0.0494, h=0.6727, sigma8=0.8120, n_s=0.9649)


a = np.linspace(0.17, 1, 1000)  # fator de escala

# definindo o redshift em função do fator de escala
z = (1/a) - 1.


from scipy.integrate import cumtrapz


# constantes
Om0 = 0.3
h0 = 70
Ow0 = 0.7
w0 = -0.957
wa = -0.29

# definindo H
H = h0*np.sqrt(Om0*(a**(-3)) + Ow0 )

# definindo O_m
Om = Om0*(a**(-3)) / (H**2/h0**2)

# definindo Ow
Ow = Ow0*(a**(-3*(w0 + wa*(1-a)) ) )


plt.plot(z, Ow, label='$\omega$CDM', color='blue')
plt.xlabel('$z$')
plt.ylabel('$\Omega_\Lambda(a)$')
plt.legend()
plt.show()


# plotando a eq 7

# definindo a integral
I3 = cumtrapz( (a**(3/2)*Ow), x=a, initial=0.000001)

G7 = - (1/2)*Ow - (1/4) * (a**(-5/2)) * I3

G7 = G7/G7[-1]

# vamos plotar a equação

plt.plot(a, G7, color='orange')
plt.xlabel('$a$')
plt.ylabel('$G(a)$')
plt.show()



# definindo O_m
Ow_lcdm = 1-Om


# definindo a integral
I4 = cumtrapz( (a**(3/2)*Ow_lcdm), x=a, initial=0.000001)

G7_lcdm = - (1/2)*Ow_lcdm - (1/4) * (a**(-5/2)) * I4

G7_lcdm = G7_lcdm/G7_lcdm[-1]


# vamos plotar a equação
plt.plot(a, G7_lcdm, color='blue', label='$\Lambda$CDM')
plt.xlabel('a')
plt.ylabel('$G(a)$')
plt.legend()
plt.show()


# vamos plotar a equação
plt.plot(a, G7, color='orange', label='$\omega$CDM')
plt.plot(a, G7_lcdm, color='blue', label='$\Lambda$CDM')
plt.xlabel('a')
plt.ylabel('$G(a)$')
plt.legend()
plt.show()




# NOVA FORMA - SUGESTÃO FELIPE

# definindo a integral
#I5 = cumtrapz( (a**(3/2)*Ow_lcdm), x=1, initial=0.0)

#G7_lcdm2 = - (1/2)*Ow_lcdm - (1/4) * (a**(-5/2)) * I5

#G7_lcdm2 = G7_lcdm2/G7_lcdm2[-1]


# vamos plotar a equação
#plt.plot(a, G7_lcdm2, color='red', label='$\Lambda$CDM')
#plt.xlabel('a')
#plt.ylabel('$G(a)$')
#plt.legend()
#plt.show()
