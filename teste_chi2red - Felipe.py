#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:38:55 2024

@author: felipe
"""

import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

N = 35

# dados originais de fsig8

data = recon = np.genfromtxt('/home/usuario/Documentos/Códigos/GaPP SNIa/fsig8_bold_data.dat')

z = data[:, 0]
f = data[:, 1]

f = f[z<=1]
z = z[z<=1]


recon = np.genfromtxt('/home/usuario/Documentos/Códigos/Report do artigo/fs8_recon_gapp.csv',delimiter=', ')

x = recon[:, 0]
y = recon[:, 1]
e = recon[:, 2]

f_recon = CubicSpline(x, y)

e_recon = CubicSpline(x, e)

V_r =  f_recon(z)

sig_V = e_recon(z)


#####################################

model_1 = np.genfromtxt('/home/usuario/Documentos/Códigos/Estudo do contraste/Teste chi2/fs8_lcdm_pontos_original.csv',delimiter=', ')

x1 = model_1[:, 0]
y1 = model_1[:, 1]

p1 = 3

# Obtendo os índices que ordenam x
indices = np.argsort(x1)

# Reordenando x e y com base nesses índices
x_sorted = x1[indices]
y_sorted = y1[indices]

x1 = x_sorted
y1 = y_sorted

# plt.plot(x1, y1, color='red')


f_y1 = CubicSpline(x1, y1)


#####################################

model_2 = np.genfromtxt('/home/usuario/Documentos/Códigos/Estudo do contraste/Teste chi2/fs8_wcdm_pontos_original.csv',delimiter=',')

x2 = model_2[:, 0]
y2 = model_2[:, 1]

p2 = 3

indices_2 = np.argsort(x2)

x2_sorted = x2[indices_2]
y2_sorted = y2[indices_2]

x2 = x2_sorted
y2 = y2_sorted

f_y2 = CubicSpline(x2, y2)

####################################

model_3 = np.genfromtxt('/home/usuario/Documentos/Códigos/Estudo do contraste/Teste chi2/fs8_AB_pontos_original.csv',delimiter=',')

x3 = model_3[:, 0]
y3 = model_3[:, 1]

p3 = 4

indices_3 = np.argsort(x3)

x3_sorted = x3[indices_3]
y3_sorted = y3[indices_3]

x3 = x3_sorted
y3 = y3_sorted

f_y3 = CubicSpline(x3, y3)


############################# teste chi2_red

X2r_model_1 = (1 / (N - p1)) * np.sum(((f_y1(z) - V_r) / sig_V)**2)

print(X2r_model_1)


X2r_model_2 = (1 / (N - p2)) * np.sum(((f_y2(z) - V_r) / sig_V)**2)

print(X2r_model_2)


X2r_model_3 = (1 / (N - p3)) * np.sum(((f_y3(z) - V_r) / sig_V)**2)

print(X2r_model_3)

