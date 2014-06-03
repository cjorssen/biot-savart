# -*- coding: utf-8 -*-

import numpy as np

# mu0 = 12.566370614e-7
mu0 = 4 * np.pi
I = 1.

def circular_loop(R = 1e-2, C = [0., 0., 0.], N = 200):
    XP = R * np.cos(np.linspace(0, 2*np.pi, N)) - C[0]
    ZP = R * np.sin(np.linspace(0, 2*np.pi, N)) - C[1]
    YP = np.zeros(N) + C[2]
    return XP, YP, ZP
    
def biot_savart(xP, yP, zP, XM, YM, ZM):
    """
    xP, yP et zP sont des ndarrays à une dimension contenant
    les NP coordonnées des points P représentant la distribution
    de courant.
    
    XM, YM sont des ndarrays à deux dimensions obtenus typiquement
    par 
    XM, YM = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY))
    
    ZM est un ndarray à deux dimensions représentant la cote des points de la
    grille XM, YM (typiquement rempli de 0 : observation dans le plan z = 0).
    
    La fonction renvoit deux ndarrays de dimension (NX, NY) qui sont les
    composantes (utiles) du champ dans le plan d'observation.
    """
    
    P = np.array([xP, yP, zP])
    # P.shape = (3, NP)
    
    # Vecteur déplacement élémentaire le long de la distribution
    vec_dlP = P[:,1:] - P[:,:-1]
    # vec_dlP.shape = (3, NP-1)

    # Localisation du point courant de la distribution
    mid_dlP = (P[:,1:] + P[:,:-1]) / float(2)
    # mid_dlP.shape = (3, NP-1)
    
    XPM = XM[:,:,np.newaxis] - mid_dlP[0,:]
    YPM = YM[:,:,np.newaxis] - mid_dlP[1,:]
    ZPM = ZM[:,:,np.newaxis] - mid_dlP[2,:]
    # XPM.shape = YPM.shape = ZPM.shape = (NX, NY, NP)
    
    PMcubed = (XPM**2 + YPM**2 + ZPM**2)**(3/2.)
    # PMcubed.shape = (NX, NY, NP)

    # Éviter la division par zéro (trop proche des sources)
    PMcubed[PMcubed < 1e-6] = np.nan

    Xdl_cross_PM = - YPM * vec_dlP[2,:] + ZPM * vec_dlP[1,:]
    Ydl_cross_PM = - ZPM * vec_dlP[0,:] + XPM * vec_dlP[2,:]
    Zdl_cross_PM = - XPM * vec_dlP[1,:] + YPM * vec_dlP[0,:]
    # Xdl_cross_PM.shape = Ydl_cross_PM.shape = Zdl_cross_PM.shape = 
    # (NX, NY, NP)
    
    Xdl_cross_PM_over_PMcubed = Xdl_cross_PM / PMcubed
    Ydl_cross_PM_over_PMcubed = Ydl_cross_PM / PMcubed
    Zdl_cross_PM_over_PMcubed = Zdl_cross_PM / PMcubed
    
    BX = np.sum(Xdl_cross_PM_over_PMcubed, axis = 2)
    BY = np.sum(Ydl_cross_PM_over_PMcubed, axis = 2)
    BZ = np.sum(Zdl_cross_PM_over_PMcubed, axis = 2)
    # BX.shape = BY.shape = BZ.shape = (NX, NY)
    
    BNorme = (BX**2 + BY**2 + BZ**2)**(.5)
    # BNorme.shape = (NX, NY)

    return mu0 * I * BX/(4 * np.pi), mu0 * I * BY/(4 * np.pi),\
           mu0 * I * BNorme/(4 * np.pi)

def Bzloop(z):
    return mu0 * I * R**2/(2 * (R**2 + z**2)**(3/2.))

import matplotlib.pyplot as plt
import matplotlib.cm as cm

R = 1

xP, yP, zP = circular_loop(R, [0., 0., 0.])

# Grille de NX*NY points
NX = 20
NY = 20

# Coordonnées min et max des points de la grille
xmax = 2 * R
xmin = -xmax
ymax = 2 * R
ymin = -ymax

# Les coordonnées des points de la grille sont répartis uniformément
# sur [xmin, xmax]x[ymin, ymax]
xM = np.linspace(xmin, xmax, NX)
yM = np.linspace(ymin, ymax, NY)

# Création de la grille
XM, YM = np.meshgrid(xM, yM)

# On impose la cote de la grille : z = 0
ZM = np.zeros((yM.size, xM.size))

# Calcul du champ magnétique
# La norme de B (BNorme) permet de normer le champ
BX, BY, BNorme = biot_savart(xP, yP, zP, XM, YM, ZM)

# Si on norme le champ, les flèches ont toutes la même taille (c'est plus
# joli, mais on perd de l'information).
plt.quiver(XM, YM, BX/BNorme, BY/BNorme, pivot = 'middle', units = 'width')
# Du coup, on peut retrouver cette information avec une "troisième" dimension
plt.imshow(BNorme, interpolation = 'bilinear', origin = 'lower',
                   cmap = cm.jet, extent = (xmin, xmax, ymin, ymax))
# Pour le fun
plt.contour(XM, YM, BNorme, 10)
plt.show()