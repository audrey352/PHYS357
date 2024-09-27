import numpy as np
import scipy as sp

# constants
hbar = sp.constants.hbar

# Functions
def commutation(A,B):
    return A@B - B@A

# Momentum operators in the z basis
Jz = hbar/2 * np.array([[1,0],[0,-1]])
Jx = hbar/2 * np.array([[0,1],[1,0]])
Jy = hbar/2 * np.array([[0,-1j],[1j,0]])

# [Jx,Jy]
Jx_Jy = commutation(Jx,Jy)
print('Jx_Jy:\n', Jx_Jy==1j*hbar*Jz)
# [Jy,Jz]
Jy_Jz = commutation(Jy,Jz)
print('Jy_Jz:\n', Jy_Jz==1j*hbar*Jx)
# [Jz,Jx]
Jz_Jx = commutation(Jz,Jx)
print('Jz_Jx:\n', Jz_Jx==1j*hbar*Jy)
