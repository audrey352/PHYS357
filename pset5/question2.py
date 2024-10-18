import numpy as np
import astropy.units as u
import scipy.constants as const

hbar = const.hbar

# Ping pong ball info
m = 2.7 * u.g
r = 20 * u.mm
w = 10 *u.rad/u.s

# Angular momentum
I = m*r**2
L = I*w
print(f"L = {L}")
L = L.value

# Solve for j
j = np.roots([hbar**2, hbar**2, -L**2])
j = j[j>0][0]
print(f"j = {j}")

# Momentum 
Jz = hbar * j
J_xy_plane = hbar* np.sqrt(L/hbar)
print(f"J_xy_plane = {J_xy_plane}")

# Uncertainty
sigma = J_xy_plane/Jz
print(f"sigma = {sigma}")


