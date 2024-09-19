import numpy as np
import scipy as sp

# constants
hbar = sp.constants.hbar

# state in its own basis
np_in_n = np.array([1,0])
nm_in_n = np.array([0,1])
Jnn = hbar/2*np.array([[1,0],[0,-1]])  # Jn in n basis (n=x,y,z)


# 6 a)
print('6 a)')
# x kets
xp_in_z = np.array([1,1])/np.sqrt(2)
xm_in_z = np.array([1,-1])/np.sqrt(2)
# y kets
yp_in_z = np.array([1,1j])/np.sqrt(2)
ym_in_z = np.array([1,-1j])/np.sqrt(2)

# Rotation matrices
R_ztox = np.outer(np_in_n,np.conj(xp_in_z.T)) + np.outer(nm_in_n,np.conj(xm_in_z.T))  # z to x basis
# print('Rotation z to x:\n', R_ztox)
# print(f'Checking Rz-x:\n {R_ztox@xp_in_z}, {xp_in_x}, {R_ztox@xm_in_z},{xm_in_x}')
R_ztoy = np.outer(np_in_n,np.conj(yp_in_z.T)) + np.outer(nm_in_n,np.conj(ym_in_z.T))  # z to y basis
# print('Rotation z to y:\n', R_ztoy)
# print(f'Checking Rz-y:\n {R_ztoy@yp_in_z},{yp_in_y}, {R_ztoy@ym_in_z}, {ym_in_y}')

# Momentum operators
Jxz = np.conj(R_ztox.T) @ Jnn @ R_ztox  # Jx in z basis
Jyz = np.conj(R_ztoy.T) @ Jnn @ R_ztoy  # Jy in z basis
print('Jxz:\n', Jxz)
print('Jyz:\n', Jyz)

# Hermitian check
print(f'Checking hermitian:\n(Jxz)\n {np.conj(Jxz.T)==Jxz}\n(Jyz)\n {np.conj(Jyz.T)==Jyz}')


# 6 b)
print('\n6 b)')
# converting the x and z to the y basis
# x kets
xp_in_y = R_ztoy@xp_in_z
xm_in_y = R_ztoy@xm_in_z
# z kets
zp_in_y = R_ztoy@np_in_n
zm_in_y = R_ztoy@nm_in_n

# Rotation matrices
R_ytox = np.outer(np_in_n,np.conj(xp_in_y.T)) + np.outer(nm_in_n,np.conj(xm_in_y.T))  # y to x basis
R_ytoz = np.conj(R_ztoy.T)  # y to z basis  (inverse of R_ztoy)

# Momentum operators
# Momentum operators
Jxy = np.conj(R_ytox.T) @ Jnn @ R_ytox  # Jx in y basis
Jzy = np.conj(R_ytoz.T) @ Jnn @ R_ytoz  # Jz in y basis
print('Jxy:\n', Jxy)
print('Jzy:\n', Jzy)

# Hermitian check
print(f'Checking hermitian:\n(Jxy)\n {np.conj(Jxy.T)}\n(Jzy)\n {np.conj(Jzy.T)==Jzy}')

