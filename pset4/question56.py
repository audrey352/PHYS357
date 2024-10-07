import numpy as np
from scipy.constants import hbar

# Function
def commutator(A,B):
    return A@B - B@A

# Define the matrices for spin-1 basis
Jx = 1/np.sqrt(2)*np.array([[0,1,0],[1,0,1],[0,1,0]])
Jy = 1/np.sqrt(2)*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
Jz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
J_squared = Jx@Jx + Jy@Jy + Jz@Jz

# Question 5
# Calculate the commutators
# print(f"[Jx,Jy]\n{commutator(Jx,Jy)}")
# print(f"[Jy,Jz]\n{commutator(Jy,Jz)}")
# print(f"[Jx,Jz]\n{commutator(Jx,Jz)}")

# print(f"[Jx,J^2]\n{commutator(Jx,J_squared)}")
# print(f"[Jy,J^2]\n{commutator(Jy,J_squared)}")
# print(f"[Jz,J^2]\n{commutator(Jz,J_squared)}")


# Question 6
# Get the eigenstates and eigenvalues
# *eigenstates are the columns of the eigenvector matrix
eigvals_Jx, eigvec_Jx = np.linalg.eigh(Jx)
eigvals_Jy, eigvec_Jy = np.linalg.eigh(Jy)
eigvals_Jx, eigvec_Jx = np.flipud(eigvals_Jx), np.fliplr(eigvec_Jx)
eigvals_Jy, eigvec_Jy = np.flipud(eigvals_Jy), np.fliplr(eigvec_Jy)
print(f"Eigenvalues of Jx: {eigvals_Jx}")
print(f"Eigenstates of Jx:\n{eigvec_Jx}")
print(f"Eigenvalues of Jy: {eigvals_Jy}")
print(f"Eigenstates of Jy:\n{eigvec_Jy}")

# Raising & lowering operators in the Jx basis
J_plus_x = Jy + 1j*Jz
J_minus_x = np.conj(J_plus_x).T
print(f"J_plus:\n{J_plus_x}")
print(f"J_minus:\n{J_minus_x}")
# Raising & lowering operators in the Jy basis
J_plus_y = Jz + 1j*Jx
J_minus_y = np.conj(J_plus_y).T
print(f"J_plus:\n{J_plus_y}")
print(f"J_minus:\n{J_minus_y}")

# Apply the raising operator to the eigenstates
Jx_eigvec_raised = J_plus_x@eigvec_Jx
Jy_eigvec_raised = J_plus_y@eigvec_Jy
# Apply the lowering operator to the eigenstates
Jx_eigvec_lowered = J_minus_x@eigvec_Jx
Jy_eigvec_lowered = J_minus_y@eigvec_Jy

print(f"Raised Jx eigenstates:\n{Jx_eigvec_raised}")
print(f"Raised Jy eigenstates:\n{Jy_eigvec_raised}")
print(f"Lowered Jx eigenstates:\n{Jx_eigvec_lowered}")
print(f"Lowered Jy eigenstates:\n{Jy_eigvec_lowered}")


