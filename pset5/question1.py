import numpy as np
import scipy.constants as const

# constants
# hbar = const.hbar
hbar = 1

# state
N = 1/np.sqrt(30)
ket = N*np.array([[1j],[2],[3], [4j]])
bra = np.conj(ket.T)

# operators
Sx = np.array([[0,np.sqrt(3),0,0],[np.sqrt(3),0,2,0],[0,2,0,np.sqrt(3)],[0,0,np.sqrt(3),0]])

Sx_exp = bra@Sx@ket
print(f"<Sx> = {Sx_exp}")


# Eigenstates
eigvals_Sx, eigvec_Sx = np.linalg.eigh(Sx)
eigvals_Sx, eigvec_Sx = np.flipud(eigvals_Sx), np.fliplr(eigvec_Sx)
print(f"Eigenvalues of Sx: {eigvals_Sx}")
print(f"Eigenstates of Sx:\n{eigvec_Sx}")

# Probability
ket_Sx = eigvec_Sx[:,1]
bra_Sx = np.conj(ket_Sx.T)
prob = np.abs(bra_Sx@ket)**2
print(f"Probability of measuring Sx = hbar/2: {prob}")