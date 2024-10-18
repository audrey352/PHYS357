import numpy as np
import scipy.constants as const
hbar = const.hbar

np.set_printoptions(precision=4) # settings to make arrays easier to read
np.set_printoptions(suppress=True)

# Functions
def compute_prob(SGn, state, nbra, en):
    """
    Get the probability of each eigenstate of Jn for a given state in the Jz basis
    nbra and state expressed in the Jz basis
    """
    # Change basis of state to Jn (Jx or Jy)
    # equiv.: how much of each eigenstate of Jn is in the initial state?
    state_in_n = nbra@state
    print(f"\nState in J{SGn[-1]} basis: {np.round(state_in_n,2)}")
    # Calculate probabilities for each basis state of Jn
    prob_n = np.abs(state_in_n)**2
    print(f"Probabilities in J{SGn[-1]} basis:")
    for i,e in enumerate(en):
        print(f"P({e:.0f}) = {prob_n[i].real:.2f}")

def R_matrix_z(theta):
    """
    Rotation matrix around z-axis for an angle theta
    """
    r1 = [np.exp(1j*theta), 0, 0]
    r2 = [0, 1, 0]
    r3 = [0,0,np.exp(-1j*theta)]
    return np.array([r1,r2,r3])

def R_from_J(J,th):
    """
    rotation about an axis is exp(-i th J/hbar)
    """
    e,v=np.linalg.eigh(J)
    e_new=-1j*th*e
    return v@np.diag(np.exp(e_new))@v.conj().T


# Operators for spin-1
Jz = np.diag([1,0,-1])
Jx = np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
Jy = np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])/np.sqrt(2)

# Jx eigenstates
ex,vx = np.linalg.eigh(Jx)
ex,vx = np.flipud(ex), np.fliplr(vx)
xbra=np.conj(vx).T  # z to x

# Jy eigenstates
ey,vy = np.linalg.eigh(Jy)
ey,vy = np.flipud(ey), np.fliplr(vy)
ybra=np.conj(vy).T  # z to y


# Part A -------------------------------------
print("Part A")
# State (kets) in Jz basis
state_in_z = np.array([1,0,1])/np.sqrt(2)
print(f'State after SGz: {np.round(state_in_z,2)}')

# Go through SGx
compute_prob('SGx', state_in_z, xbra, ex)
# Go through SGy
compute_prob('SGy', state_in_z, ybra, ey)


# Part B -------------------------------------
print("\nPart B")
# Rotation matrix (90 deg around z-axis)
theta = np.pi/2
Rz = R_matrix_z(theta)
# Rz = R_from_J(Jz,theta)  # gives same thing

# Rotate the eigenstates
xbra_rotated = Rz@xbra
ybra_rotated = Rz@ybra

# Compute the probabilities
# Rotating should not change the probabilities
compute_prob('SGx', state_in_z, nbra=xbra_rotated, en=ex)
compute_prob('SGy', state_in_z, nbra=ybra_rotated, en=ey)
print('The probabilities are NOT rotation invariant so the state must be wrong')


# Part C -------------------------------------
print("\nPart C")
# Correct state
correct_state_in_z = np.array([1,0,1j])/np.sqrt(2)
print(f'Correct state after SGz: {np.round(correct_state_in_z,2)}')
# Compute the probabilities
compute_prob('SGx', correct_state_in_z, xbra, ex)
compute_prob('SGy', correct_state_in_z, ybra, ey)

# Rotate the setup --> use the rotated bras
compute_prob('SGx', correct_state_in_z, xbra_rotated, ex)
compute_prob('SGy', correct_state_in_z, ybra_rotated, ey)