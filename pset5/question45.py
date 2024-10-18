import numpy as np

# Functions -------------------------------------
def commutator(A,B):
    return A@B - B@A

def uncertainty(bra, ket, J):
    """
    Calculate the uncertainty of a measurement
    sigma(A) = sqrt(<A^2> - <A>^2)
    """
    term1 = bra@J@J@ket  # <J^2>
    term2 = (bra@J@ket)**2  # <J>^2
    return np.sqrt(term1 - term2)

def de_angle(mat):
    mm = mat.copy()
    # this will loop over every column
    for i in range(mat.shape[1]):
        th = np.angle(mat[0,i]) # return theta from a=c exp(i theta)
        mm[:,i] = mm[:,i]/np.exp(1J*th)
    return mm


# Setting up the problem -------------------------------------
j = 2
m = np.linspace(j,-j,int(2*j)+1)
Jz = np.diag(m)

# State |2,2>z
state_ket = np.array([1,0,0,0,0])
state_bra = np.conj(state_ket.T)

# Building the raising & lowering operators
# (for Jz, in the Jz basis)
# Use these to get Jx and Jy for spin-2
n = len(m)
Jp_z = np.zeros([n,n])
for i in range(1,n):
    val = np.sqrt(j*(j+1)-m[i]*(m[i]+1))
    Jp_z[i-1,i]=val  # we want the i'th element to end up in the (i-1)'th spot
Jm_z = Jp_z.conj().T

# Momentum operators (in the Jz basis)
Jx = (Jp_z + Jm_z)/2
Jy = (Jp_z - Jm_z)/(2j)


# Question 4 -------------------------------------
print('Question 4')

# Part A
print("Part A")
J_sum = Jx@Jx + Jy@Jy
print(f'[Jx, Jx^2+Jy^2]=\n{commutator(Jx, J_sum)}')

# Part B
print("\nPart B")
# Measurement of Jx^2 + Jy^2
# <state|Jx^2+Jy^2|state>
measure_J_sum = state_bra@J_sum@state_ket
print(f"Measurement for Jx^2 + Jy^2: {measure_J_sum.real} hbar^2")

# Uncertainty
# sigma(A) = sqrt(<A^2> - <A>^2)
uncert = uncertainty(state_bra, state_ket, J_sum)
print(f"Uncertainty: {uncert.real}")

# Part C
print("\nPart C")
# <state|Jx^2|state>
measure_Jx2 = state_bra@Jx@Jx@state_ket
print(f"Measurement for Jx^2: {measure_Jx2.real} hbar^2")

# Part D
print("\nPart D")
# Jx eigenstates
ex,vx = np.linalg.eigh(Jx)
ex,vx = np.flipud(ex), np.fliplr(vx)
xbra = np.conj(vx).T  # expressed in Jz basis, use to rotate to Jx basis

# Compute probability for Jx eigenstates
# (everything is expressed in the Jz basis)
prob = np.abs(xbra@state_ket)**2
print(f"P(Jx={ex[0]:.0f}hbar) = {prob[0]:.4f}")


# Question 5 -------------------------------------
print('\nQuestion 5')

# Part A
print("Part A")
# Jy eigenstates
ey,vy = np.linalg.eigh(Jy)
ey,vy = np.flipud(ey), np.fliplr(vy)
ybra = np.conj(vy).T  # expressed in Jz basis, use to rotate to Jy basis

# |2,2>y in the Jz basis
state_y_in_z = de_angle(vy)[:,0]
print(f'|2,2>y in Jz basis: {np.round(state_y_in_z,2)}')
# other method: transforming from y to z basis
# Jz isnt normalized so need /2 to have the state (1,0,0,0,0) with eigenvalue 2hbar
# instead of (2,0,0,0,0) with eigenvalue hbar
rotated_y_to_z = vy@(Jz[:,0]/2)
phase = np.exp(1j*np.pi)  # phase to make amplitude of Jz=+2hbar real and positive. 
print(f'                    {np.round(rotated_y_to_z*phase,2)}')

# |2,2>y in the Jy basis
# this is the same as the |2,2>z in the Jz basis
# dividing by 2 to get the correct normalization
print(f'|2,2>y in Jy basis: {Jz[:,0]/2}')


# Part B
print("\nPart B")