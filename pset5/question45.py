import numpy as np

# Functions -------------------------------------
def commutator(A,B):
    return A@B - B@A

def uncertainty(bra, ket, J):
    """
    Calculate the uncertainty of a measurement
    sigma(A) = sqrt(<A^2> - <A>^2)
    * all matrices need to be expressed in the same basis
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

def R_from_J(J,th):
    """
    rotation about an axis is exp(-i th J/hbar)
    """
    e,v=np.linalg.eigh(J)
    e_new=-1j*th*e
    return v@np.diag(np.exp(e_new))@v.conj().T  

def R_matrix(theta):
    """
    Rotation matrix for an angle theta
    rotate a state around n in the n basis
    """
    r0 = [np.exp(1j*2*theta),0,0,0,0]
    r1 = [0,np.exp(1j*theta),0,0,0]
    r2 = [0,0,1,0,0]
    r3 = [0,0,0,np.exp(-1j*theta),0]
    r4 = [0,0,0,0,np.exp(-1j*2*theta)]
    return np.array([r0,r1,r2,r3,r4])    


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
xket = vx
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
yket = vy
ybra = np.conj(yket).T  # expressed in Jz basis, use to rotate to Jy basis

# |2,2>y in the Jz basis
state_y_bra = de_angle(yket)[:,0]
print(f'|2,2>y in Jz basis: {np.round(state_y_bra,2)}')
# other method: transforming from y to z basis
# Jz isnt normalized so need /2 to have the state (1,0,0,0,0) with eigenvalue 2hbar
# instead of (2,0,0,0,0) with eigenvalue hbar
rotated_y_to_z = yket@(Jz[:,0]/2)
phase = np.exp(1j*np.pi)  # phase to make amplitude of Jz=+2hbar real and positive. 
print(f'                    {np.round(rotated_y_to_z*phase,2)}')

# |2,2>y in the Jy basis
# this is the same as the |2,2>z in the Jz basis
# dividing by 2 to get the correct normalization
print(f'|2,2>y in Jy basis: {Jz[:,0]/2}')


# Part B
print("\nPart B")
state_y_ket = state_y_bra.conj().T
# Note: uncertainty values are real, using .real to remove the 
# imaginary part (which is 0) when printing 

# Calculate uncertainties for Jx and Jz
uncert_Jx = uncertainty(state_y_ket, state_y_bra, Jx)
print(f"Uncertainty for Jx: {uncert_Jx.real:.2f} hbar")
uncert_Jz = uncertainty(state_y_ket, state_y_bra, Jz)
print(f"Uncertainty for Jz: {uncert_Jz.real:.2f} hbar")

# Checkig uncertainty relation: σ(Jx)σ(Jz) ≥ 1/2|<Jy>|
print(f"σ(Jx)σ(Jz) = {uncert_Jx.real*uncert_Jz.real:.2f} hbar^2")
print(f"1/2|<Jy>| = {np.abs(state_y_ket@Jy@state_y_bra).real/2:.2f} hbar^2")


# Question 6 -------------------------------------
print('\nQuestion 6')

# Part A
print("Part A")
# Rotation matrix (90 deg around y-axis)
theta = np.pi/2
Ry = R_from_J(Jy,theta)
print(f'Rotation matrix:\n{np.round(np.abs(Ry),2)}')
# Other method
R_in_y = R_matrix(theta)  # rotate around y-axis, in Jy basis
R = yket@R_in_y@ybra # rotate the state expressed in Jz basis
# print(np.round(np.abs(R),2))  # gives same thing as above

# Part B
print("\nPart B")
# Initial state, |2,2>z
print(f'|2,2>z: {np.round(state_ket,2)}')
# Rotate |2,2>z --> |2,2>x
state_rotated = Ry@state_ket
print(f"Rotated |2,2>z: {np.round(state_rotated.real,2)}")  # values are real, using .real just for printing clarity
state_x_in_z = vx[:,0]
print(f"Expected |2,2>x: {np.round(state_x_in_z,2)}")

# Rotate |2,2>x --> |2,-2>z
state_rotated2 = Ry@state_rotated
print(f'Rotated |2,2>x: {np.round(state_rotated2,2).real}')  # values are real, using .real just for printing clarity
# print(np.round(R@state_x_in_z,2).real)  # gives the same thing! (just another method)
state_z2 = Jz[:,4]/(-2)  # dividing by -2 to get the correct normalization for the eigenvalue -2hbar
print(f'Expected |2,-2>z: {np.round(state_z2,2)}')
