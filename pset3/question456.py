import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

hbar = sp.constants.hbar

# Coordinates
greenwich_lat = 51.476852
greenwich_long = 0.000
mtl_lat = 45.50884
mtl_long = -73.58781

# Functions
def R_matrix_n(angle):
    """
    When defining n, the n vector gets mapped to (1,0,0), so the equiv. to the x axis
    So we want to rotate around 'x'
    """
    r1 = [1, 0, 0]
    r2 = [0,np.cos(angle), -np.sin(angle)]
    r3 = [0,np.sin(angle), np.cos(angle)]
    return np.array([r1,r2,r3])

def genbasis(vector):
    """
    Create new basis with the input vector as the first basis vector
    Returns the change of basis matrix from xyz to the new basis
    """
    # Normalize the input vector
    vector = vector/np.linalg.norm(vector)

    # Arbitrary vector
    v = np.array([0,1,0]) if np.all(vector[1:]==[0,0]) else np.array([1,0,0])

    # Get the other basis vectors (normalized)
    r2 = np.cross(vector,v)
    r2 = r2/np.linalg.norm(r2)
    r3 = np.cross(vector,r2)

    # Change of basis matrix
    M = np.array([vector,r2,r3])

    return M

def genrot(n, gamma):
    """
    xyz: axis we rotate around
    gamma: angle of rotation
    """
    # generate a basis with n
    M = genbasis(n)  # go from x,y,z to n basis
    M_inv = np.linalg.inv(M)  # go from n basis to x,y,z

    # generate the rotation matrix
    R = R_matrix_n(gamma)  # rotates around n by gamma, in the n basis

    return M_inv@R@M

def cartesian_to_spherical(x,y,z):
    # Radial distance
    r = np.sqrt(x**2 + y**2 + z**2)
    # Polar angle theta
    theta = np.arccos(z/r)
    # Azimuthal angle phi
    phi = np.arctan2(y,x)
    
    return theta, phi

def earth_to_cartesian(lat, lon):
    theta = np.pi/2-np.radians(lat)
    phi = np.radians(lon)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def angle_between_vectors(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos_theta = dot_product / (norm1 * norm2)
    return np.degrees(np.arccos(cos_theta))



# Question 4
print('Question 4')
theta = np.pi/4
phi = np.pi/6
gamma = 0.01

# make the rotation axis, n in x,y,z coordinates
n = np.array([np.sin(theta)*np.cos(phi), 
                np.sin(theta)*np.sin(phi), 
                np.cos(theta)])

R_n = genrot(n, gamma)
print(f'Rotation matrix:\n{R_n}')


# Question 5
print('\nQuestion 5')
# Convert to cartesian
g_cart = earth_to_cartesian(greenwich_lat, greenwich_long)  # Greenwich
m_cart = earth_to_cartesian(mtl_lat, mtl_long)  # Montreal

# Angle we want to rotate by (to get Greenwich at north pole)
g_theta = cartesian_to_spherical(*g_cart)[0]

# Get the axis we want to rotate around
n = np.cross(g_cart, [0,0,1])  # g x NP

# Rotation matrix in xyz coordinates
G = genrot(n,g_theta)  # to rotate around n by g_theta
print(f'Rotation matrix for greenwich at NP:\n{G}')

# New Montreal coordinates, in cartesian
m_cart_rotated = G@m_cart
# Convert back to spherical coordinates
m_theta_rotated, m_phi_rotated = cartesian_to_spherical(*m_cart_rotated)

# Get new lat & long of Montreal
mtl_lat_new = np.degrees(np.pi/2 - m_theta_rotated)
mtl_long_new = np.degrees(m_phi_rotated)
print(f'New Montreal lat: {mtl_lat_new}, long: {mtl_long_new}')

# Compare angles between Montreal and Greenwich
initial_angle = angle_between_vectors(m_cart, g_cart)
current_angle = angle_between_vectors(m_cart_rotated, G@g_cart)
print(f'Initial angle between Montreal and Greenwich: {initial_angle}')
print(f'Angle between Montreal and Greenwich: {current_angle}')


# Question 6
print('\nQuestion 6')

gamma = np.linspace(0.01, 0.001, 100)

# Get orthogonal vectors
n1 = np.array([1,0,0])
n2 = np.array([0,1,0])
n3 = np.array([0,0,1])

# Get rotation matrices, shape (N_gamma,3,3)
R1 = np.array([genrot(n1, g) for g in gamma])
R2 = np.array([genrot(n2, g) for g in gamma])
R3 = np.array([genrot(n3, g) for g in gamma])

# Commutators
R1R2 = np.array([R1[i]@R2[i] - R2[i]@R1[i] for i in range(len(gamma))])
pred = np.array([np.eye(3)-R3[i] for i in range(len(gamma))])

# Errors
errors = np.array([np.linalg.norm(R1R2[i]) for i in range(len(gamma))])
R1R2_max = np.array([np.max(np.abs(R1R2[i])) for i in range(len(gamma))])
R1R2_max_err = np.array([np.max(np.abs(R1R2[i]-pred[i])) for i in range(len(gamma))])

# Plot
plt.plot(gamma, R1R2_max, label='Max in [R1,R2]')
plt.plot(gamma, errors, label='Error')
# plt.plot(gamma, R1R2_max_err, label='Max error')
plt.xlabel(r'$\gamma$')
plt.legend()
plt.savefig('/Users/audrey/Documents/PHYS357/pset3/question6.png')
plt.show()


