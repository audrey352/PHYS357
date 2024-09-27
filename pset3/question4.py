import numpy as np

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


def genrot(phi, theta, gamma):
    """
    phi, theta: directions of n
    gamma: angle of rotation
    """
    # pick a rotation axis, n in x,y,z coordinates
    n = np.array([np.sin(theta)*np.cos(phi), 
                  np.sin(theta)*np.sin(phi), 
                  np.cos(theta)])

    # generate a basis with n
    M = genbasis(n)  # go from x,y,z to n basis
    M_inv = np.linalg.inv(M)  # go from n basis to x,y,z

    # generate the rotation matrix
    R = R_matrix_n(gamma)  # rotates around n by gamma, in the n basis

    return M_inv@R@M


theta = np.pi/4
phi = np.pi/6
gamma = 0.01

R_n = genrot(phi, theta, gamma)
print(f'Rotation matrix:\n{R_n}')

