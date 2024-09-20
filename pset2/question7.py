import numpy as np

# state in its own basis
np_in_n = np.array([1,0])
nm_in_n = np.array([0,1])

# x kets
xp_in_z = np.array([1,1])/np.sqrt(2)
xm_in_z = np.array([1,-1])/np.sqrt(2)
# y kets
yp_in_z = np.array([1,1j])/np.sqrt(2)
ym_in_z = np.array([1,-1j])/np.sqrt(2)


# First way
print("First Way:")
left_matrix = np.zeros([2,2])
left_matrix[:,0]=nm_in_n
left_matrix[:,1]=xp_in_z
right_matrix = np.zeros([2,2])
right_matrix[:,0]=xp_in_z
right_matrix[:,1]=np_in_n

R_first_way = left_matrix @ np.linalg.inv(right_matrix)
print('R_first_way:\n', R_first_way)


# Second way
print("\nSecond Way:")
# Rotation matrices
R_ztoy = np.outer(np_in_n,np.conj(yp_in_z.T)) + np.outer(nm_in_n,np.conj(ym_in_z.T))  # z to y basis
R_ytoz = np.conj(R_ztoy.T)  # y to z basis  (inverse of R_ztoy)
def rotation_theta(theta):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]])  # rotation matrix for theta
R_theta = rotation_theta(np.pi/2)
# print(R_ztoy, '\n', R_ytoz)

R_second_way = R_ytoz @ R_theta @ R_ztoy
print('R_second_way:\n', R_second_way)

