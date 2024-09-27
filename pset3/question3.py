import numpy as np

def genbasis(xyz): 
    row_1 = np.array([1,0,0]) if xyz=='x' else (np.array([0,1,0]) if xyz=='y' else np.array([0,0,1]))
    row_2 = np.array([0,1,0]) if xyz=='x' else np.array([1,0,0])
    row_3 = np.cross(row_1, row_2)

    A = np.array([row_1, row_2, row_3])
    return A


# Change of basis matrix
axis = 'y'
A = genbasis(axis)
print(f'Input axis is {axis}:\n{A}')

# Get inverse
A_inv = np.linalg.inv(A)

# Checking
x = np.array([1,0,0])
y = np.array([0,1,0])
z = np.array([0,0,1])

print(f'{A@y}')
print(f'{A_inv@A}')


