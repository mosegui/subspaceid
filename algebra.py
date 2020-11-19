import numpy as np

def matdiv(mat1, mat2):
    """
    Relevant links:
        https://de.mathworks.com/help/fixedpoint/ref/embedded.fi.mrdivide.html
        https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
    """
    A = np.asmatrix(mat1)
    B = np.asmatrix(mat2)

    X = A * B.T * (B * B.T).I

    return np.asarray(X)

def project(mat1, mat2):
    """Orthogonal projection of mat1 on mat2 along a direction
    perpendicular to mat2
    """
    return matdiv(mat1, mat2) @ mat2

def project_on_perpendicular(mat1, mat2):
    """Projects mat1 onto the orthogonal component of the row space
    of mat2
    """
    return mat1 - project(mat1, mat2)
