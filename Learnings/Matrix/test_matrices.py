from matrices import Matrix


A = Matrix([
    [1, 2],
    [3, 4]
])

B = Matrix([
    [5, 6],
    [7, 8]
])

print("Matrix A:", A)
print("Matrix B:", B)

print("\nShape of A:", A.shape())

print("\nAddition:")
print(A + B)

print("\nSubtraction:")
print(A - B)

print("\nScalar multiplication:")
print(3 * A)

print("\nMatrix multiplication:")
print(A @ B)

print("\nTranspose:")
print(A.transpose())

print("\nZero matrix:")
print(Matrix.zeros(2, 3))

print("\nOnes matrix:")
print(Matrix.ones(2, 3))

print("\nIdentity matrix:")
print(Matrix.identity(3))

print("\nEquality:")
print(A == B)
