from vec import Vec

# Test Mean
v = Vec([1, 5, 6])
assert v.mean() == 4

# Vector with negative values
v = Vec([-2, 4, 6])
assert abs(v.mean() - (8 / 3)) < 1e-10

# Vector with decimal values
v = Vec([1.5, 2.5, 3.5])
assert v.mean() == 2.5

# Mean of constant vector
v = Vec([5, 5, 5, 5])
assert v.mean() == 5

# Mean of single-element vector
v = Vec([10])
assert v.mean() == 10

# Empty vector raising ValueError
v = Vec([])

try:
    v.mean()
    assert False, "Expected ValueError"
except ValueError:
    pass

print("tests for mean(): working")

# ================================================ 

# Testing actual demeaned vector
v = Vec([1, 5, 6])
result = v.demean()

assert result.elements == [-3, 1, 2]

# Mean of a demeaned vector should be 0
v = Vec([1, 5, 6])
result = v.demean()

assert abs(result.mean()) < 1e-10

print("tests for demean(): working")

# ================================================ 

# Test Standard Deviation
v = Vec([1, 5, 6])

expected = (14 / 3) ** 0.5

assert abs(v.std() - expected) < 1e-10

# Standard Deviation of a constant vector should be 0
v = Vec([9, 9, 9, 9])

assert v.std() == 0

# Standard Deviation cannot be negative
v = Vec([-2, 5, 6])

assert v.std() >= 0

print("tests for std(): working")

print("All tests passed successfully!")
