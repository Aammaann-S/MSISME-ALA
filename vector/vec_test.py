from vec import Vec

# Vector creation
v = Vec([1, 2, 3])
assert v.elements == [1, 2, 3]

v = Vec()
assert v.elements == []

try:
    Vec([1, "hello", 3])
    assert False, "Expected TypeError"
except TypeError:
    pass


# Vector addition
v1 = Vec([1, 2, 3])
v2 = Vec([4, 5, 6])

result = v1 + v2
assert result.elements == [5, 7, 9]


# Vector subtraction
result = v2 - v1
assert result.elements == [3, 3, 3]

# Scalar multiplication
result = 2 * v1
assert result.elements == [2, 4, 6]

# In-place multiplication
v3 = Vec([1, 2, 3])
v3 *= 3
assert v3.elements == [3, 6, 9]

# Test negation
result = -v1
assert result.elements == [-1, -2, -3]

# Test in-place addition
v4 = Vec([1, 2, 3])
v4 += Vec([4, 5, 6])
assert v4.elements == [5, 7, 9]

# Test sum() / __radd__()
result = sum([v1, v2])
assert result.elements == [5, 7, 9]

# Test zeros()
result = Vec.zeros(5)
assert result.elements == [0, 0, 0, 0, 0]

# Test ones()
result = Vec.ones(4)
assert result.elements == [1, 1, 1, 1]

# Test uniform()
result = Vec.uniform(100)

assert len(result) == 100

for x in result.elements:
    assert 0 <= x <= 1


# Test norm()
v5 = Vec([3, 4])
assert v5.norm() == 5

v5 = Vec([1, 2, 2])
assert v5.norm() == 3


# Test dimension
v6 = Vec([10, 20, 30, 40])
assert len(v6) == 4

# Test equality
v7 = Vec([1, 2, 3])
v8 = Vec([1, 2, 3])
v9 = Vec([1, 2, 4])

assert v7 == v8
assert not (v7 == v9)
assert not (v7 == [1, 2, 3])


# Test dimension errors
v10 = Vec([1, 2, 3])
v11 = Vec([1, 2])

try:
    v10 + v11
    assert False, "Expected ValueError"
except ValueError:
    pass

try:
    v10 - v11
    assert False, "Expected ValueError"
except ValueError:
    pass

try:
    v10 += v11
    assert False, "Expected ValueError"
except ValueError:
    pass


# Test invalid scalar
v12 = Vec([1, 2, 3])

try:
    "hello" * v12
    assert False, "Expected TypeError"
except TypeError:
    pass

try:
    v12 *= "hello"
    assert False, "Expected TypeError"
except TypeError:
    pass


# Test invalid dimensions
try:
    Vec.zeros(0)
    assert False, "Expected RuntimeError"
except RuntimeError:
    pass

try:
    Vec.ones(-5)
    assert False, "Expected RuntimeError"
except RuntimeError:
    pass

try:
    Vec.uniform(0)
    assert False, "Expected RuntimeError"
except RuntimeError:
    pass


print("All tests passed successfully!")