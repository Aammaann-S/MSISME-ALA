
import sys
from typing import Self
import math
import random

"""
A custom vector class implementation for educational purposes.
"""

class Vec:
    def __init__(self, src=None) -> Self: # src = data used to create a vector
        if src is None:
            self.elements = [] # elements is where the data is stored
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec): # if t(another variable object that we are adding to) is not of the type Vec (self defined vector class)
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)]) # zip pairs the values as need (1st of both obj's elments and further)


    def __rmul__(self, scalar: int | float) -> Self: # rmul? r = right i.e when the object is at right side of multiplication (number*vec)
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        #
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5) # means the seprate values of vector under going v*=5
        return self

    # rmul creates a new vectore (5*v)
    # imul modifies existing vector (v*5)

    def __repr__(self) -> str: # default print shows data list (otherwise would have shown an encoded output)
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        # raise RuntimeError("vec subtraction unimplemented")
        if not isinstance(t, Vec): # if t(another variableobject that we are adding to) is not of the type Vec (self defined vector class)
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError(f"Type error - vectors must be of same dimensions")

        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)]) # zip pairs the values as need (1st of both obj's elments and further)
        

    def __neg__(self) -> Self:
        # raise RuntimeError("vec negation unimplemented")
        return Vec([round(x * -1, 5) for x in self.elements])

    def __radd__(self, other): # add with vector on right
        # raise RuntimeError("vec _radd_ unimplemented")

        if other == 0:
            return self

        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        return other + self # redirects/uses __add__

    def __iadd__(self, other): # add with vector on left # would handle inplace vector addition
        # raise RuntimeError("vec _iadd_ unimplemented")
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        if len(self.elements) != len(other):
            raise TypeError("Type error - vectors must be of same dimensions")

        for i in range(len(self.elements)):
            self.elements[i] = round(
                self.elements[i] + other.elements[i],
                5
            )

        return self

    # return a vector of @n zeroes. precondition: @n > 0
    @staticmethod # no self method needed
    def zeros(n: int) -> Self:
        # raise RuntimeError("zeros unimplemented")
        if (n<=0):
            raise RuntimeError("Wrong Value of n")
        return Vec([0]*n)

    # return a vector of @n ones. precondition: @n > 0
    @staticmethod
    def ones(n: int) -> Self:
        # raise RuntimeError("ones unimplemented")
        if (n<=0):
            raise RuntimeError("Wrong Value of n")
        return Vec([1]*n)

    # return a vector of @n uniformly distributed numbers in [0, 1]. precondition: @n > 0
    @staticmethod
    def uniform(n: int) -> Self:
        # raise RuntimeError("random unimplemented")
        if (n<0):
            raise RuntimeError("Wrong Value of n")
        arr = [];
        for i in range(n):
            arr.append(random.random())
        return Vec(arr)

            

    # Calculates the Euclidean norm (L2 norm) of the vector.
    # sqrt(e[0]^2 + e[1]^2 + e[2]^2 + ... + e[n-1]^2)
    def norm(self) -> float:
        # raise RuntimeError("norm unimplemented")
        total = 0
        for i in self.elements:
            total += i**2 # each element raised to power 2
        return math.sqrt(total)


"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.
(5) Test this implementation by importing the class in a sepatate python script.
(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements? # Use "Timeit"    
(7) Measure the performance on your machine. Check it on colab.
(8) use numpy and compare the performance.
"""


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")

if __name__ == "__main__":
    #z1 = Vec.zeros(10)
    v1 = Vec([0, 1, 1.03])
    print(v1)
    v3 = 2.2 * v1
    v3 *= 5
    # v3 = 1 + v3
    print(v3)
    v2 = v1 + v3
    print(v1 + v3)
    #print(-(v1 + v3))
