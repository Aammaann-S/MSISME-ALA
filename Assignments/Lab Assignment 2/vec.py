# vector > vec.py
import sys
from typing import Self
import math
import random
import numpy as np

class Vec:

    def __init__(self, src=None) -> Self: # src = data used to create a vector
        """Create a vector from a numeric values."""
        if src is None:
            self.elements = [] # elements is where the data is stored
        else:
            elements = list(src)
            for x in elements:
                if not isinstance(x, (int, float, np.number)):
                    raise TypeError(f"Scalar must be a number: {type(x)}")
            self.elements = elements

    def __add__(self, t: Self) -> Self:
        """Return a new vector containing the sum of two vectors."""
        if not isinstance(t, Vec): # if (another variable object that we are adding to) is not of the type Vec (self defined vector class)
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise ValueError(f"Error - vectors must be of same dimensions")

        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)]) # zip pairs the values as need (1st of both obj's elments and further)

    def __rmul__(self, scalar: int | float) -> Self: # rmul? r = right i.e when the object is at right side of multiplication (number*vec)
        """Return a new vector multiplied with a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        """Modifies current vector multiplied with a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")

        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5) # means the seprate values of vector under going v*=5
        return self

    # rmul creates a new vectore (5*v)
    # imul modifies existing vector (v*5)

    def __repr__(self) -> str: # default print shows data list (otherwise would have shown an encoded output)
        """Returns a readable string representation of the vector."""
        return repr(self.elements)

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return len(self.elements)

    def __sub__(self, t: Self) -> Self:
        """Return a new vector containing the difference of 2 vectors. """
        # raise RuntimeError("vec subtraction unimplemented")
        if not isinstance(t, Vec): # if (another variable object that we are subtracting) is not of the type Vec (self defined vector class)
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise ValueError(f"Error - vectors must be of same dimensions")
        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)]) # zip pairs the values as need (1st of both obj's elments and further)
        
    def __neg__(self) -> Self:
        """Return a new vector with every element with sign inverted."""
        # raise RuntimeError("vec negation unimplemented")
        return Vec([round(x * -1, 5) for x in self.elements])

    def __radd__(self, other): # add with vector on right
        """Return a new vector containing the sum of two vectors."""

        if other == 0:
            return self

        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        return other + self 

    def __iadd__(self, other): # add with vector on left # inplace vector addition
        """Add another vector to this vector in place."""
        # raise RuntimeError("vec _iadd_ unimplemented")
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        if len(self.elements) != len(other):
            raise ValueError(f"Error - vectors must be of same dimensions")

        for i in range(len(self.elements)):
            self.elements[i] = round(
                self.elements[i] + other.elements[i],
                5
            )

        return self

    # return a vector of @n zeroes. precondition: @n > 0
    @staticmethod # no self method needed
    def zeros(n: int) -> Self:
        """Returns a vector with n zeros."""
        # raise RuntimeError("zeros unimplemented")
        if (n<=0):
            raise RuntimeError("Wrong Value of n")
        return Vec([0]*n)

    # return a vector of @n ones. precondition: @n > 0
    @staticmethod
    def ones(n: int) -> Self:
        """Returns a vector with n ones."""
        # raise RuntimeError("ones unimplemented")
        if (n<=0):
            raise RuntimeError("Wrong Value of n")
        return Vec([1]*n)

    # return a vector of @n uniformly distributed numbers in [0, 1]. precondition: @n > 0
    @staticmethod
    def uniform(n: int) -> Self:
        """Returns a vector with n random values in the range [0, 1]."""
        # raise RuntimeError("random unimplemented")
        if (n<=0):
            raise RuntimeError("Wrong Value of n")
        arr = []
        for i in range(n):
            arr.append(random.random())
        return Vec(arr)

    # Calculates the Euclidean norm (L2 norm) of the vector.
    # sqrt(e[0]^2 + e[1]^2 + e[2]^2 + ... + e[n-1]^2)
    def norm(self) -> float:
        """Returns the Euclidiean norm of the vector."""
        # raise RuntimeError("norm unimplemented")
        total = 0
        for i in self.elements:
            total += i**2 # each element raised to power 2
        return math.sqrt(total)
    
    # Implement the "==" operator to compare 2 Vec instances. (normally calling "==" will give "false")
    def __eq__(self, other) -> bool:
        """Returns boolean value based on comparison of 2 vectors(element comparison)."""
        if not isinstance(other, Vec):
            return False
        return self.elements == other.elements

    # Assignment 1 
    def mean(self) -> float:
        """Return the mean of the vector values."""
        if len(self.elements) == 0:
            # raise RuntimeError("Cannot compute mean of empty vector.")
            raise ValueError("Cannot compute mean of empty vector.")
        return sum(self.elements) / len(self.elements)

    def demean(self) -> Self:
        """ Return De-mean vector of the given vector."""
        # De-mean = resulting vector of mean subtracted from every entry of the vector.
        mean_value = self.mean()
        return Vec([x - mean_value for x in self.elements])

    def std(self) -> float:
        """Return the Standard Deviation of the vector entries."""
        demeaned = self.demean()
        squared_sum = sum(x ** 2 for x in demeaned.elements)
        return math.sqrt(squared_sum / len(self.elements))

    def dot(self, other: Self) -> float:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        if len(self) != len(other):
            raise ValueError("Vectors must be of same dimensions")

        return sum(
            x * y
            for x, y in zip(self.elements, other.elements)
        )
    
    def cosine_similarity(self, other: Self) -> float:
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")

        self_norm = self.norm()
        other_norm = other.norm()

        if self_norm == 0 or other_norm == 0:
            raise ValueError(
                "Cosine similarity is undefined for a zero vector"
            )

        return self.dot(other) / (self_norm * other_norm)


TAGS = [
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "technology",
    "curriculum",
    "collaboration",
    "publication",
    "laboratory",
    "scholarship",
    "mentorship",
    "internship",
    "entrepreneurship",
    "accreditation",
    "alumni",
]


if sys.version_info < (3, 11):
    sys.exit("Error: This script requires Python 3.11 or higher.")
