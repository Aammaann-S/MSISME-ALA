import math
import random
from typing import Self

"""
A custom Matrix class implementation for educational purposes.
Matrix operations are implemented using Python lists and loops.
No NumPy matrix operations are used.
"""


class Matrix:

    def __init__(self, src=None) -> Self:
        """Create a matrix from a nested list of numeric values."""

        if src is None:
            self.elements = []

        else:
            elements = [list(row) for row in src]

            # Check that all rows have the same number of columns.
            if len(elements) > 0:
                columns = len(elements[0])

                for row in elements:
                    if len(row) != columns:
                        raise ValueError("All rows must have the same number of columns.")

            # Check that every element is numeric.
            for row in elements:
                for x in row:
                    if not isinstance(x, (int, float)):
                        raise TypeError(f"Matrix elements must be numeric: {type(x)}")

            self.elements = elements

    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns."""

        rows = len(self.elements)

        if rows == 0:
            return (0, 0)

        columns = len(self.elements[0])

        return (rows, columns)

    def __repr__(self) -> str:
        """Return a readable representation of the matrix."""

        return repr(self.elements)

    def __len__(self) -> int:
        """Return the number of rows."""

        return len(self.elements)

    def __eq__(self, other) -> bool:
        """Compare two matrices element by element."""

        if not isinstance(other, Matrix):
            return False

        return self.elements == other.elements

    def __add__(self, other: Self) -> Self:
        """Return a new matrix containing the sum of two matrices."""

        if not isinstance(other, Matrix):
            raise TypeError(f"Expected Matrix: {type(other)}")

        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions.")

        rows, columns = self.shape()

        result = []

        for i in range(rows):
            row = []

            for j in range(columns):
                value = self.elements[i][j] + other.elements[i][j]
                row.append(round(value, 5))

            result.append(row)

        return Matrix(result)

    def __sub__(self, other: Self) -> Self:
        """Return a new matrix containing the difference of two matrices."""

        if not isinstance(other, Matrix):
            raise TypeError(f"Expected Matrix: {type(other)}")

        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions.")

        rows, columns = self.shape()

        result = []

        for i in range(rows):
            row = []

            for j in range(columns):
                value = self.elements[i][j] - other.elements[i][j]
                row.append(round(value, 5))

            result.append(row)

        return Matrix(result)

    def __neg__(self) -> Self:
        """Return a new matrix with every element's sign inverted."""

        result = []

        for row in self.elements:
            new_row = []

            for value in row:
                new_row.append(round(-value, 5))

            result.append(new_row)

        return Matrix(result)

    def __mul__(self, scalar: int | float) -> Self:
        """Return a new matrix multiplied by a scalar."""

        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Matrix multiplication requires a scalar: {type(scalar)}")

        result = []

        for row in self.elements:
            new_row = []

            for value in row:
                new_row.append(round(value * scalar, 5))

            result.append(new_row)

        return Matrix(result)

    def __rmul__(self, scalar: int | float) -> Self:
        """Support scalar multiplication when the scalar is on the left."""

        return self.__mul__(scalar)

    def __imul__(self, scalar: int | float) -> Self:
        """Multiply the current matrix by a scalar in place."""

        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Matrix multiplication requires a scalar: {type(scalar)}")

        for i in range(len(self.elements)):
            for j in range(len(self.elements[i])):
                self.elements[i][j] = round(
                    self.elements[i][j] * scalar, 5
                )

        return self

    def __matmul__(self, other: Self) -> Self:
        """Perform matrix multiplication using rows and columns."""

        if not isinstance(other, Matrix):
            raise TypeError(f"Expected Matrix: {type(other)}")

        rows_a, columns_a = self.shape()
        rows_b, columns_b = other.shape()

        # Number of columns in A must equal number of rows in B.
        if columns_a != rows_b:
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication."
            )

        result = []

        for i in range(rows_a):
            row = []

            for j in range(columns_b):

                total = 0

                for k in range(columns_a):
                    total += (
                        self.elements[i][k] * other.elements[k][j]
                    )

                row.append(round(total, 5))

            result.append(row)

        return Matrix(result)

    def transpose(self) -> Self:
        """Return the transpose of the matrix."""

        rows, columns = self.shape()

        result = []

        for j in range(columns):
            new_row = []

            for i in range(rows):
                new_row.append(self.elements[i][j])

            result.append(new_row)

        return Matrix(result)

    @staticmethod
    def zeros(rows: int, columns: int) -> Self:
        """Create a matrix containing only zeros."""

        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and columns must be positive.")

        result = []

        for i in range(rows):
            row = []

            for j in range(columns):
                row.append(0)

            result.append(row)

        return Matrix(result)

    @staticmethod
    def ones(rows: int, columns: int) -> Self:
        """Create a matrix containing only ones."""

        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and columns must be positive.")

        result = []

        for i in range(rows):
            row = []

            for j in range(columns):
                row.append(1)

            result.append(row)

        return Matrix(result)

    @staticmethod
    def identity(n: int) -> Self:
        """Create an n x n identity matrix."""

        if n <= 0:
            raise ValueError("Matrix size must be positive.")

        result = []

        for i in range(n):
            row = []

            for j in range(n):

                if i == j:
                    row.append(1)
                else:
                    row.append(0)

            result.append(row)

        return Matrix(result)

    @staticmethod
    def uniform(rows: int, columns: int) -> Self:
        """Create a matrix with random values between 0 and 1."""

        if rows <= 0 or columns <= 0:
            raise ValueError("Rows and columns must be positive.")

        result = []

        for i in range(rows):
            row = []

            for j in range(columns):
                row.append(random.random())

            result.append(row)

        return Matrix(result)
