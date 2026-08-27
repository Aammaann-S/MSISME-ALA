import timeit
from vec import Vec

sizes = [2000, 4000, 8000, 16000, 32000, 64000]

for n in sizes:

    print(f"Vector dimension: {n}")
    v1 = Vec.uniform(n)
    v2 = Vec.uniform(n)

    addition_time = timeit.timeit(
        lambda: v1 + v2,
        number=10
    )

    subtraction_time = timeit.timeit(
        lambda: v1 - v2,
        number=10
    )

    multiplication_time = timeit.timeit(
        lambda: 2 * v1,
        number=10
    )

    negation_time = timeit.timeit(
        lambda: -v1,
        number=10
    )

    norm_time = timeit.timeit(
        lambda: v1.norm(),
        number=10
    )

    print(f"Addition:       {addition_time:.6f} seconds")
    print(f"Subtraction:    {subtraction_time:.6f} seconds")
    print(f"Multiplication: {multiplication_time:.6f} seconds")
    print(f"Negation:       {negation_time:.6f} seconds")
    print(f"Norm:           {norm_time:.6f} seconds")