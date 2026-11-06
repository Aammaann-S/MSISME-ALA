import numpy as np


def similarity_matrix_numpy(W, T):
    """
    Compute all text-word/tag cosine similarities using
    NumPy row normalization and matrix multiplication.

    Parameters:
        W: Text matrix, shape (n_words, 50).
        T: Tag matrix, shape (n_tags, 50).

    Returns:
        S: NumPy array, shape (n_words, n_tags).
    """

    # Calculate the Euclidean norm of every row
    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)

    # Check for zero vectors before normalization
    if np.any(W_norm == 0) or np.any(T_norm == 0):
        raise ValueError("Cannot normalize a zero vector.")

    # Normalize every row
    W_hat = W / W_norm
    T_hat = T / T_norm

    # Calculate all cosine similarities
    S = W_hat @ T_hat.T

    return S