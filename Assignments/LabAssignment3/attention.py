
import numpy as np


def attention(X, W, mask=None):
    """
    Algorithm 4: Attention (self-attention implementation)

    Parameters:
        X    : Input matrix (d_in, sequence_length)
        W    : Tuple (Wq, Wk, Wv, Bq, Bk, Bv)
        mask : Optional attention mask (sequence_length, sequence_length)

    Returns:
        output : Attention output (d_out, sequence_length)
    """

    # Unpack parameters
    Wq, Wk, Wv, Bq, Bk, Bv = W

    # Step 1: Calculate Query, Key and Value matrices
    Q = Wq @ X + Bq[:, None]
    K = Wk @ X + Bk[:, None]
    V = Wv @ X + Bv[:, None]

    # Step 2: Calculate attention scores
    d_attn = Q.shape[0]
    S = (K.T @ Q) / np.sqrt(d_attn)

    # Step 3: Apply attention mask, if provided
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != S.shape:
            raise ValueError("Mask must have the same shape as the score matrix")

        S = np.where(mask, S, -np.inf)

    # Step 4: Apply softmax along context-token dimension
    # to convert scores into attention weights
    # Tokens are columns, so normalize along axis=0
    S = S - np.max(S, axis=0, keepdims=True)

    A = np.exp(S)
    A = A / np.sum(A, axis=0, keepdims=True)

    # Step 5: Calculate weighted sum of Value vectors
    # Combine the value vectors using attention weights
    output = V @ A

    return output
