
import numpy as np

def layer_norm(e, gamma, beta):
    """
    Algorithm 6: Layer Normalization

    Parameters:
        e     : Input embedding vector (d_e,)
        gamma : Learnable scale vector (d_e,)
        beta  : Learnable offset vector (d_e,)

    Returns:
        e_hat : Normalized embedding vector (d_e,)
    """

    # Step 1: Calculate mean
    m = np.mean(e)

    # Step 2: Calculate variance
    v = np.mean((e - m) ** 2)

    # Step 3: Normalize the embedding
    e_normalized = (e - m) / np.sqrt(v)

    # Step 4: Apply scale and offset
    e_hat = e_normalized * gamma + beta

    return e_hat
