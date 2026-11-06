
import numpy as np

from attention import attention


def mhattention(X, head_weights, Wo, bo, mask=None):
    """
    Algorithm 5: Multi-Head Attention

    Parameters:
        X           : Input matrix (d_in, sequence_length)
        head_weights: List of parameter tuples, one per attention head
        Wo          : Output projection matrix (d_out, H * head_output_dim)
        bo          : Output projection bias (d_out,)
        mask        : Optional attention mask (sequence_length, sequence_length)

    Returns:
        output      : Final multi-head attention output
    """

    # Step 1: Apply attention independently to every head
    head_outputs = []

    for W in head_weights:
        head_output = attention(X, W, mask)
        head_outputs.append(head_output)

    # Step 2: Concatenate all head outputs vertically
    Y = np.concatenate(head_outputs, axis=0)

    # Step 3: Apply final output projection
    output = Wo @ Y + bo[:, None]

    return output
