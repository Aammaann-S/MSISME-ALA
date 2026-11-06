import numpy as np


def token_embedding(token_id, W_e):
    """
    Algorithm 1: Token embedding

    Parameters:
        token_id : Token ID
        W_e      : Embedding matrix (d_e, vocab_size)

    Returns:
        e        : Embedding vector (d_e,)
    """

    return W_e[:, token_id]