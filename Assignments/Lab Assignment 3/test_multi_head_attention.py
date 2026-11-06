
import numpy as np

from multi_head_attention import mhattention


def test_mhattention():

    np.random.seed(42)

    # GPT-2 Small dimensions
    H = 12
    d_in = 768
    d_attn = 64
    sequence_length = 5

    # Input: 5 tokens, 768 features each
    X = np.random.rand(d_in, sequence_length)

    # Generate independent parameters for 12 heads
    head_weights = []

    for h in range(H):

        Wq = np.random.rand(d_attn, d_in)
        Wk = np.random.rand(d_attn, d_in)
        Wv = np.random.rand(d_attn, d_in)

        Bq = np.random.rand(d_attn)
        Bk = np.random.rand(d_attn)
        Bv = np.random.rand(d_attn)

        W = (Wq, Wk, Wv, Bq, Bk, Bv)

        head_weights.append(W)

    # Final output projection
    Wo = np.random.rand(d_in, H * d_attn)
    bo = np.random.rand(d_in)

    # Causal attention mask
    mask = np.tril(
        np.ones((sequence_length, sequence_length), dtype=bool)
    )

    # Run multi-head attention
    result = mhattention(X, head_weights, Wo, bo, mask)

    # Verify output dimensions
    assert result.shape == (768, 5)

    # Verify numerical validity
    assert np.all(np.isfinite(result))

    print("Multi-head attention test passed!")
    print("Input shape:", X.shape)
    print("Number of heads:", H)
    print("Output shape:", result.shape)


if __name__ == "__main__":
    test_mhattention()
