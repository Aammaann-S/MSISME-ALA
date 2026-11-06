
import numpy as np

from attention import attention


def test_attention():

    np.random.seed(42)

    # create a NumPy array containing 5 tokens
    # each token has 768 features
    # x =[(768... element token),(768...),(768...),..]
    # sequence_length = no of tokens in current space

    # 5 tokens, 768 features per token
    sequence_length = 5
    X = np.random.rand(768, sequence_length)

    # One attention head in GPT-2 Small
    # Wq, Wk, Wv have shape (64, 768)
    # Bq, Bk, Bv have shape (64,)
    # Wq is a n dimentional array 
    Wq = np.random.rand(64, 768)
    Wk = np.random.rand(64, 768)
    Wv = np.random.rand(64, 768)

    Bq = np.random.rand(64)
    Bk = np.random.rand(64)
    Bv = np.random.rand(64)
    
    # Combine all parameters into one tuple
    W = (Wq, Wk, Wv, Bq, Bk, Bv)

    # Call attention
    # Test 1: Unmasked self-attention
    result = attention(X, W)

    assert result.shape == (64, 5)
    assert np.all(np.isfinite(result))
    print("Input shape:", X.shape)

    print("Unmasked attention passed!")
    print("Output shape:", result.shape)

    # Test 2: Causal self-attention
    mask = np.tril(np.ones((5, 5), dtype=bool))

    result_masked = attention(X, W, mask)

    assert result_masked.shape == (64, 5)
    assert np.all(np.isfinite(result_masked))

    print("Causal attention passed!")
    print("Masked output shape:", result_masked.shape)


if __name__ == "__main__":
    test_attention()
