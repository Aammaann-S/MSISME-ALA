import numpy as np

from layer_norm import layer_norm


def test_layer_norm():

    # Input embedding
    e = np.array([1.0, 2.0, 3.0, 4.0])

    # Scale and offset
    gamma = np.ones(4)
    beta = np.zeros(4)

    # Apply layer normalization
    result = layer_norm(e, gamma, beta)

    # Expected output
    expected = np.array([
        -1.3416407865,
        -0.4472135955,
         0.4472135955,
         1.3416407865
    ])

    # Verify output
    np.testing.assert_allclose(result, expected, atol=1e-8)

    # Verify shape
    assert result.shape == (4,)

    print("Layer normalization test passed!")
    print("Output:", result)


if __name__ == "__main__":
    test_layer_norm()
