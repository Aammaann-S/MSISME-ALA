import numpy as np
from token_embedding import token_embedding


def test_token_embedding():

    W_e = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [1.1, 1.2, 1.3, 1.4, 1.5],
        [2.1, 2.2, 2.3, 2.4, 2.5],
        [3.1, 3.2, 3.3, 3.4, 3.5]
    ])

    token_id = 2

    result = token_embedding(token_id, W_e)

    expected = np.array([0.3, 1.3, 2.3, 3.3])

    assert np.allclose(result, expected)
    assert result.shape == (4,)

    print("Token embedding test passed!")


if __name__ == "__main__":
    test_token_embedding()