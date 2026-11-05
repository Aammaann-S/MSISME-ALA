import unittest
import numpy as np

from similarity_vector import similarity_matrix_vector_class
from similarity_numpy import similarity_matrix_numpy
from ranking import rank_tags


class TestSimilarityMatrix(unittest.TestCase):

    def setUp(self):
        """
        Create small matrices with predictable results.
        """

        self.W = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [3.0, 4.0]
        ])

        self.T = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])

    def test_numpy_similarity_shape(self):
        """
        Similarity matrix should have shape
        (number of words, number of tags).
        """

        S = similarity_matrix_numpy(self.W, self.T)

        self.assertEqual(S.shape, (3, 2))

    def test_numpy_similarity_values(self):
        """
        Check known cosine similarity values.
        """

        S = similarity_matrix_numpy(self.W, self.T)

        expected = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8]
        ])

        np.testing.assert_allclose(S, expected)

    def test_vector_similarity_matches_numpy(self):
        """
        Both implementations should produce
        numerically equivalent results.
        """

        S_vector = similarity_matrix_vector_class(
            self.W, self.T
        )

        S_vector_array = np.array(S_vector.elements)

        S_numpy = similarity_matrix_numpy(
            self.W, self.T
        )

        np.testing.assert_allclose(
            S_vector_array,
            S_numpy,
            rtol=1e-5,
            atol=1e-8
        )

    def test_zero_vector_rejected_by_numpy(self):
        """
        NumPy implementation should reject
        zero vectors before normalization.
        """

        W_zero = np.array([
            [0.0, 0.0]
        ])

        with self.assertRaises(ValueError):
            similarity_matrix_numpy(W_zero, self.T)


class TestTagRanking(unittest.TestCase):

    def test_rank_tags(self):
        """
        Tags should be sorted by descending
        max-pool score, with matching words.
        """

        tag_names = ["research", "technology"]

        S = np.array([
            [0.1, 0.8],
            [0.9, 0.2],
            [0.4, 0.7]
        ])

        tokens = ["word1", "word2", "word3"]

        result = rank_tags(
            tag_names,
            S,
            tokens
        )

        expected = [
            ("research", 0.9, "word2"),
            ("technology", 0.8, "word1")
        ]

        self.assertEqual(result, expected)

    def test_ranking_dimension_mismatch(self):
        """
        Ranking should reject inconsistent
        matrix and token dimensions.
        """

        tag_names = ["research", "technology"]

        S = np.array([
            [0.5, 0.7]
        ])

        tokens = ["word1", "word2"]

        with self.assertRaises(ValueError):
            rank_tags(tag_names, S, tokens)


if __name__ == "__main__":
    unittest.main()


'''
Output:

(.virtualenv) > python -m unittest test_assignment2.py -v
test_numpy_similarity_shape (test_assignment2.TestSimilarityMatrix.test_numpy_similarity_shape)
Similarity matrix should have shape ... ok
test_numpy_similarity_values (test_assignment2.TestSimilarityMatrix.test_numpy_similarity_values)
Check known cosine similarity values. ... ok
test_vector_similarity_matches_numpy (test_assignment2.TestSimilarityMatrix.test_vector_similarity_matches_numpy)
Both implementations should produce ... ok
test_zero_vector_rejected_by_numpy (test_assignment2.TestSimilarityMatrix.test_zero_vector_rejected_by_numpy)
NumPy implementation should reject ... ok
test_rank_tags (test_assignment2.TestTagRanking.test_rank_tags)
Tags should be sorted by descending ... ok
test_ranking_dimension_mismatch (test_assignment2.TestTagRanking.test_ranking_dimension_mismatch)
Ranking should reject inconsistent ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.245s

OK

'''