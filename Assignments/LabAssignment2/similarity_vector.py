from vec import Vec
from matrices import Matrix


def similarity_matrix_vector_class(W, T):
    """
    Compute the complete cosine-similarity matrix
    using the custom Vec and Matrix classes.

    Parameters:
        W: Text matrix, shape (n_words, 50).
        T: Tag matrix, shape (n_tags, 50).

    Returns:
        S: Custom Matrix object, shape (n_words, n_tags).
    """

    # Check that both matrices have the same vector dimensions
    if len(W[0]) != len(T[0]):
        raise ValueError(
            "W and T must have the same vector dimensions."
        )

    n_words = len(W)
    n_tags = len(T)

    # Store similarity scores
    S = []

    # Loop through every text word
    for i in range(n_words):

        row = []

        # Compare this word against every tag
        for j in range(n_tags):

            # Convert rows into custom Vec objects
            word_vector = Vec(W[i])
            tag_vector = Vec(T[j])

            # Calculate cosine similarity
            score = float(word_vector.cosine_similarity(tag_vector))

            # Store the similarity score
            row.append(score)

        # Add completed row
        S.append(row)

    # Convert the result into our custom Matrix object
    return Matrix(S)