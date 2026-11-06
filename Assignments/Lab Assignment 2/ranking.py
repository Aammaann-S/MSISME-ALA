import numpy as np


def rank_tags(tag_names, S, in_vocab_tokens):
    """
    Rank tags using max pooling.

    For every tag:
    1. Find its maximum similarity score.
    2. Find the text word responsible for that score.
    3. Sort tags in descending order of their scores.

    Returns:
        List of (tag, score, best_word) tuples.
    """

    # Validate matrix dimensions
    if S.shape[0] != len(in_vocab_tokens):
        raise ValueError("Number of text words does not match S.")

    if S.shape[1] != len(tag_names):
        raise ValueError("Number of tags does not match S.")

    # Find the maximum score for each tag (column-wise)
    max_scores = np.max(S, axis=0)

    # Find the text-word index responsible for each maximum
    best_word_indices = np.argmax(S, axis=0)

    ranked_tags = []

    # Collect each tag, its score, and its matching text word
    for j in range(len(tag_names)):

        best_word = in_vocab_tokens[best_word_indices[j]]

        ranked_tags.append((
            tag_names[j],
            float(max_scores[j]),
            best_word
        ))

    # Sort tags from highest to lowest score
    ranked_tags.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_tags