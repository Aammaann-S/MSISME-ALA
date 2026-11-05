import re

# Small custom stopword set.
# The additional words are chosen for our MAHE LinkedIn text.
STOPWORDS = {
    # Assignment starter words
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",

    # Additional common words
    "it", "this", "that", "our", "we",
    "these", "those", "their", "they", "them",
    "also", "as"
}


def preprocess_text(text, model):
    """
    Preprocess input text, 
    identify in-vocabulary and out-of-vocabulary tokens.

    Returns:
        tokens: tokens remaining after the first five steps
        in_vocab_tokens: tokens present in the model vocabulary
        oov_tokens: tokens absent from the model vocabulary
    """

    # 1. Convert the text to lowercase
    text = text.lower()

    # Count original whitespace-separated tokens
    tokens_before = len(text.split())

    # 2. Remove punctuation
    # Replace punctuation with spaces so words do not get joined.
    text = re.sub(r"[^\w\s]", " ", text)

    # 3. Split the resulting text into tokens
    tokens = text.split()

    # 4. Remove stopwords
    tokens = [
        word for word in tokens
        if word not in STOPWORDS
    ]

    # 5. Remove tokens whose length is <= 2
    tokens = [
        word for word in tokens
        if len(word) > 2
    ]

    # 6. Check which tokens occur in the embedding vocabulary
    in_vocab_tokens = []
    oov_tokens = []

    for word in tokens:
        if word in model.key_to_index:
            in_vocab_tokens.append(word)
        else:
            oov_tokens.append(word)

    # Print the required statistics
    print("\n----- TEXT PREPROCESSING RESULTS -----")

    print("Tokens before preprocessing:", tokens_before)
    print("Tokens after preprocessing:", len(tokens))

    print("In-vocabulary tokens:", len(in_vocab_tokens))
    print("Out-of-vocabulary tokens:", len(oov_tokens))

    print("Distinct OOV tokens:", sorted(set(oov_tokens)))

    print("--------------------------------------")

    return tokens, in_vocab_tokens, oov_tokens
