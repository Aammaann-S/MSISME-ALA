from pathlib import Path
import numpy as np

from word2vec.src.w2v import load_model
from preprocessing import preprocess_text
from matrices import build_tag_matrix, build_text_matrix
from similarity_vector import similarity_matrix_vector_class
from similarity_numpy import similarity_matrix_numpy
from ranking import rank_tags
from vec import TAGS

BASE_DIR = Path(__file__).resolve().parent

# Load input text
with open(BASE_DIR / "input.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Load GloVe model
model_path = (
    BASE_DIR / "word2vec" / "glove50"
    / "glove_50_fast.wordvectors"
)

model = load_model(str(model_path))

if model is None:
    raise RuntimeError("Model loading failed.")

# Part C: Preprocess
tokens, _, _ = preprocess_text(text, model)

# Part D: Build matrices
tag_names, T = build_tag_matrix(model, TAGS)

in_vocab_tokens, oov_tokens, W = build_text_matrix(
    model, tokens
)

# Part E: Calculate similarity using custom Vec and Matrix classes
S_vector = similarity_matrix_vector_class(W, T)

# Display results
print("\n========== PART E RESULTS ==========")

print("W shape:", W.shape)
print("T shape:", T.shape)
print("Similarity matrix shape:", S_vector.shape())

print("\nFirst text word:", in_vocab_tokens[0])
print("First tag:", TAGS[0])

print("Similarity score:", S_vector[0, 0])

print("\nFirst row (similarity with all 20 tags):")
print(S_vector[0])

# Validate similarity matrix
all_scores = [
    value
    for row in S_vector.elements
    for value in row
]

print("\n========== SIMILARITY VALIDATION ==========")

print("Total similarity scores:", len(all_scores))
print("Minimum similarity:", min(all_scores))
print("Maximum similarity:", max(all_scores))

print(
    "All scores within [-1, 1]:",
    all(-1 <= score <= 1 for score in all_scores)
)

print("===========================================")

# Part F: Calculate similarity using NumPy
S_numpy = similarity_matrix_numpy(W, T)

print("\n========== PART F RESULTS ==========")
print("NumPy similarity matrix shape:", S_numpy.shape)
print("First similarity score:", S_numpy[0, 0])
print("====================================")

# Part G: Verify both implementations

# Convert custom Matrix into a NumPy array
S_vector_array = np.array(S_vector.elements)

# Compare both similarity matrices
are_equal = np.allclose(
    S_vector_array,
    S_numpy,
    rtol=1e-5,
    atol=1e-8
)

# Calculate maximum absolute difference
max_difference = np.max(
    np.abs(S_vector_array - S_numpy)
)

print("\n========== PART G VERIFICATION ==========")
print("Vector similarity shape:", S_vector_array.shape)
print("NumPy similarity shape:", S_numpy.shape)
print("Both implementations match:", are_equal)
print("Maximum absolute difference:", max_difference)
print("=========================================")

# Part H: Rank tags using max pooling

ranked_tags = rank_tags(
    tag_names,
    S_numpy,
    in_vocab_tokens
)

print("\n========== PART H: TOP 8 TAGS ==========")
print(f"{'Rank':<8}{'Tag':<22}{'Score':<15}{'Best Matching Word'}")
print("-" * 65)

for rank, (tag, score, best_word) in enumerate(
    ranked_tags[:8],
    start=1
):
    print(
        f"{rank:<8}{tag:<22}{score:<15.6f}{best_word}"
    )

print("=" * 65)

'''
Output:

----- TEXT PREPROCESSING RESULTS -----
Tokens before preprocessing: 400
Tokens after preprocessing: 282
In-vocabulary tokens: 263
Out-of-vocabulary tokens: 19
Distinct OOV tokens: ['academicexcellence', 'globalhealth', 'globalrankings', 'highereducation', 'infectiousdiseases', 'institutionofeminence','mahemanipal', 'mahenatureconference', 'manipalacademyofhighereducation', 'onehealth', 'proudmahe', 'qsasia2027', 'qsasiarankings', 'researchexcellence', 'springernatureindia', 'therankings2027', 'theworlduniversityrankings', 'universityrankings']
--------------------------------------

========== PART E RESULTS ==========
W shape: (263, 50)
T shape: (20, 50)
Similarity matrix shape: (263, 20)

First text word: manipal
First tag: research
Similarity score: 0.19424225389957428

First row (similarity with all 20 tags):
[0.19424225389957428, 0.23752376437187195, 0.2668042480945587, 0.3900168240070343, 0.1942349672317505, 0.3087814748287201, 0.3053966164588928, 0.2975148856639862, 0.3121398389339447, 0.16783995926380157, 0.3395756781101227, -0.08232572674751282, -0.08520716428756714, 0.25680598616600037, 0.25940343737602234, 0.28712478280067444, 0.21079021692276, 0.3601820766925812, 0.3412853479385376, 0.16525718569755554]

========== SIMILARITY VALIDATION ==========
Total similarity scores: 5260
Minimum similarity: -0.35546305775642395
Maximum similarity: 1.0
All scores within [-1, 1]: True
===========================================

========== PART F RESULTS ==========
NumPy similarity matrix shape: (263, 20)
First similarity score: 0.19424221
====================================

========== PART G VERIFICATION ==========
Vector similarity shape: (263, 20)
NumPy similarity shape: (263, 20)
Both implementations match: False
Maximum absolute difference: 2.980232238769531e-07
=========================================

========== PART H: TOP 8 TAGS ==========
Rank    Tag                   Score          Best Matching Word
-----------------------------------------------------------------
1       research              1.000000       research
2       university            1.000000       university
3       collaboration         1.000000       collaboration
4       medicine              1.000000       medicine
5       education             1.000000       education
6       curriculum            0.840056       teaching
7       students              0.830565       teaching
8       faculty               0.827109       academic
=================================================================

'''