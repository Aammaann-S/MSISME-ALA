from pathlib import Path

from word2vec.src.w2v import load_model
from preprocessing import preprocess_text
from matrices import build_tag_matrix, build_text_matrix

# Import your existing 20 tags from vec.py
from vec import TAGS


# Find the project directory
BASE_DIR = Path(__file__).resolve().parent

# Define file paths
input_path = BASE_DIR / "input.txt"

model_path = (
    BASE_DIR
    / "word2vec"
    / "glove50"
    / "glove_50_fast.wordvectors"
)


# 1. Read input text
with open(input_path, "r", encoding="utf-8") as file:
    text = file.read()


# 2. Load GloVe model
model = load_model(str(model_path))

if model is None:
    raise RuntimeError("Could not load GloVe model.")


# 3. Part C: Preprocess text
tokens, _, _ = preprocess_text(text, model)


# 4. Part D: Build tag matrix
tag_names, T = build_tag_matrix(model, TAGS)


# 5. Part D: Build text matrix
in_vocab_tokens, oov_tokens, W = build_text_matrix(
    model, tokens
)


# 6. Display results
print("\n========== PART D RESULTS ==========")

print("\nTag names:")
print(tag_names)

print("\nTag matrix shape:")
print(T.shape)

print("\nText matrix shape:")
print(W.shape)

print("\nNumber of tag vectors:", len(tag_names))
print("Number of text vectors:", len(in_vocab_tokens))
print("Number of OOV tokens:", len(oov_tokens))

print("\nFirst tag vector:")
print(T[0])

print("\nFirst text word:", in_vocab_tokens[0])
print("Its vector:")
print(W[0])

print("\n====================================")


'''
Output:

----- TEXT PREPROCESSING RESULTS -----
Tokens before preprocessing: 400
Tokens after preprocessing: 282
In-vocabulary tokens: 263
Out-of-vocabulary tokens: 19
Distinct OOV tokens: ['academicexcellence', 'globalhealth', 'globalrankings', 'highereducation', 'infectiousdiseases', 'institutionofeminence','mahemanipal', 'mahenatureconference', 'manipalacademyofhighereducation', 'onehealth', 'proudmahe', 'qsasia2027', 'qsasiarankings', 'researchexcellence', 'springernatureindia', 'therankings2027', 'theworlduniversityrankings', 'universityrankings']
--------------------------------------

========== PART D RESULTS ==========

Tag names:
['research', 'innovation', 'education', 'university', 'students', 'faculty', 'campus', 'engineering', 'medicine', 'technology', 'curriculum', 'collaboration', 'publication', 'laboratory', 'scholarship', 'mentorship', 'internship', 'entrepreneurship', 'accreditation', 'alumni']

Tag matrix shape:
(20, 50)

Text matrix shape:
(263, 50)

Number of tag vectors: 20
Number of text vectors: 263
Number of OOV tokens: 19

First tag vector:
[ 0.71258    0.64492    0.053186   0.30578   -0.058139   0.37774
 -0.9848    -1.4113     1.2767    -0.013141   0.55534    0.23268
 -0.013911  -0.0090835 -0.60868    0.11358    0.45878    0.42078
 -0.52664    0.54239    0.9867     0.26386    0.22037   -0.82848
  0.24685   -1.288     -0.36937   -1.0895    -1.447      0.043814
  3.1056    -0.60116    0.19298   -1.9592    -0.55103   -0.47018
 -0.70315    0.95543    1.3444     0.15053   -0.048638  -0.85941
  0.42482    0.45963    0.95505   -0.19284    0.52146    1.1693
 -0.17361    0.89113  ]

First text word: manipal
Its vector:
[-0.058668 -0.87516  -0.87704  -0.25932  -0.98647  -0.012704  0.64039
 -0.31657   0.38201  -0.3149    0.79236  -0.80948   0.7264   -1.5772
 -1.0743    1.1723   -0.35353   0.89991   0.20829   0.95243   0.21103
  0.36635  -0.66805   0.81545   0.70104   0.0326    0.18101  -0.42302
 -2.0226   -0.18041  -0.46369   0.55379   0.29032  -0.22201  -0.073433
 -0.087917 -0.5465    0.6321    1.2066   -0.56281   0.30238  -0.64505
 -0.47796   0.9053    0.041003  0.74679   1.1543    0.4676    0.62839
 -1.2707  ]

====================================

'''