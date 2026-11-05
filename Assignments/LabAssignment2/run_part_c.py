
from pathlib import Path

from word2vec.src.w2v import load_model
from preprocessing import preprocess_text


# Get the current project directory
BASE_DIR = Path(__file__).resolve().parent

# File paths
input_path = BASE_DIR / "input.txt"

# Update this filename if your actual model filename is different
model_path = (
    BASE_DIR
    / "word2vec"
    / "glove50"
    / "glove_50_fast.wordvectors"
)


# 1. Load the input text
with open(input_path, "r", encoding="utf-8") as file:
    text = file.read()


# 2. Load the GloVe model
model = load_model(str(model_path))

if model is None:
    raise RuntimeError("Could not load the GloVe model.")


# 3. Run Part C preprocessing
tokens, in_vocab_tokens, oov_tokens = preprocess_text(
    text,
    model
)


# 4. Display some results
print("\nFirst 20 processed tokens:")
print(tokens[:20])

print("\nFirst 20 in-vocabulary tokens:")
print(in_vocab_tokens[:20])

print("\nFirst 20 OOV tokens:")
print(oov_tokens[:20])




'''
Output:

----- TEXT PREPROCESSING RESULTS -----
Tokens before preprocessing: 400
Tokens after preprocessing: 282
In-vocabulary tokens: 263
Out-of-vocabulary tokens: 19
Distinct OOV tokens: ['academicexcellence', 'globalhealth', 'globalrankings', 'highereducation', 'infectiousdiseases', 'institutionofeminence','mahemanipal', 'mahenatureconference', 'manipalacademyofhighereducation', 'onehealth', 'proudmahe', 'qsasia2027', 'qsasiarankings', 'researchexcellence', 'springernatureindia', 'therankings2027', 'theworlduniversityrankings', 'universityrankings']
--------------------------------------

First 20 processed tokens:
['manipal', 'academy', 'higher', 'education', 'mahe', 'institution', 'eminence', 'deemed', 'university', 'has', 'strengthened', 'its', 'standing', 'times', 'higher', 'education', 'world', 'university', 'rankings', '2027']

First 20 in-vocabulary tokens:
['manipal', 'academy', 'higher', 'education', 'mahe', 'institution', 'eminence', 'deemed', 'university', 'has', 'strengthened', 'its', 'standing', 'times', 'higher', 'education', 'world', 'university', 'rankings', '2027']

First 20 OOV tokens:
['theworlduniversityrankings', 'therankings2027', 'institutionofeminence', 'globalrankings', 'mahenatureconference', 'springernatureindia', 'globalhealth', 'infectiousdiseases', 'onehealth', 'researchexcellence', 'highereducation', 'mahemanipal', 'manipalacademyofhighereducation', 'qsasiarankings', 'qsasia2027', 'universityrankings', 'highereducation', 'academicexcellence', 'proudmahe']

'''