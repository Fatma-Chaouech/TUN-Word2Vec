
from gensim.models import Word2Vec 
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

CORPUS_PATH = "TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset/transliterated_dialect.txt"
MODEL_PATH = "aravec/spacyModel/aravec.txt"
DEST_PATH = "aravec/fine-tuned/finetuned_model.txt"
print("Reading the corpus...")
corpus = []
with open(CORPUS_PATH) as f:
    lines = f.readlines()
    for row in lines :
        row = row.split()
        corpus.append(row)


print("Building a new model...")

model_2 = Word2Vec(corpus, vector_size=300, min_count=1)
# model_2.build_vocab(corpus)
total_examples = model_2.corpus_count
vocab = list(model_2.wv.key_to_index.keys())

print("Loading an existing model...")
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
model_2.build_vocab([list(model.key_to_index.keys())], update=True)

print("Fine-tuning the model...")
model_2.train(corpus, total_examples=total_examples, epochs= 20)
model_2.wv.save_word2vec_format(DEST_PATH)
word_embeddings = np.array([ model_2.wv[k] if k in model_2.wv else np.zeros(100) for k in vocab ])
print(word_embeddings.shape) # Should be len(vocab) by 100
print(len(vocab)/100)
print(model_2.wv.similarity('ولد','راجل'))