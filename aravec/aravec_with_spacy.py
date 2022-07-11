# -*- coding: utf-8 -*-
"""aravec-with-spacy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eAAbnNVnzeqIjGWlELLggKDVhsWwnycy

## Integrate [AraVec](https://github.com/bakrianoo/aravec) with [Spacy.io](https://spacy.io/)

This notebook demonstrates how to integrate an [AraVec](https://github.com/bakrianoo/aravec) model with [spaCy.io](https://spacy.io/)

## Outlines

- Install/Load the required modules
- Load AraVec
- Export the Word2Vec format + gzip it.
- Initialize the spaCy model using AraVec vectors
- Run Your AraVec Spacy Model
- Test the Model

## Install/Load the required modules
"""


import gensim
import re
import spacy

# Clean/Normalize Arabic Text
def clean_str(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']
    
    #remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])
    
    #trim    
    text = text.strip()

    return text

"""## Load AraVec
Download a model from the [AraVec Repository](https://github.com/bakrianoo/aravec), then follow the below steps to load it.
"""


"""
# load the AraVec model
model = gensim.models.Word2Vec.load("./full_grams_cbow_100_twitter/full_grams_cbow_100_twitter.mdl")
print("We've",len(model.wv.index_to_key),"vocabularies")

"""## Export the Word2Vec format + gzip it."""

# Commented out IPython magic to ensure Python compatibility.
# make a directory called "spacyModel"
# %mkdir spacyModel

# export the word2vec fomart to the directory
"""
model.wv.save_word2vec_format("./spacyModel/aravec.txt")

# using `gzip` to compress the .txt file

"""## Initialize the spaCy model using AraVec vectors

"""
- This will create a folder called `/spacy.aravec.model` within your current working directory.
- This step could take several minutes to be completed.
"""

# !python3 -m spacy  init vectors ar ./spacyModel/aravec.txt.gz spacy.aravec.model

"""## Run Your AraVec Spacy Model

"""

# load AraVec Spacy model
nlp = spacy.load("./spacy.aravec.model/")

# Define the preprocessing Class
class Preprocessor:
    def __init__(self, tokenizer, **cfg):
        self.tokenizer = tokenizer

    def __call__(self, text):
        preprocessed = clean_str(text)
        return self.tokenizer(preprocessed)

# Apply the `Preprocessor` Class
nlp.tokenizer = Preprocessor(nlp.tokenizer)

"""## Test the Model"""

# Test your model
nlp("قطة").vector

egypt = nlp("مصر")
tunisia = nlp("تونس")
apple = nlp("تفاح")

print("egypt Vs. tunisia = ", egypt.similarity(tunisia))
print("egypt Vs. apple = ", egypt.similarity(apple))

"""## Done !!

Congratulations, now you have your AraVec model running on spaCy.
"""


