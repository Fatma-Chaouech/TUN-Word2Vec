from gensim.models import KeyedVectors
from aaransia import transliterate
import sys
sys.path.insert(0, '/home/pipanther/Desktop/Tunisian Dialect/TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset/')
import transliteration

MODEL_PATH = "aravec/fine-tuned/finetuned_model.txt"

model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)

print("ماكوم : " + str(model.most_similar('ماكوم',topn = 8)))
print("لبلد : " + str(model.most_similar('لبلد',topn = 8)))
print("لمرا : " + str(model.most_similar('لمرا',topn = 8)) )
print("سبوي : " + str(model.most_similar('سبوي',topn = 8)))
print("بارشا : " + str(model.most_similar('بارشا',topn = 8)))
print(model.similarity('نرس','نهب'))

word = transliteration.preprocess('nheb')
similar = str(model.most_similar(word,topn = 8))
print("nheb : " + similar)
