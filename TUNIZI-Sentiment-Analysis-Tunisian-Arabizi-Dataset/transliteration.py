from aaransia import transliterate
import re
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from translate import Translator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import sys
sys.path.insert(0, '/home/pipanther/Desktop/Tunisian Dialect/aravec/')
import utilities
nltk.download('stopwords')
nltk.download('punkt')

def get_lang_detector(nlp, name):
    return LanguageDetector()  # We use the seed 42

PATH = "TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset/"
file_name = "TUNIZI-Dataset.txt"
labels = []
text = []


nlp = spacy.load("en_core_web_sm")
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe('language_detector', last=True)

#def filter_language(text):
#	translated = []
#	text = text[1:]
#	text = text.split()
#	for word in text:
#		doc = nlp(word)
#		if doc._.language['language'] == "fr" or doc._.language['language'] == "en":
#			continue
#		else :
#			translated.append(word)
#	translated = ' '.join(translated)
			
#	return translated


def preprocess_tunisian(text):
	text = re.sub("a[h]+","ah",text,flags=re.IGNORECASE)
	text = re.sub("hh","haha",text,flags=re.IGNORECASE)
	text = re.sub("hahaha","haha",text,flags=re.IGNORECASE)
	text = re.sub("ana","ena",text,flags=re.IGNORECASE)
	text = re.sub("fil","fi el",text,flags=re.IGNORECASE)
	text = re.sub("ou","w",text,flags=re.IGNORECASE)
	text = re.sub("yerhmou","yarahmou",text,flags=re.IGNORECASE)
	text = re.sub("wou","w",text,flags=re.IGNORECASE)
	text = re.sub("é","e",text,flags=re.IGNORECASE)
	text = re.sub("jy","jey",text,flags=re.IGNORECASE)
	text = re.sub("choufli hal","chouflihal",text,flags=re.IGNORECASE)
	text = re.sub("y7ebe","y7eb",text,flags=re.IGNORECASE)
	text = re.sub("y7ote","y7ot",text,flags=re.IGNORECASE)
	text = re.sub("inti","enti",text,flags=re.IGNORECASE)
	text = re.sub("alah","allah",text,flags=re.IGNORECASE)
	text = re.sub("sboui","sbou3i",text,flags=re.IGNORECASE)
	text = re.sub("lake","nheb",text,flags=re.IGNORECASE)
	text = re.sub("j aime","nheb",text,flags=re.IGNORECASE)
	text = re.sub("jem","nheb",text,flags=re.IGNORECASE)
	text = re.sub("laik","nheb",text,flags=re.IGNORECASE)
	text = re.sub("yar7meke","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yarahmik","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("wlh","wallah",text,flags=re.IGNORECASE)
	text = re.sub("yrhmik","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yahahamke","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("y7b","yheb",text,flags=re.IGNORECASE)
	text = re.sub("alahi","allah",text,flags=re.IGNORECASE)
	text = re.sub("yar7mk","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yarhemk","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yarhemk","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yar7mek","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yarmik","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yar7mou","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("yr7mk","yarahmek",text,flags=re.IGNORECASE)
	text = re.sub("allh","allah",text,flags=re.IGNORECASE)
	text = re.sub("chfli","choufli",text,flags=re.IGNORECASE)
	text = re.sub("layk","nheb",text,flags=re.IGNORECASE)
	text = re.sub("bravo","sahit",text,flags=re.IGNORECASE)
	text = re.sub("brqvo","sahit",text,flags=re.IGNORECASE)
	text = re.sub("ma7leha","mahleha",text,flags=re.IGNORECASE)
	text = re.sub("ma7lek","mahlek",text,flags=re.IGNORECASE)
	return text



def preprocess(line) : 


	line = re.sub(r'(.)\1+', r'\1', line)
	line = line.replace('8','gh')
	line = re.sub('\d+','',line)
	line = line.replace('ı','i')
	line = re.sub('[©§¨ª“¯»¢—¬´®-]','',line)
	line = re.sub('[ÅÄÃ]','a',line)
	line = line.replace('ô','o')
	line = line.replace('œ','oe')
	line = preprocess_tunisian(line)
	trans_text = transliterate(line, source='tn', target='ar')
	line = re.sub('[A-Z]','',line)
	clean_text = utilities.clean_str(trans_text)
	stop_words = set(stopwords.words('arabic'))
	clean_text = [w for w in clean_text.split() if not w in stop_words]
	tokens = sent_tokenize(' '.join(clean_text))
	return tokens


with open(PATH+file_name) as file:
	lines = file.readlines()
	lines = list(set(lines))
	
	for row in lines:
		txt = row.split(";")
		labels.append(txt[0])
		line = preprocess(txt[1])
		try :
			# ngrams = utilities.get_all_ngrams(' '.join(line),2)
			text.append(line)	
		except :
			print("An exception occurred")
			break


f_text = open(PATH+"transliterated_dialect.txt", "w")
f_text.truncate()
# f_label = open(PATH+"labels.txt", "w")
# f_label.truncate()
index = 0


for txt in text :
	if str(txt) != '[]':
		txt = ' '.join(txt)
		f_text.write(txt+'\n')
		#f_label.write(labels[index]+'\n')
	index += 1

f_text.close()
# f_label.close()

