from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup

def removeSpecialCharacters(text):
	good = {'\n'}
	for i in "abcdefghijklmnopqrstuvwxyz áéíóúñü":
		good.add(i)
	ans = ""
	for c in text:
		if c in good:
			ans += c
	return ans

def cleanHTML(html):
	return BeautifulSoup(html,'html.parser').get_text().lower()

def opFile(root):
    corpus = PlaintextCorpusReader(root, '.*')
    return corpus.raw()

def splitText(txt):
	return txt.replace('/', ' ').replace('.', ' ').replace('-', ' ')

def lemmaDict(path):
    with open(path, encoding='latin-1') as f:
        lines = f.readlines()
    
    lemm = {}

    for line in lines:
        line = line.strip()
        if line != '':
            words = line.split()
            token = words[0].strip()
            token = token.replace("#", "")
            lemma = words[-1].strip()
            lemm[token] = lemma
    return list(lemm.items())

def lemmatize(text, lemm_dir):
    lemmatized = []
    lemmas = dict(lemmaDict(lemm_dir))
    for word in text:
        if word in lemmas.keys():
            lemmatized.append(lemmas[word])
        else:
            lemmatized.append(word)
    return lemmatized

def comenzar():
    org_txt = opFile("c:/Users/sebas/Desktop/EXCELSIOR_100_files")

    html_clean_text = cleanHTML(org_txt)

    html_clean_text = splitText(html_clean_text)

    html_clean_text = removeSpecialCharacters(html_clean_text)

    tokens = word_tokenize(html_clean_text,"spanish")

    normalized_tokens = []  

    nltk_stop_words = stopwords.words("spanish")

    for w in tokens:  
        if w not in nltk_stop_words:  
            normalized_tokens.append(w)

    lemmatized_tokens = lemmatize(normalized_tokens, "C:/Users/sebas/Downloads/generate.txt")

comenzar()