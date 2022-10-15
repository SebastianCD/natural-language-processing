from nltk.tokenize import sent_tokenize, word_tokenize  
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import numpy

def openFile(root):
    corpus = PlaintextCorpusReader(root, '.*')
    return corpus.raw()

def vocabulario(tokens):
	vocabulary = sorted(set(tokens))
	return vocabulary

def encontrarContextos(voc, window):
    contextos = dict()
    id = 0

    for id in range(len(voc)):
        word = voc[id]

        if word not in contextos:
            contextos[word] = []
        start = max(0, id - window)
        end = min(id + window, len(voc) - 1)

        for i in range(start, end + 1):
            if i != id:
                contextos[word].append(voc[i])

    return contextos

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

def removeStopWords(tokens, stop_words):
    normalized_tokens = []

    for w in tokens:  
        if w not in stop_words:  
            normalized_tokens.append(w)

    return normalized_tokens

def encontrarSimilares(voc, con):
    vectors = dict()

    for target, context in con.items():
        
        vectors2 = dict()
        
        for word in context:
            if word not in vectors2:
                vectors2[word] = 0
            vectors2[word] += 1

        l = numpy.zeros(len(voc))

        for i in range(len(voc)):
            word = voc[i]
            if word in vectors2:
                l[i] = vectors2[word]

        vectors[target] = l

    palabra = "banco"

    ans = []
    v1 = vectors[palabra]

    print(v1)

    for word in voc:
        v2 = vectors[word]
        cosine = numpy.dot(v1, v2) / (numpy.sqrt(numpy.sum(v1 ** 2)) * numpy.sqrt(numpy.sum(v2 ** 2)))
        ans.append((word, cosine))

    ans.sort(key = lambda x: x[1], reverse=True)

    return ans

def imprimirResultados(similares):
    for pares in similares:
        if pares[1] != 0.0:
            print(pares)

def comenzar():
    html_original = openFile("c:/Users/sebas/Desktop/EXCELSIOR_100_files")

    html_limpio = cleanHTML(html_original)

    html_limpio = splitText(html_limpio)

    html_limpio = removeSpecialCharacters(html_limpio)

    tokens = word_tokenize(html_limpio, "spanish")

    stop_words = stopwords.words("spanish")

    normalized_tokens = removeStopWords(tokens, stop_words)

    lemmatized_tokens = lemmatize(normalized_tokens, "C:/Users/sebas/Downloads/generate.txt")

    voc = vocabulario(lemmatized_tokens)

    con = encontrarContextos(lemmatized_tokens, 4)

    similares = encontrarSimilares(voc, con)

    imprimirResultados(similares)

comenzar()