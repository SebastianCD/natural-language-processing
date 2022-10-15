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

def encontrarContextos(lemmatized_tokens):
    contextos = dict()

    for sentence in lemmatized_tokens:
        sentence = set(sentence)
        sentence = list(sentence)
        for word in sentence:
            if word not in contextos:
                contextos[word] = []
            for id in range(0,len(sentence)):
                if word != sentence[id]:
                    contextos[word].append(sentence[id])
    
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

def clearHTML(html):
	return BeautifulSoup(html,'html.parser').get_text().lower()

def splitText(txt):
	return txt.replace('/', ' ').replace('.', '').replace('—', ' ')

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

def lemmatize(text, dic):
    lemmatized = []
    lemmas = dic
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

    palabra = "poder"

    ans = []
    v1 = vectors[palabra]

    for word in voc:
        v2 = vectors[word]
        cosine = numpy.dot(v1, v2) / (numpy.sqrt(numpy.sum(v1 ** 2)) * (numpy.sqrt(numpy.sum(v2 ** 2))))
        ans.append((word, cosine))
    ans.sort(key = lambda x: x[1], reverse=True)
    return ans

def imprimirResultados(similares):
    for pares in similares:
        if pares[1] != 0.0:
            print(pares)

def comenzar():
    html_original = openFile("c:/Users/sebas/Desktop/EXCELSIOR_100_files")

    html_limpio = clearHTML(html_original)

    sentences = sent_tokenize(html_limpio, "spanish")

    new_sentences = []

    for sentence in sentences:
        if sentence.rfind('\n') != -1:
            sentence = sentence[sentence.rfind('\n') + 1:len(sentence)]
        sentence = splitText(sentence)
        sentence = removeSpecialCharacters(sentence)
        new_sentences.append(sentence)

    tokenized_sentences = []

    for sentence in new_sentences:
        tokenized_sentences.append(word_tokenize(sentence, "spanish"))

    stop_words = stopwords.words("spanish")

    normalized_tokens = []

    for sentence in tokenized_sentences:
        normalized_tokens.append(removeStopWords(sentence, stop_words))

    lemmatized_tokens = []

    dic = dict(lemmaDict("C:/Users/sebas/Downloads/generate.txt"))

    for sentence in normalized_tokens:
        lemmatized_tokens.append(lemmatize(sentence, dic))

    voc = set()

    for sentence in lemmatized_tokens:
        for word in sentence:
            voc.add(word)

    voc = sorted(voc)

    con = encontrarContextos(lemmatized_tokens)

    similares = encontrarSimilares(voc, con)

    for par in similares:
        print(par)
