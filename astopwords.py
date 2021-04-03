#part 2
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

text = "this is a test code for stopwords"

stopwords = set(stopwords.words("english"))

#print(stopwords)

words = word_tokenize(text)

filtered = []

for word in words:
    if word not in stopwords:
        filtered.append(word)

#can be written in one line
#filtered_sentence = [word for word in words if not word in stopwords]
#print(filtered_sentence)
