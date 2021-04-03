#part 3
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

texts = ['python', 'pythoner', 'pythoning', 'pythonly']

for text in texts:
    print(ps.stem(text))