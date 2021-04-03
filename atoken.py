
#part 1
from nltk.tokenize import sent_tokenize, word_tokenize

examples = "Hello Mr. John, how are you?"

print(sent_tokenize(examples))
print(word_tokenize(examples))

for example in word_tokenize(examples):
    print(example)
