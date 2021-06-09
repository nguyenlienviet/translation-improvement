import enchant
import string

d = enchant.Dict('en_US')

with open('vocab.txt') as f:
    vocab = set()
    for word in f.readlines():
        word = word.strip()
        word_no_punct = word.strip(string.punctuation)
        if word_no_punct != '':
            word = word_no_punct
        if d.check(word):
            vocab.add(word)

with open('vocab.txt', 'w') as f:
    f.write('\n'.join(sorted(vocab)))
