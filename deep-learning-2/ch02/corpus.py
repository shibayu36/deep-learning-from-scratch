text = "You say goodbye and I say hello."
text = text.replace(".", " .")

words = text.split(" ")

word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

print(
    word_to_id
)  # {'You': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'I': 4, 'hello': 5, '.': 6}
print(id_to_word)

import numpy as np

corpus = np.array([word_to_id[w] for w in words])
print(corpus)
