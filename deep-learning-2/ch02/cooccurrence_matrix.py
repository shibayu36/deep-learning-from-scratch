import append_path
import numpy as np
from common.util import preprocess, create_co_matrix

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

C = create_co_matrix(corpus, len(word_to_id))
print(C)
