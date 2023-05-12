
from sentence_transformers import SentenceTransformer
from vector_memory import Memory, MemoryBank
import numpy as np
import torch
import timeit

embedding_model_name = 'sentence-transformers/paraphrase-albert-small-v2'
embedding_model = SentenceTransformer(embedding_model_name)
embedding_lengths = 768

bank = MemoryBank(embedding_lengths)

bank.load_memories('test_memory_bank')

memory_list = [mem for mem in bank]

# print("Testing rust simliarity calcs: {} seconds".format(timeit.timeit("test_rust_similarity()", globals=locals(), number=100)))
# print("Testing torch similarity calcs: {} seconds".format(timeit.timeit("test_torch_similarity()", globals=locals(), number=100)))

table = np.zeros((6,6))

for i in range(6):
    print(memory_list[i])
    # print(memory_list[i].get_text())
    for j in range(i,6):
        table[i,j] = memory_list[i].test_similarity(memory_list[j])

print(table)

bank.save_memories('test_memory_bank')
