from sentence_transformers import SentenceTransformer
from vector_memory import Memory, MemoryBank
import random
import numpy as np
import torch
import timeit
from collections import Counter

random.seed(42)


embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',)
embedding_lengths = 1024

bank = MemoryBank(embedding_lengths)

memory_list = []

with open('sample_text1.txt', 'r') as f:
    word_list = f.read().split()

seen_words = set()
for w in word_list:
    if w in seen_words:
        continue
    seen_words.add(w)
    test_embed = [random.random() for _ in range(embedding_lengths)]
    test_mem = Memory(w, test_embed, embedding_lengths)
    memory_list.append(test_mem)
    bank.add_memory(test_mem)


query_mem = memory_list[0]
print(query_mem)
print(query_mem.embedding[0:5])

def rust_find_similar():
    return bank.search_memories_like(query_mem)

get_cosine = torch.nn.CosineSimilarity(0)

def torch_find_similar(): #memory_list: list[Memory], query_mem: Memory, top_n=5):
    dist_mems = [(get_cosine(torch.tensor(query_mem.embedding, dtype=float), torch.tensor(check_mem.embedding, dtype=float)), check_mem) for check_mem in memory_list]
    return sorted(dist_mems, key=lambda t: t[0], reverse=True)[0:5]

def npfft_find_similar():
    fft1 = np.fft.fft(query_mem.embedding).flatten()[0]
    dist_mems = [(np.dot(query_mem.embedding, check_mem.embedding)/(fft1*np.fft.fft(check_mem.embedding).flatten()[0]), check_mem) for check_mem in memory_list]
    return sorted(dist_mems, key=lambda t: t[0], reverse=True)[0:5]

def print_results(result_set):
    print('\n'.join([str(x) for x in result_set]))

print("Testing rust search calcs:", timeit.timeit("rust_find_similar()", globals=locals(), number=20))
# print_results(rust_find_similar())
print("Testing torch search calcs:", timeit.timeit("torch_find_similar()", globals=locals(), number=20))
# print_results(torch_find_similar())
print("Testing np fft search calcs:", timeit.timeit("npfft_find_similar()", globals=locals(), number=20))
# print_results(npfft_find_similar())