
from sentence_transformers import SentenceTransformer
from vector_memory import Memory, MemoryBank
import numpy as np
import torch
import timeit



embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',)
embedding_lengths = 1024

bank = MemoryBank(embedding_lengths)

memory_list = []

for i in range(6) :
    with open(f'sample_text{i+1}.txt', 'r') as f:
        text = f.read()
        embedding = embedding_model.encode(
            text, 
            show_progress_bar=True, 
            # convert_to_numpy=True
        )
        mem = Memory(text, embedding)
        memory_list.append(mem)
        bank.add_memory(mem)
        
        print(f'\nFile {f.name} gives embedding of length {len(embedding)}:\n{embedding}')


def test_rust_similarity():
    results = []
    for i, mem in enumerate(memory_list):
        for j in range(6):
            results.append(f"Similarity between memory {i+1} and memory {j+1}: {mem.test_similarity(memory_list[j])}")
    return results


torch_cosine_sim = torch.nn.CosineSimilarity(dim=0)
def test_torch_similarity():
    results = []
    for i, mem in enumerate(memory_list):
        for j in range(6):
            results.append(
                f"Similarity between memory {i+1} and memory {j+1}: {torch_cosine_sim(torch.tensor(mem.embedding), torch.tensor(memory_list[j].embedding))}")
    return results


print("Testing rust simliarity calcs:", timeit.timeit("test_rust_similarity()", globals=locals(), number=100))
print("Testing torch similarity calcs:", timeit.timeit("test_torch_similarity()", globals=locals(), number=100))

