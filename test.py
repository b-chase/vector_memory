
from sentence_transformers import SentenceTransformer
from vector_memory import Memory, MemoryBank
import numpy as np
import torch
import re
import timeit
import os

embedding_model_name = 'sentence-transformers/paraphrase-albert-small-v2'
embedding_model = SentenceTransformer(embedding_model_name)
embedding_lengths = 768

bank = MemoryBank(embedding_lengths)


def ingest_folder(target_folder: str, split_on_paragraph=False, file_ext_list=['.txt']) -> list[Memory]:
    """ingest_folder

    Args:
        target_folder (str): The folder containing text files that you want to ingest.
        file_ext_list (list[str]): A list of file extensions you want to include. 
            Defaults to only '.txt' files.
    """
    text_inputs = []
    outputs = []
    for file in os.listdir(target_folder):
        file_path = os.path.join(target_folder, file)
        if not os.path.isdir(file_path) and os.path.splitext(file)[1] in file_ext_list:
            with open(file_path, 'r') as f:
                file_text = f.read()
            if split_on_paragraph:
                text_parts = re.split(r'[\n\r\.]+', file_text)
                text_inputs.extend([t.strip() for t in text_parts if re.match('[A-Za-z]', t)])
            else:
                text_inputs.append(file_text.strip())
    
    for i, t in enumerate(text_inputs):
        perc = (i+1) / len(text_inputs)
        perc_bars = int(25*perc)
        print(f'Progress: ({i}/{len(text_inputs)}) [{perc_bars*"="}{(25-perc_bars)*" "}] {int(perc*100)}%', end='\r')
        embed_vector = embedding_model.encode(t)
        outputs.append(Memory(t, embed_vector))
        # break
    print('\n')
    return outputs

bank.add_memories(ingest_folder('test_samples', True))


for mem in bank[0::25]:
    print("Query Memory: ")
    print(mem)
    print("Top Matches:")
    top_matches = bank.get_top_n_matches(mem,3)
    print('\n'.join([f'matches with {int(100*x[0])}%: {x[1]}' for x in top_matches]))
    print('\n\n')

quit()

memory_list = []

for i in range(6) :
    with open(f'sample_text{i+1}.txt', 'r') as f:
        text = f.read()
        embedding = embedding_model.encode(
            text, 
            show_progress_bar=True, 
            # convert_to_numpy=True
        )
        # print(embedding)
        # quit()
        mem = Memory(text, embedding)
        memory_list.append(mem)
        bank.add_memory(mem)
        
        print(f'\nFile {f.name} gives embedding of length {len(embedding)}')


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


# print("Testing rust simliarity calcs: {} seconds".format(timeit.timeit("test_rust_similarity()", globals=locals(), number=100)))
# print("Testing torch similarity calcs: {} seconds".format(timeit.timeit("test_torch_similarity()", globals=locals(), number=100)))

table = np.zeros((6,6))

memory_list.sort(key=lambda x: x.get_text())
for i in range(6):
    mem1 = memory_list[i]
    print(mem1.get_text().split('\n')[0])
    for j in range(i,6):
        table[i,j] = memory_list[i].test_similarity(memory_list[j])

print(table)

bank.save_to_folder('test_memory_bank', True)

bank2 = MemoryBank(embedding_lengths, save_directory=bank.get_save_file_dir())

bank2.load_from_folder()

bank2_list = [mem for mem in bank2]
bank2_list.sort(key=lambda x: x.get_text())
table = np.zeros((6,6))

for i in range(6):
    mem1 = bank2_list[i]
    print(mem1.get_text().split('\n')[0])
    for j in range(i,6):
        table[i,j] = bank2_list[i].test_similarity(bank2_list[j])

print(table)