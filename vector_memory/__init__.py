# from __future__ import annotations
from .vector_memory import Memory as _rust_mem
from .vector_memory import MemoryStore as _rust_mem_store

# Define memory class and available methods
class Memory(_rust_mem):
    def __init__(self, text: str, embedding: list[float], embed_vector_len:int=None):
        """A unit of stored context or "memory", including its numerical representation, or "embedding".

        Args:
            text (str): The text to be stored or a key to the text.
            embedding ([float]): 
            embed_vector_len (int, optional): Force the length of embedding vector by truncating or padding with zeroes. Defaults to None.
        """
        super().__init__()
    
    def get_text(self) -> str :
        return self.text

    def get_embedding(self) -> list[float]:
        return self.embedding
    
    def similarity(self, other_memory) -> float:
        return self._compare(other_memory)



class MemoryBank(_rust_mem_store):
    def __init__(self: super, embedding_length, initial_memories: list[Memory]):
        """A bank of 'memories', text with associated vector embeddings, for easy search and retrieval.

        Args:
            embedding_length (int): The length of memories being added to the store, will force new memories to have this length
            memories (list[Memory], optional): A list of starter memories
        """
        super().__init__()
        
    
    def __iter__(self):
        for mem in self.memories:
            yield mem
    
    def add_memory(self, memory: Memory):
        """_summary_

        Args:
            memories (Memory): memories to add 
        """
        if len(memory.get_embedding) != self.embedding_length:
            print("\nWarning! Received embedding of wrong size. Saved result will be a truncated or padded vector.\n")
        self._add_memory(memory)

    def add_memories(self, *args):
        """Adds new memories to the memory store.

        Args:
            memory (Memory): _description_
        """
        mems_to_add = []
        for item in args:
            if hasattr(item, '__iter__'):
                self.add_memories(item)
            elif isinstance(item, Memory):
                mems_to_add.append(item)
            else:
                print(f"WARNING: this item is not of class Memory and is being skipped:\n{item}")
        
        for mem in mems_to_add:
            print(f"Adding new memory:\n{mem}\n")
            self._add_memory(mem)

    
    def search_memories_like(self, query_memory: Memory, top_n=5, must_include_text: str = None):
        """Returns similar memories to the query memory, default returns 5

        Returns:
            list[Memory]: The top matches.
        """
        return self._top_n_matches(query_memory, top_n, must_include_text)
    