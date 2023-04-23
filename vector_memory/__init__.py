# from __future__ import annotations
from .vector_memory import Memory as _rust_mem
from .vector_memory import MemoryStore as _rust_mem_store

# Define memory class and available methods
class VMemory(_rust_mem):
    def __init__(self, text: str, embedding: list[float], embed_vector_len:int=None):
        """A unit of stored context or "memory", including its numerical representation, or "embedding".

        Args:
            text (str): The text to be stored or a key to the text.
            embedding ([float]): 
            embed_vector_len (int, optional): Force the length of embedding vector by truncating or padding with zeroes. Defaults to None.
        """
        super().__init__(text, embedding, embed_vector_len)
    
    def get_text(self) -> str :
        return self.text
    

    def get_embedding(self) -> list[float]:
        return self.embedding
    
    def similarity(self, other_memory) -> float:
        self._compare(other_memory)
        
    
    
    
    
    
    
    
    