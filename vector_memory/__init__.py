# from __future__ import annotations
import os
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
        return super().get_text()

    def get_embedding(self) -> list[float]:
        return super().get_embedding()
    
    def test_similarity(self, other_memory) -> float:
        return super().test_similarity(other_memory)



class MemoryBank(_rust_mem_store):
    def __init__(self: super, embedding_length, 
                 initial_memories: list[Memory]=[],
                 save_directory:str=None):
        """A bank of 'memories', text with associated vector embeddings.
        for easy search and retrieval.

        Args:
            embedding_length (int): The length of memories being added to the store, 
                will force new memories to have this length
            memories (list[Memory], optional): A list of starter memories
        """
        super().__init__()
        if save_directory:
            super().set_save_folder(os.path.abspath(save_directory))
        

    def __iter__(self):
        for mem in self.memories:
            yield mem
    
    def __getitem__(self, subscript) -> Memory:
        result = self.memories.__getitem__(subscript)
        if isinstance(subscript, slice):
            return list(result)
        else:
            return result
    

    def load_from_folder(self, load_directory=None):
        """load_from_folder

        Args:
            directory_path (str, optional): the path where the memories should load from.
                Defaults to memory bank's default storage directory.
        """
        print(f'Loading from directory: {self.save_directory}')
        super().load_from_folder()


    def save_to_folder(self, save_directory=None, overwrite_directory=True):
        """load_from_folder

        Args:
            directory_path (str, optional): the path where the memories should be saved.
                Defaults to memory bank's default storage directory.
            overwrite_save_dir (bool, optional): Default True. 
                Resets the Memory Bank's save directory to use the supplied directory.
        """
        if not save_directory and self.save_directory:
            save_directory = self.save_directory
        elif overwrite_directory and save_directory:
            super().set_save_folder(save_directory)
        elif overwrite_directory:
            raise Warning("Cannot overwrite save directory with empty value!")
        else:
            raise ValueError("Cannot save to directory, no folder specified!")
        
        print(f"Saving to folder: {save_directory}")
        super().save_to_folder(save_directory)


    def get_save_file_dir(self):
        return self.save_directory
    

    def add_memory(self, memory: Memory):
        """_summary_

        Args:
            memories (Memory): memories to add 
        """
        if len(memory.get_embedding()) != self.embedding_length:
            print("\nWarning! Received embedding of wrong size. Saved result will be a truncated or padded vector.\n")
        super().add_memory(memory)

    def add_memories(self, *args):
        """Adds new memories to the memory store.

        Args:
            1 to N memories to add to the memory store. Also accepts a list.
        """
        mems_to_add = []
        for item in args:
            if hasattr(item, '__iter__'):
                self.add_memories(*item)
            elif isinstance(item, Memory):
                mems_to_add.append(item)
            else:
                print(f"WARNING: this item is not of class Memory and is being skipped:\n{item}")
        
        super().add_memories(mems_to_add)

    
    def get_top_n_matches(self, query_memory: Memory, top_n=5, must_include_text: str = None):
        """Returns similar memories to the query memory, default returns 5

        Returns:
            list[(similarity_score, Memory)]: The top matches, sorted in descending order, including score.
        """
        return super().get_top_n_matches(query_memory, top_n, must_include_text)
    

    