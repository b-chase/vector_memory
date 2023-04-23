use pyo3::{prelude::*, exceptions::PyWarning, types::PyList};
use rayon::prelude::*;
use std::{fs::File, fmt::format};
use serde::{Serialize, Deserialize};
use bincode::{config, serialize, deserialize};


fn dot_product(a:&Vec<f64>, b:&Vec<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, b)| a*b)
        .sum::<f64>()
}

/// Memory struct for memories to be stored
#[pyclass]
#[derive(Clone)]
struct Memory {
    text: String,
    embed_size: usize, 
    embedding: Vec<f64>, 
    magnitude: f64
}

#[pymethods]
impl Memory {
    #[new]
    fn new(text: String, embed_vector: Vec<f64>, embedding_size: Option<usize>) -> Self {
        let resize_to = embedding_size.unwrap_or(embed_vector.len());
        let mut fixed_embedding = embed_vector.clone();
        fixed_embedding.resize_with(resize_to, Default::default);
        let mag = dot_product(&fixed_embedding, &fixed_embedding).powf(0.5);

        Memory {
            text: text, 
            embed_size: resize_to, 
            embedding: fixed_embedding, 
            magnitude: mag
        }
    }

    fn __repr__(&self) -> String {
        // Just the text of the function
        format!("Embedding vector of length {}\n with text: {}", self.embed_size, self.text)
    }

    fn get_text(&self) -> String {
        self.text.clone()
    }

    fn get_embedding(&self) -> Vec<f64> {
        self.embedding.clone()
    }

    fn compare(&self, other: &Memory) -> f64 {
        // cosine similarity calculations
        assert_eq!(self.embed_size, other.embed_size);
        dot_product(&self.embedding, &other.embedding) / (self.magnitude * other.magnitude)
    }
}


#[pyclass(get_all, dict, sequence)]
struct MemoryStore {
    memories: Vec<Memory>, 
    embedding_size: usize
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new(embedding_length: usize, initial_memories: Option<Vec<Memory>>) -> Self {
        // if initial_memories
        MemoryStore { 
            memories: initial_memories.unwrap_or(Vec::new()), 
            embedding_size: embedding_length
        }
    }

    fn __len__(&self) -> usize {
        self.memories.len()
    }

    #[pyo3(signature=(*args))]
    fn add_memories(&mut self, args: Vec<Memory>) {
        // let new_memory = Memory::new(text, numbers);
        self.memories.extend_from_slice(args.as_slice());
    }

    fn top_n_matches(&mut self, query: &Memory, n: usize, must_include_text: Option<&str>) -> Vec<(f64,Memory)> {
        /*
            Returns the top 'n' Memories by cosine similarity of their embedding vectors to the supplied 'query' memory.
            Optionally include a must-have text string, 'must_include_text', to filter results before searching.
         */
        let mut dist_mem_pairs = self.memories.par_iter()
            .filter_map(|memory| {
                if must_include_text.is_none() ||
                memory.text.to_lowercase().contains(&(must_include_text.unwrap().to_lowercase())) {
                    Some((query.compare(memory), memory.clone()))
                } else {None}
            }).collect::<Vec<(f64, Memory)>>();

            dist_mem_pairs.sort_by_key(|(dist, _mem)| -(1e6*dist) as i32);

            dist_mem_pairs[0..(n.min(dist_mem_pairs.len()))].to_vec()
    }

}


// #[pymodule]
// fn modular_memory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//     m.add_class::<Memory>()?;
//     m.add_class::<MemoryStore>()?;
//     Ok(())
// }
