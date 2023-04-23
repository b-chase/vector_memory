use pyo3::{prelude::*, exceptions::PyWarning, types::PyList};
use rayon::prelude::*;
use std::{fs::File, fmt::format};
use serde::{Serialize, Deserialize};
use bincode::{config, serialize, deserialize};

const EMBED_LEN : usize = 1536;

fn dot_product(a:&Vec<f64>, b:&Vec<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, b)| a*b)
        .sum::<f64>()
}

/// Memory struct for memories to be stored
#[derive(Clone)]
#[pyclass]
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
        return Memory {
            text: text, 
            embed_size: resize_to, 
            embedding: fixed_embedding, 
            magnitude: dot_product(&fixed_embedding, &fixed_embedding).powf(0.5)
        };
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
    embed_len: usize
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new(embedding_length: usize, initial_memories: Option<PyList>) -> Self {
        // if initial_memories
        MemoryStore { 
            memories: Vec::new(), 
            embed_len: embedding_length
        }
    }

    fn __len__(&self) -> usize {
        self.memories.len()
    }

    #[pyo3(signature=(*args))]
    fn add_memories(&mut self, args: PyList) {
        // let new_memory = Memory::new(text, numbers);
        for i in 0..(args.len().unwrap_or(0)) {
            let new_mem = args.get_item(i).unwrap();
            self.memories.push(new_mem.extract().unwrap());
        }
    }

    fn top_n_matches(&mut self, query: &Memory, n: usize, must_include_text: Option<&str>) -> Vec<(f64,Memory)> {
        
        let mut dist_index_pairs = self.memories.par_iter()
            .filter_map(|memory| {
            if memory.text.contains(must_include_text.unwrap_or_default()) {
                Some(query.compare(memory), memory.to_owned())
            } else { None }
            })
            // .map(|(index, memory)| (memory.compare(query), ))
            .collect::<Vec<(f64, Memory)>>();
        dist_index_pairs.sort_by_key(|(dist, index)| -(1e6*dist) as i32);

        if let Some(text) = must_include_text {
            let mut filtered_pairs = Vec::new();
            for (distance, index) in dist_index_pairs {
                if memory.memory_json.contains(text) {
                    filtered_pairs.push((distance, memory));
                }
            }
            dist_pairs = filtered_pairs;
        }

        dist_pairs[0..n].to_vec()
    }

}


// #[pymodule]
// fn modular_memory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
//     m.add_class::<Memory>()?;
//     m.add_class::<MemoryStore>()?;
//     Ok(())
// }
