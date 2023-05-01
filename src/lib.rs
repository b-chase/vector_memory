use pyo3::prelude::*;
use rayon::prelude::*;
use std::io::{Write, Read};
use std::fs::File;
// use serde::{Serialize, Deserialize};
use std::path::Path;
use std::collections::HashMap;


fn dot_product(a:&Vec<f64>, b:&Vec<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, b)| a*b)
        .sum::<f64>()
}

static COMPRESSION_MAP: [char; 65] = [
    '0', '1', '2', '3', '4', '5', '6', '7', 
    '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 
    'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z', '-', '_',
    ' '
];


fn compress_embedding(input_embedding: &Vec<f64>) -> String {
    // takes vector of float64s as input, outputs the vector compressed down to a text string that can be used as a filename    
    let n = input_embedding.len();

    let compressed_to_string: String = input_embedding.par_chunks(64)
        .map(|chunk_x| -> char {
            let index: usize = chunk_x.iter().map(|x| if x.is_sign_positive() {1} else {0}).sum();
            COMPRESSION_MAP[index]
        })
        .collect();

    compressed_to_string

}

/// Memory struct for memories to be stored
// #[derive(Serialize, Deserialize)]
#[pyclass(subclass)]
#[derive(Clone)]
struct Memory {
    text: String,
    embed_size: usize, 
    embedding: Vec<f64>, 
    compressed_name: String, 
    magnitude: f64
}

#[pymethods]
impl Memory {
    #[new]
    fn new(text: String, embed_vector: Vec<f64>, embedding_length: Option<usize>) -> Self {
        let resize_to = embedding_length.unwrap_or(embed_vector.len());
        let mut fixed_embedding = embed_vector.clone();
        fixed_embedding.resize_with(resize_to, Default::default);
        let mag = dot_product(&fixed_embedding, &fixed_embedding).powf(0.5);
        let compressed_name = compress_embedding(&fixed_embedding);

        Memory {
            text: text, 
            embed_size: resize_to, 
            embedding: fixed_embedding, 
            compressed_name: compressed_name, 
            magnitude: mag
        }

    }

    fn __repr__(&self) -> String {
        // Just the text of the function
        format!("Text: {}\nEmbedding vector with length: {}", self.text, self.embed_size)
    }

    fn _get_text(&self) -> String {
        self.text.clone()
    }

    fn _get_embedding(&self) -> Vec<f64> {
        self.embedding.clone()
    }

    fn _compare(&self, other: &Memory) -> f64 {
        // cosine similarity calculations
        assert_eq!(self.embed_size, other.embed_size);
        dot_product(&self.embedding, &other.embedding) / (self.magnitude * other.magnitude)
    }
}


#[pyclass(get_all, dict, sequence, subclass)]
struct MemoryStore {
    memories: Vec<Memory>, 
    embedding_length: usize, 
    save_path: Option<String>
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new(embedding_length: usize, initial_memories: Option<Vec<Memory>>, save_path: Option<String>) -> Self {
        // if initial_memories
        // println!("Debug Rust: found initial memories of length: {}", &(initial_memories.clone().unwrap_or(Vec::new()).len()));
        MemoryStore { 
            embedding_length: embedding_length, 
            memories: initial_memories.unwrap_or(Vec::new()), 
            save_path: save_path
        }
    }

    fn __len__(&self) -> usize {
        self.memories.len()
    }

    fn _add_memory(&mut self, memory_to_add: Memory) {
        // let new_memory = Memory::new(text, numbers);
        let mut resized = memory_to_add.clone();
        resized.embedding.resize_with(self.embedding_length, || 0.0);
        self.memories.push(resized);
    }

    fn _top_n_matches(&mut self, query: &Memory, n: usize, must_include_text: Option<&str>) -> Vec<(f64,Memory)> {
        /*
            Returns the top 'n' Memories by cosine similarity of their embedding vectors to the supplied 'query' memory.
            Optionally include a must-have text string, 'must_include_text', to filter results before searching.
         */
        let mut dist_mem_pairs = self.memories.par_iter()
            .filter_map(|memory| {
                if must_include_text.is_none() ||
                memory.text.to_lowercase().contains(&(must_include_text.unwrap().to_lowercase())) {
                    Some((query._compare(memory), memory.clone()))
                } else {None}
            }).collect::<Vec<(f64, Memory)>>();

            dist_mem_pairs.sort_by_key(|(dist, _mem)| -(1e6*dist) as i32);

            dist_mem_pairs[0..(n.min(dist_mem_pairs.len()))].to_vec()
    }

}


#[pymodule]
fn vector_memory(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<MemoryStore>()?;
    m.add("__all__", vec!["Memory", "MemoryStore"])?;
    Ok(())
}
