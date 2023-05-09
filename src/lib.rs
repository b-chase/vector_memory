use pyo3::prelude::*;
use rayon::prelude::*;
use std::io::Write;
use std::fs::File;
use serde::{Serialize, Deserialize};
use bincode::{serialize, deserialize_from};
use std::path::Path;


fn dot_product(a:&Vec<f64>, b:&Vec<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, b)| a*b)
        .sum::<f64>()
}

const COMPRESSION_MAP_SIZE: usize = 65;

static COMPRESSION_MAP: [char; COMPRESSION_MAP_SIZE] = [
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
    // let n = input_embedding.len();

    let compressed_to_string: String = input_embedding.par_chunks(COMPRESSION_MAP_SIZE-1)
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
#[derive(Clone, Serialize, Deserialize, Debug)]
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

    fn _save_memories(&self) -> std::io::Result<()> {
        // saves memories to the given directory
        assert!(self.save_path.is_some(), "ERROR! This memory store has no specified save location!");

        let mem_dir_str = self.save_path.as_ref().unwrap();
        if Path::exists(Path::new(&mem_dir_str)) {
            for mem in self.memories.iter() {
                let file_name = compress_embedding(&mem.embedding);
                let file_path_str = format!("{}/{}.vmem", &mem_dir_str, &file_name);
                let file_path = Path::new(&file_path_str);
                let mut save_file = File::create(file_path)?;
                
                if let Ok(encoded_mem) = serialize(&mem) {
                    match save_file.write_all(encoded_mem.as_slice()) {
                        Ok(()) => (), 
                        Err(save_result) => eprintln!("Error saving memory to file {}. Error msg:\n{:?}", file_path_str, save_result)
                    };
                }
            }
        }
        Ok(())
    }

    fn _load_memories(&mut self, from_dir: Option<String>) -> std::io::Result<()> {

        let mem_dir_str = match from_dir {
            Some(dir_string) => dir_string,
            None => if let Some(self_dir) = &self.save_path {String::from(self_dir)} else {String::from("default_memory_folder")}
        };

        let mem_files = Path::new(&mem_dir_str).read_dir()?;

        for memf in mem_files {
            let mem_file = File::open(memf.unwrap().path())?;
            let deserialized = deserialize_from(mem_file);
            match deserialized {
                Ok(deserialized_memory) => self._add_memory(deserialized_memory),
                Err(e) => eprintln!("Error loading memory from file '{}'. Error message:\n{:?}", mem_dir_str, e)
            }
        }
        Ok(())
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
