mod data;
mod model;
mod inference;

use data::{loader::load_documents, chunker::chunk_text, retriever::retrieve_best_chunk};

fn main() {
    println!("Loading school calendar documents...");

    let corpus = load_documents("data_docs")
        .expect("Failed to load documents");

    let chunks = chunk_text(&corpus);

    println!("Enter your question:");

    let mut question = String::new();
    std::io::stdin().read_line(&mut question).unwrap();

    let best_chunk = retrieve_best_chunk(&question, &chunks);

    println!("Most relevant section:\n{}\n", best_chunk);

    // Load model + tokenizer here
    // Call answer_question(...)
}