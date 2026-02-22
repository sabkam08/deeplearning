mod data;
use data::doc_loader::load_all_documents;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args[1].as_str() {
        "train" => {
            println!("Loading documents...");

            let corpus = load_all_documents("data_docs")
                .expect("Failed to load documents");

            println!("Documents loaded.");
            println!("Corpus length: {}", corpus.len());

            // Pass corpus into tokenizer + dataset creation
        }

        "ask" => {
            println!("Enter your question:");
            let mut question = String::new();
            std::io::stdin().read_line(&mut question).unwrap();

            let corpus = load_all_documents("data_docs")
                .expect("Failed to load documents");

            // Combine question + corpus for inference
        }

        _ => println!("Usage: cargo run -- train | ask"),
    }
}