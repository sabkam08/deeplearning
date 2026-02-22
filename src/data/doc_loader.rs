use docx_rs::*;
use std::fs;
use std::path::Path;

pub fn load_all_documents(folder_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut full_corpus = String::new();

    for entry in fs::read_dir(folder_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("docx") {
            println!("Loading: {:?}", path);

            let bytes = fs::read(&path)?;
            let docx = DocxFile::from_reader(bytes.as_slice())?;
            let document = docx.parse()?;

            for paragraph in document.document.paragraphs {
                for run in paragraph.runs {
                    if let Some(text) = run.text {
                        full_corpus.push_str(&text);
                        full_corpus.push(' ');
                    }
                }
            }
        }
    }

    Ok(full_corpus)
}