use std::fs::File;
use std::io::Read;
use std::fs;
use zip::ZipArchive;

pub fn load_documents(folder: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut corpus = String::new();

    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("docx") {
            println!("Loading {:?}", path);

            let file = File::open(&path)?;
            let mut archive = ZipArchive::new(file)?;

            // Word document XML lives here:
            let mut document_xml = archive.by_name("word/document.xml")?;

            let mut xml_content = String::new();
            document_xml.read_to_string(&mut xml_content)?;

            // Remove XML tags (simple cleanup)
            let text = xml_content
                .replace("<w:t>", "")
                .replace("</w:t>", " ")
                .replace("<w:p>", "\n")
                .replace("</w:p>", "");

            corpus.push_str(&text);
        }
    }

    Ok(corpus)
}