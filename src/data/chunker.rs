pub fn chunk_text(text: &str) -> Vec<String> {
    text.split('\n')
        .filter(|p| p.trim().len() > 20)
        .map(|p| p.trim().to_string())
        .collect()
}