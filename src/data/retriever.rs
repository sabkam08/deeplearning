pub fn retrieve_best_chunk(question: &str, chunks: &[String]) -> String {
    let question_words: Vec<&str> = question.to_lowercase().split_whitespace().collect();

    let mut best_score = 0;
    let mut best_chunk = String::new();

    for chunk in chunks {
        let mut score = 0;
        for word in &question_words {
            if chunk.to_lowercase().contains(word) {
                score += 1;
            }
        }

        if score > best_score {
            best_score = score;
            best_chunk = chunk.clone();
        }
    }

    best_chunk
}