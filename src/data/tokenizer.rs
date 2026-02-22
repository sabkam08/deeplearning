use tokenizers::Tokenizer;

pub fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_pretrained("bert-base-uncased", None).unwrap()
}

pub fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, true).unwrap().get_ids().to_vec()
}