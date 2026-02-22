pub fn answer_question<B: Backend>(
    model: &QAModel<B>,
    tokenizer: &Tokenizer,
    corpus: &str,
    question: &str,
) {
    let input_text = format!("{} {}", question, corpus);

    let tokens = encode(tokenizer, &input_text);

    let input_tensor = Tensor::<B, 2>::from_data(tokens);

    let (start_logits, end_logits) = model.forward(input_tensor);

    let start_index = start_logits.argmax().to_scalar() as usize;
    let end_index = end_logits.argmax().to_scalar() as usize;

    let answer_tokens = &tokens[start_index..=end_index];

    println!("Predicted answer tokens: {:?}", answer_tokens);
}