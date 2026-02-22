use tokenizers::Tokenizer;
use burn::tensor::backend::Backend;
use crate::model::qa_model::QAModel;

pub fn answer_question<B: Backend>(
    model: &QAModel<B>,
    tokenizer: &Tokenizer,
    question: &str,
    context: &str,
) {
    let input_text = format!("{} {}", question, context);

    let encoding = tokenizer.encode(input_text, true).unwrap();
    let tokens = encoding.get_ids();

    let input_tensor = Tensor::<B, 2>::from_data(tokens.to_vec());

    let (start_logits, end_logits) = model.forward(input_tensor);

    let start = start_logits.argmax().to_scalar() as usize;
    let end = end_logits.argmax().to_scalar() as usize;

    println!("Predicted answer token range: {} - {}", start, end);
}