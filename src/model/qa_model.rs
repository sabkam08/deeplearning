use burn::nn::{Embedding, Linear};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;

#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    pub token_embed: Embedding<B>,
    pub layers: Vec<Linear<B>>,
    pub start_head: Linear<B>,
    pub end_head: Linear<B>,
}

impl<B: Backend> QAModel<B> {

    // 👇 THIS is where Linear::new(...) goes
    pub fn new(vocab_size: usize) -> Self {

        let token_embed = Embedding::new(vocab_size, 256);

        let layers = vec![
            Linear::new(256, 256),
            Linear::new(256, 256),
            Linear::new(256, 256),
            Linear::new(256, 256),
            Linear::new(256, 256),
            Linear::new(256, 256),
        ];

        let start_head = Linear::new(256, 1);
        let end_head = Linear::new(256, 1);

        Self {
            token_embed,
            layers,
            start_head,
            end_head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {

        let mut x = self.token_embed.forward(input);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let start_logits = self.start_head.forward(x.clone());
        let end_logits = self.end_head.forward(x);

        (start_logits, end_logits)
    }
}