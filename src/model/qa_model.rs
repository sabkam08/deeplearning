use burn::nn::{Embedding, Linear};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;

use super::transformer::TransformerBlock;

#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    pub token_embed: Embedding<B>,
    pub position_embed: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub linear_start: Linear<B>,
    pub linear_end: Linear<B>,
}

impl<B: Backend> QAModel<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let seq_len = input_ids.dims()[1];

        let token_embeddings = self.token_embed.forward(input_ids.clone());

        let positions = Tensor::<B, 2>::arange(0..seq_len as i64)
            .unsqueeze_dim(0);

        let position_embeddings = self.position_embed.forward(positions);

        let mut x = token_embeddings + position_embeddings;

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let start_logits = self.linear_start.forward(x.clone());
        let end_logits = self.linear_end.forward(x);

        (start_logits, end_logits)
    }
}