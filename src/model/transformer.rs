use burn::nn::{
    Linear, LayerNorm, Dropout, Embedding,
};
use burn::tensor::backend::Backend;
use burn::module::Module;

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub dropout: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: burn::tensor::Tensor<B, 3>) -> burn::tensor::Tensor<B, 3> {
        let residual = x.clone();
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.linear2.forward(x);
        let x = self.norm1.forward(x + residual);
        self.norm2.forward(x)
    }
}