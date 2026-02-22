use burn::data::dataset::Dataset;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Clone)]
pub struct QADataset<B: Backend> {
    pub inputs: Vec<Tensor<B, 2>>,
    pub start_positions: Vec<Tensor<B, 1>>,
    pub end_positions: Vec<Tensor<B, 1>>,
}

impl<B: Backend> Dataset<(Tensor<B,2>, Tensor<B,1>, Tensor<B,1>)> for QADataset<B> {
    fn get(&self, index: usize) -> Option<(Tensor<B,2>, Tensor<B,1>, Tensor<B,1>)> {
        Some((
            self.inputs[index].clone(),
            self.start_positions[index].clone(),
            self.end_positions[index].clone(),
        ))
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }
}