use burn::optim::Adam;
use burn::nn::loss::CrossEntropyLoss;

pub fn train_loop<B: Backend>(
    model: &QAModel<B>,
    dataset: QADataset<B>,
) {
    let optimizer = Adam::new(1e-4);
    let loss_fn = CrossEntropyLoss::new();

    for epoch in 0..5 {
        let mut total_loss = 0.0;

        for (input, start, end) in dataset.iter() {
            let (start_logits, end_logits) = model.forward(input);

            let loss_start = loss_fn.forward(start_logits, start);
            let loss_end = loss_fn.forward(end_logits, end);

            let loss = loss_start + loss_end;

            total_loss += loss.to_scalar();
        }

        println!("Epoch {} Loss: {}", epoch, total_loss);
    }
}