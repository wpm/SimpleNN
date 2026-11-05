use crate::network::Network;
use ndarray::{Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Settings for mini-batch stochastic gradient descent.
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

/// Trains `network` on `inputs`, whose correct classes are `labels`, by mini-batch stochastic
/// gradient descent.
///
/// Each epoch shuffles the examples, walks them in batches of `batch_size` (the last may be
/// smaller), and after each batch subtracts `learning_rate` times its gradients from the
/// parameters. The same seed gives the same shuffles.
///
/// After each epoch, `on_epoch` is called with the epoch number, counted from 1, and the epoch's
/// mean batch loss.
pub fn train(
    network: &mut Network,
    inputs: &Array2<f32>,
    labels: &[usize],
    config: &TrainConfig,
    mut on_epoch: impl FnMut(usize, f32),
) {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order: Vec<usize> = (0..labels.len()).collect();
    for epoch in 1..=config.epochs {
        order.shuffle(&mut rng);
        let mut total_loss = 0.0;
        let mut batches = 0;
        for batch in order.chunks(config.batch_size) {
            let batch_inputs = inputs.select(Axis(0), batch);
            let batch_labels: Vec<usize> = batch.iter().map(|&i| labels[i]).collect();
            let (loss, gradients) = network.loss_and_gradients(&batch_inputs, &batch_labels);
            for (weights, gradient) in network.weights.iter_mut().zip(&gradients.weights) {
                weights.scaled_add(-config.learning_rate, gradient);
            }
            for (biases, gradient) in network.biases.iter_mut().zip(&gradients.biases) {
                biases.scaled_add(-config.learning_rate, gradient);
            }
            total_loss += loss;
            batches += 1;
        }
        on_epoch(epoch, total_loss / batches as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn xor() -> (Array2<f32>, Vec<usize>) {
        (
            array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            vec![0, 1, 1, 0],
        )
    }

    /// Two clusters of points, one on each side of the line x = y.
    fn separable() -> (Array2<f32>, Vec<usize>) {
        (
            array![
                [0.0, 1.0],
                [0.2, 0.9],
                [0.1, 0.6],
                [0.3, 0.8],
                [1.0, 0.0],
                [0.9, 0.2],
                [0.6, 0.1],
                [0.8, 0.3]
            ],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        )
    }

    fn config(epochs: usize) -> TrainConfig {
        TrainConfig {
            epochs,
            batch_size: 2,
            learning_rate: 0.5,
            seed: 1,
        }
    }

    #[test]
    fn network_learns_xor() {
        let (inputs, labels) = xor();
        let mut network = Network::new(&[2, 8, 2], 0);
        train(&mut network, &inputs, &labels, &config(1000), |_, _| {});
        assert_eq!(network.predict(&inputs), labels);
    }

    #[test]
    fn training_lowers_the_loss_on_a_separable_problem() {
        let (inputs, labels) = separable();
        let mut network = Network::new(&[2, 4, 2], 0);
        let (before, _) = network.loss_and_gradients(&inputs, &labels);
        train(&mut network, &inputs, &labels, &config(20), |_, _| {});
        let (after, _) = network.loss_and_gradients(&inputs, &labels);
        assert!(after < before, "{after} >= {before}");
    }

    #[test]
    fn same_seed_gives_identical_parameters() {
        let (inputs, labels) = separable();
        let mut a = Network::new(&[2, 4, 2], 0);
        let mut b = Network::new(&[2, 4, 2], 0);
        train(&mut a, &inputs, &labels, &config(5), |_, _| {});
        train(&mut b, &inputs, &labels, &config(5), |_, _| {});
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.biases, b.biases);
    }

    #[test]
    fn different_seeds_give_different_parameters() {
        let (inputs, labels) = separable();
        let mut a = Network::new(&[2, 4, 2], 0);
        let mut b = Network::new(&[2, 4, 2], 0);
        train(&mut a, &inputs, &labels, &config(5), |_, _| {});
        let other_seed = TrainConfig {
            seed: 2,
            ..config(5)
        };
        train(&mut b, &inputs, &labels, &other_seed, |_, _| {});
        assert_ne!(a.weights, b.weights);
    }

    #[test]
    fn on_epoch_is_called_once_per_epoch_with_increasing_numbers() {
        let (inputs, labels) = separable();
        let mut network = Network::new(&[2, 4, 2], 0);
        let mut epochs = Vec::new();
        train(&mut network, &inputs, &labels, &config(3), |epoch, loss| {
            assert!(loss.is_finite());
            epochs.push(epoch);
        });
        assert_eq!(epochs, vec![1, 2, 3]);
    }

    #[test]
    fn on_epoch_reports_the_mean_batch_loss() {
        // With a learning rate of zero the network never changes. Batches of 3 over 8 examples
        // give sizes 3, 3 and 2, and the mean of their losses lies within the per-example losses.
        let (inputs, labels) = separable();
        let mut network = Network::new(&[2, 4, 2], 0);
        let frozen = TrainConfig {
            epochs: 1,
            batch_size: 3,
            learning_rate: 0.0,
            seed: 1,
        };
        let mut reported = None;
        train(&mut network, &inputs, &labels, &frozen, |_, loss| {
            reported = Some(loss)
        });
        let reported = reported.unwrap();
        let losses: Vec<f32> = (0..8)
            .map(|i| {
                let row = inputs.slice(ndarray::s![i..i + 1, ..]).to_owned();
                network.loss_and_gradients(&row, &labels[i..i + 1]).0
            })
            .collect();
        let lowest = losses.iter().cloned().fold(f32::INFINITY, f32::min);
        let highest = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(lowest <= reported && reported <= highest, "{reported}");
    }
}
