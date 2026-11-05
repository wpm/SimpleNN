use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A fully connected neural network with one weight matrix and one bias vector per layer transition.
pub struct Network {
    weights: Vec<Array2<f32>>,
    biases: Vec<Array1<f32>>,
}

impl Network {
    /// Builds a network with the given layer sizes, such as `[784, 128, 10]`.
    ///
    /// Each weight is drawn uniformly from ±sqrt(6 / inputs to its layer), and biases start at zero.
    /// The same seed gives the same network.
    ///
    /// Panics if `sizes` has fewer than two entries.
    pub fn new(sizes: &[usize], seed: u64) -> Network {
        assert!(sizes.len() >= 2, "a network needs at least two layer sizes");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for pair in sizes.windows(2) {
            let (inputs, outputs) = (pair[0], pair[1]);
            let limit = (6.0 / inputs as f32).sqrt();
            weights.push(Array2::from_shape_simple_fn((inputs, outputs), || {
                rng.random_range(-limit..=limit)
            }));
            biases.push(Array1::zeros(outputs));
        }
        Network { weights, biases }
    }

    /// Returns the raw scores (logits) of the last layer, one row per example in `inputs`.
    ///
    /// ReLU is applied after every hidden layer.
    pub fn forward(&self, inputs: &Array2<f32>) -> Array2<f32> {
        let last = self.weights.len() - 1;
        let mut activations = inputs.clone();
        for (layer, (weights, biases)) in self.weights.iter().zip(&self.biases).enumerate() {
            activations = activations.dot(weights) + biases;
            if layer < last {
                activations.mapv_inplace(|a| a.max(0.0));
            }
        }
        activations
    }

    /// Returns the index of the highest score for each example in `inputs`.
    pub fn predict(&self, inputs: &Array2<f32>) -> Vec<usize> {
        self.forward(inputs)
            .axis_iter(Axis(0))
            .map(|scores| {
                (0..scores.len())
                    .reduce(|best, i| if scores[i] > scores[best] { i } else { best })
                    .unwrap()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn output_has_one_row_per_example_and_one_column_per_output() {
        let network = Network::new(&[4, 5, 3], 0);
        let output = network.forward(&Array2::zeros((2, 4)));
        assert_eq!(output.dim(), (2, 3));
    }

    #[test]
    fn same_seed_gives_same_weights() {
        let a = Network::new(&[4, 5, 3], 7);
        let b = Network::new(&[4, 5, 3], 7);
        assert_eq!(a.weights, b.weights);
    }

    #[test]
    fn different_seeds_give_different_weights() {
        let a = Network::new(&[4, 5, 3], 7);
        let b = Network::new(&[4, 5, 3], 8);
        assert_ne!(a.weights, b.weights);
    }

    #[test]
    fn weights_lie_within_limit_and_biases_start_at_zero() {
        let network = Network::new(&[6, 5, 3], 0);
        let limits = [(6.0f32 / 6.0).sqrt(), (6.0f32 / 5.0).sqrt()];
        for (weights, limit) in network.weights.iter().zip(limits) {
            assert!(weights.iter().all(|w| w.abs() <= limit));
        }
        assert_eq!(network.weights[0].dim(), (6, 5));
        assert_eq!(network.weights[1].dim(), (5, 3));
        assert!(network.biases.iter().flatten().all(|&b| b == 0.0));
    }

    #[test]
    #[should_panic]
    fn fewer_than_two_sizes_panics() {
        Network::new(&[4], 0);
    }

    #[test]
    fn hand_set_network_applies_relu_to_hidden_layer() {
        let network = Network {
            weights: vec![array![[1.0, -1.0], [2.0, 1.0]], array![[1.0], [1.0]]],
            biases: vec![array![0.0, 0.5], array![0.25]],
        };
        // Hidden: [1, 1] -> [3, 0.5]; [1, 0] -> [1, -0.5] -> ReLU [1, 0].
        let output = network.forward(&array![[1.0, 1.0], [1.0, 0.0]]);
        assert_eq!(output, array![[3.75], [1.25]]);
    }

    #[test]
    fn output_layer_is_not_rectified() {
        let network = Network {
            weights: vec![array![[-1.0]]],
            biases: vec![array![0.0]],
        };
        assert_eq!(network.forward(&array![[2.0]]), array![[-2.0]]);
    }

    #[test]
    fn predict_picks_highest_score_in_each_row() {
        let network = Network {
            weights: vec![Array2::eye(3)],
            biases: vec![Array1::zeros(3)],
        };
        let inputs = array![[0.1, 0.9, 0.3], [2.0, -1.0, 0.5], [-3.0, -2.0, -1.0]];
        assert_eq!(network.predict(&inputs), vec![1, 0, 2]);
    }
}
