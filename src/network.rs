use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// A fully connected neural network with one weight matrix and one bias vector per layer transition.
#[derive(Serialize, Deserialize)]
pub struct Network {
    pub(crate) weights: Vec<Array2<f32>>,
    pub(crate) biases: Vec<Array1<f32>>,
}

/// How the loss changes with each parameter: one matrix and one vector per layer transition,
/// shaped like the network's weights and biases.
pub struct Gradients {
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array1<f32>>,
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
        self.layer_outputs(inputs).pop().unwrap()
    }

    /// Returns the output of every layer, starting with `inputs` and ending with the logits.
    fn layer_outputs(&self, inputs: &Array2<f32>) -> Vec<Array2<f32>> {
        let last = self.weights.len() - 1;
        let mut outputs = vec![inputs.clone()];
        for (layer, (weights, biases)) in self.weights.iter().zip(&self.biases).enumerate() {
            let mut activations = outputs[layer].dot(weights) + biases;
            if layer < last {
                activations.mapv_inplace(|a| a.max(0.0));
            }
            outputs.push(activations);
        }
        outputs
    }

    /// Returns the softmax cross-entropy loss averaged over the examples in `inputs`, whose correct
    /// classes are `labels`, and the gradient of that loss with respect to every parameter.
    pub fn loss_and_gradients(&self, inputs: &Array2<f32>, labels: &[usize]) -> (f32, Gradients) {
        let mut outputs = self.layer_outputs(inputs);
        let log_probabilities = log_softmax(&outputs.pop().unwrap());
        let examples = labels.len() as f32;
        let loss = -labels
            .iter()
            .enumerate()
            .map(|(row, &label)| log_probabilities[[row, label]])
            .sum::<f32>()
            / examples;

        // The loss's gradient with respect to the logits is (softmax − one-hot) / examples.
        let mut delta = log_probabilities.mapv(f32::exp);
        for (row, &label) in labels.iter().enumerate() {
            delta[[row, label]] -= 1.0;
        }
        delta /= examples;

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for layer in (0..self.weights.len()).rev() {
            let layer_inputs = &outputs[layer];
            weights.push(layer_inputs.t().dot(&delta));
            biases.push(delta.sum_axis(Axis(0)));
            if layer > 0 {
                // Gradient passes back only through hidden units that ReLU left positive.
                delta = delta.dot(&self.weights[layer].t());
                delta.zip_mut_with(layer_inputs, |d, &a| {
                    if a <= 0.0 {
                        *d = 0.0
                    }
                });
            }
        }
        weights.reverse();
        biases.reverse();
        (loss, Gradients { weights, biases })
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

    /// Returns the fraction of examples in `inputs` whose prediction matches its label.
    pub fn accuracy(&self, inputs: &Array2<f32>, labels: &[usize]) -> f32 {
        let correct = self
            .predict(inputs)
            .iter()
            .zip(labels)
            .filter(|(predicted, label)| predicted == label)
            .count();
        correct as f32 / labels.len() as f32
    }

    /// Writes the network to `path` as JSON.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        std::fs::write(path, serde_json::to_string(self)?)
    }

    /// Reads a network written by [`Network::save`].
    ///
    /// Malformed JSON gives an [`io::ErrorKind::InvalidData`] error.
    pub fn load(path: &Path) -> io::Result<Network> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// Returns the log of the softmax of each row, subtracting the row's largest score first so that
/// large scores do not overflow.
fn log_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let largest = row.fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        row -= largest;
        let log_sum = row.mapv(f32::exp).sum().ln();
        row -= log_sum;
    }
    result
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

    #[test]
    fn accuracy_is_the_fraction_of_correct_predictions() {
        let network = Network {
            weights: vec![Array2::eye(3)],
            biases: vec![Array1::zeros(3)],
        };
        // Predictions are 1, 0, 2 and 2; three of the four labels match.
        let inputs = array![
            [0.1, 0.9, 0.3],
            [2.0, -1.0, 0.5],
            [-3.0, -2.0, -1.0],
            [0.0, 0.0, 1.0]
        ];
        assert_eq!(network.accuracy(&inputs, &[1, 0, 2, 0]), 0.75);
    }

    #[test]
    fn loss_for_hand_set_logits_matches_hand_computed_value() {
        let network = Network {
            weights: vec![Array2::eye(2)],
            biases: vec![Array1::zeros(2)],
        };
        // Row 1: logits [0, 0], label 0 -> -ln(1/2) = ln 2.
        // Row 2: logits [ln 3, 0], label 1 -> -ln(1/4) = ln 4.
        let inputs = array![[0.0, 0.0], [3.0f32.ln(), 0.0]];
        let (loss, _) = network.loss_and_gradients(&inputs, &[0, 1]);
        let expected = (2.0f32.ln() + 4.0f32.ln()) / 2.0;
        assert!((loss - expected).abs() < 1e-6, "{loss} != {expected}");
    }

    #[test]
    fn loss_stays_finite_for_very_large_scores() {
        let network = Network {
            weights: vec![Array2::eye(3)],
            biases: vec![Array1::zeros(3)],
        };
        let inputs = array![[1000.0, -1000.0, 500.0]];
        let (loss, gradients) = network.loss_and_gradients(&inputs, &[1]);
        assert!((loss - 2000.0).abs() < 1e-2, "{loss}");
        assert!(gradients.weights.iter().flatten().all(|g| g.is_finite()));
        assert!(gradients.biases.iter().flatten().all(|g| g.is_finite()));
    }

    #[test]
    fn gradients_have_the_shapes_of_the_parameters() {
        let network = Network::new(&[3, 4, 5, 2], 0);
        let (_, gradients) = network.loss_and_gradients(&Array2::ones((2, 3)), &[0, 1]);
        for (g, w) in gradients.weights.iter().zip(&network.weights) {
            assert_eq!(g.dim(), w.dim());
        }
        for (g, b) in gradients.biases.iter().zip(&network.biases) {
            assert_eq!(g.dim(), b.dim());
        }
    }

    /// Estimates the loss's slope with respect to one parameter by nudging it both ways.
    fn central_difference(
        network: &mut Network,
        inputs: &Array2<f32>,
        labels: &[usize],
        epsilon: f32,
        parameter: impl Fn(&mut Network) -> &mut f32,
    ) -> f32 {
        let original = *parameter(network);
        *parameter(network) = original + epsilon;
        let (plus, _) = network.loss_and_gradients(inputs, labels);
        *parameter(network) = original - epsilon;
        let (minus, _) = network.loss_and_gradients(inputs, labels);
        *parameter(network) = original;
        (plus - minus) / (2.0 * epsilon)
    }

    /// Checks every gradient against the central difference (loss(p + ε) − loss(p − ε)) / 2ε.
    fn assert_gradients_match_numerical_estimates(sizes: &[usize]) {
        const EPSILON: f32 = 1e-2;
        const TOLERANCE: f32 = 1e-2;
        let mut network = Network::new(sizes, 42);
        // Nonzero biases so their gradients are exercised away from the initial state.
        for biases in &mut network.biases {
            biases
                .iter_mut()
                .enumerate()
                .for_each(|(i, b)| *b = 0.1 * i as f32);
        }
        let mut rng = StdRng::seed_from_u64(1);
        let inputs = Array2::from_shape_simple_fn((5, sizes[0]), || rng.random_range(-1.0..1.0));
        let classes = *sizes.last().unwrap();
        let labels: Vec<usize> = (0..5).map(|i| i % classes).collect();
        let (_, gradients) = network.loss_and_gradients(&inputs, &labels);

        for layer in 0..network.weights.len() {
            let (rows, columns) = network.weights[layer].dim();
            for row in 0..rows {
                for column in 0..columns {
                    let numerical =
                        central_difference(&mut network, &inputs, &labels, EPSILON, |n| {
                            &mut n.weights[layer][[row, column]]
                        });
                    let analytic = gradients.weights[layer][[row, column]];
                    assert!(
                        (analytic - numerical).abs() < TOLERANCE,
                        "weight {layer} [{row}, {column}]: {analytic} != {numerical}"
                    );
                }
            }
            for i in 0..network.biases[layer].len() {
                let numerical = central_difference(&mut network, &inputs, &labels, EPSILON, |n| {
                    &mut n.biases[layer][i]
                });
                let analytic = gradients.biases[layer][i];
                assert!(
                    (analytic - numerical).abs() < TOLERANCE,
                    "bias {layer} [{i}]: {analytic} != {numerical}"
                );
            }
        }
    }

    #[test]
    fn gradients_match_numerical_estimates_with_one_hidden_layer() {
        assert_gradients_match_numerical_estimates(&[3, 4, 3]);
    }

    #[test]
    fn gradients_match_numerical_estimates_with_two_hidden_layers() {
        assert_gradients_match_numerical_estimates(&[3, 5, 4, 3]);
    }

    /// A path in the system temporary directory, unique to this process and test.
    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("simplenn-{}-{name}.json", std::process::id()))
    }

    #[test]
    fn saved_and_loaded_network_gives_identical_output() {
        let network = Network::new(&[4, 5, 3], 3);
        let path = temp_path("round-trip");
        network.save(&path).unwrap();
        let loaded = Network::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let inputs = array![[0.5, -1.0, 2.0, 0.25], [1.0, 1.0, 1.0, 1.0]];
        assert_eq!(loaded.forward(&inputs), network.forward(&inputs));
    }

    #[test]
    fn loading_malformed_json_returns_invalid_data() {
        let path = temp_path("malformed");
        std::fs::write(&path, "{ not json").unwrap();
        let error = Network::load(&path).err().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn loading_missing_file_returns_error() {
        assert!(Network::load(&temp_path("missing")).is_err());
    }
}
