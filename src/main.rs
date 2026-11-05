use clap::{Parser, Subcommand};
use simplenn::mnist::Dataset;
use simplenn::network::Network;
use simplenn::train::{TrainConfig, train};
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Trains and evaluates a neural network on MNIST.
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Trains a [784, hidden, 10] network on the MNIST training set and saves it.
    Train {
        /// Directory holding train-images-idx3-ubyte and train-labels-idx1-ubyte.
        #[arg(long)]
        data: PathBuf,
        /// Where to write the trained model.
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 128)]
        hidden: usize,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long, default_value_t = 0.1)]
        learning_rate: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Prints a saved model's accuracy on the MNIST test set.
    Evaluate {
        /// Directory holding t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte.
        #[arg(long)]
        data: PathBuf,
        /// A model written by `train`.
        #[arg(long)]
        model: PathBuf,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("simplenn: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Train {
            data,
            out,
            hidden,
            epochs,
            batch_size,
            learning_rate,
            seed,
        } => {
            let dataset = load_dataset(&data, "train")?;
            let mut network = Network::new(&[784, hidden, 10], seed);
            let config = TrainConfig {
                epochs,
                batch_size,
                learning_rate,
                seed,
            };
            train(
                &mut network,
                &dataset.inputs,
                &dataset.labels,
                &config,
                |epoch, loss| println!("epoch {epoch}: loss {loss:.4}"),
            );
            network.save(&out).map_err(|e| describe(&out, e))
        }
        Command::Evaluate { data, model } => {
            let dataset = load_dataset(&data, "t10k")?;
            let network = Network::load(&model).map_err(|e| describe(&model, e))?;
            let accuracy = network.accuracy(&dataset.inputs, &dataset.labels);
            println!("accuracy: {accuracy:.4}");
            Ok(())
        }
    }
}

/// Reads `<prefix>-images-idx3-ubyte` and `<prefix>-labels-idx1-ubyte` from `dir`.
fn load_dataset(dir: &Path, prefix: &str) -> Result<Dataset, String> {
    let images = dir.join(format!("{prefix}-images-idx3-ubyte"));
    let labels = dir.join(format!("{prefix}-labels-idx1-ubyte"));
    Dataset::load(&images, &labels).map_err(|e| {
        format!(
            "cannot read {} and {}: {e}",
            images.display(),
            labels.display()
        )
    })
}

fn describe(path: &Path, error: io::Error) -> String {
    format!("{}: {error}", path.display())
}
