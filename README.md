# SimpleNN

A small neural network library and command line written in Rust, trained on the
[MNIST](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits. The `simplenn` command trains a
network, measures its accuracy, and predicts the digit in an image.

## Design

The network is fully connected, with layers of 784 inputs (one per pixel of a 28×28 image), a hidden layer
of 128 units, and 10 outputs (one per digit). ReLU follows the hidden layer, and the outputs are scored with
softmax cross-entropy loss. Weights start uniformly random within ±sqrt(6 / inputs) and biases at zero.
Training is mini-batch stochastic gradient descent with gradients computed by backpropagation. A trained
network is saved as JSON.

## Get MNIST

```sh
mkdir -p data
cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -O https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
    gunzip $f.gz
done
cd ..
```

## Train, evaluate and predict

Train a network on the 60,000 training images and save it:

```sh
cargo run --release -- train --data data --out model.json
```

The defaults are `--hidden 128 --epochs 10 --batch-size 32 --learning-rate 0.1 --seed 42`. Training prints
the mean loss after each epoch.

Print the model's accuracy on the 10,000 test images:

```sh
cargo run --release -- evaluate --data data --model model.json
```

Print the digit the model sees in an image:

```sh
cargo run --release -- predict --model model.json digit.png
```

The image is converted to grayscale, resized to 28×28, and inverted if it is dark on light. `predict`
doesn't center the digit in the image the way MNIST digits are centered, so accuracy on your own images
will be lower than the MNIST score.

## Accuracy

With the default settings, the trained network reaches **97.83%** accuracy on the MNIST test set. Training
took 4.7 seconds on an Apple M3 Pro.

## Tests

```sh
cargo test
```

The tests don't need the MNIST data.
