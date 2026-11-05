use image::{GrayImage, Luma};
use simplenn::network::Network;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn simplenn(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_simplenn"))
        .args(args)
        .output()
        .unwrap()
}

/// Writes an IDX image file and label file for `labels.len()` 28×28 images. Image `i` has its
/// pixels lit only in a band of rows chosen by its label, so the classes are easy to tell apart.
fn write_idx(dir: &Path, prefix: &str, labels: &[u8]) {
    let count = labels.len() as u32;
    let mut images = Vec::new();
    for n in [2051, count, 28, 28] {
        images.extend(n.to_be_bytes());
    }
    for &label in labels {
        let mut pixels = [0u8; 784];
        let start = label as usize * 28 * 2;
        pixels[start..start + 56].fill(255);
        images.extend(pixels);
    }
    fs::write(dir.join(format!("{prefix}-images-idx3-ubyte")), images).unwrap();

    let mut label_bytes = Vec::new();
    for n in [2049, count] {
        label_bytes.extend(n.to_be_bytes());
    }
    label_bytes.extend(labels);
    fs::write(dir.join(format!("{prefix}-labels-idx1-ubyte")), label_bytes).unwrap();
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn train_writes_a_model_that_evaluate_scores() {
    let dir = TempDir::new().unwrap();
    let labels: Vec<u8> = (0..10).collect();
    write_idx(dir.path(), "train", &labels);
    write_idx(dir.path(), "t10k", &labels);
    let data = dir.path().to_str().unwrap();
    let model = dir.path().join("model.json");
    let model = model.to_str().unwrap();

    let output = simplenn(&[
        "train", "--data", data, "--out", model, "--hidden", "8", "--epochs", "3",
    ]);
    assert!(output.status.success(), "{output:?}");
    let lines: Vec<String> = stdout(&output).lines().map(String::from).collect();
    assert_eq!(lines.len(), 3, "{lines:?}");
    for (i, line) in lines.iter().enumerate() {
        assert!(line.starts_with(&format!("epoch {}", i + 1)), "{line}");
        assert!(line.contains("loss"), "{line}");
    }
    assert!(Path::new(model).exists());

    let output = simplenn(&["evaluate", "--data", data, "--model", model]);
    assert!(output.status.success(), "{output:?}");
    let text = stdout(&output);
    let accuracy = text.trim().strip_prefix("accuracy: ").unwrap();
    assert_eq!(accuracy.len(), 6, "{text}");
    let accuracy: f32 = accuracy.parse().unwrap();
    assert!((0.0..=1.0).contains(&accuracy));
}

#[test]
fn missing_data_directory_exits_non_zero_with_a_message() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("missing");
    let model = dir.path().join("model.json");
    let output = simplenn(&[
        "train",
        "--data",
        missing.to_str().unwrap(),
        "--out",
        model.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    assert!(!output.stderr.is_empty());
}

#[test]
fn predict_prints_a_single_digit() {
    let dir = TempDir::new().unwrap();
    let model = dir.path().join("model.json");
    Network::new(&[784, 8, 10], 42).save(&model).unwrap();
    let image = dir.path().join("digit.png");
    GrayImage::from_pixel(28, 28, Luma([0]))
        .save(&image)
        .unwrap();

    let output = simplenn(&[
        "predict",
        "--model",
        model.to_str().unwrap(),
        image.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{output:?}");
    let text = stdout(&output);
    let digit: usize = text.strip_suffix('\n').unwrap().parse().unwrap();
    assert!(digit < 10, "{text}");
}

#[test]
fn predict_on_a_missing_or_non_image_file_exits_non_zero_with_a_message() {
    let dir = TempDir::new().unwrap();
    let model = dir.path().join("model.json");
    Network::new(&[784, 8, 10], 42).save(&model).unwrap();
    let text = dir.path().join("digit.png");
    fs::write(&text, "not an image").unwrap();
    let missing = dir.path().join("missing.png");

    for image in [text, missing] {
        let output = simplenn(&[
            "predict",
            "--model",
            model.to_str().unwrap(),
            image.to_str().unwrap(),
        ]);
        assert!(!output.status.success());
        assert!(!output.stderr.is_empty());
    }
}
