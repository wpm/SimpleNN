use ndarray::Array2;
use std::fs;
use std::io;
use std::path::Path;

/// Images and their labels, read from MNIST's uncompressed IDX files.
pub struct Dataset {
    /// One row per image, 784 columns, pixels scaled to 0.0–1.0.
    pub inputs: Array2<f32>,
    pub labels: Vec<usize>,
}

impl Dataset {
    /// Reads an IDX image file and its matching IDX label file.
    ///
    /// Returns an `InvalidData` error for a wrong magic number, disagreeing counts, images other than
    /// 28×28, or a truncated file.
    pub fn load(images_path: &Path, labels_path: &Path) -> io::Result<Dataset> {
        let images = fs::read(images_path)?;
        let labels = fs::read(labels_path)?;
        let [count, rows, columns] = header(&images, 2051)?;
        let [label_count] = header(&labels, 2049)?;
        if count != label_count {
            return Err(invalid_data("image and label counts disagree"));
        }
        if (rows, columns) != (28, 28) {
            return Err(invalid_data("images are not 28×28"));
        }
        let pixels = body(&images, 16, count * 784)?;
        let labels = body(&labels, 8, count)?;
        Ok(Dataset {
            inputs: Array2::from_shape_vec(
                (count, 784),
                pixels.iter().map(|&pixel| pixel as f32 / 255.0).collect(),
            )
            .unwrap(),
            labels: labels.iter().map(|&label| label as usize).collect(),
        })
    }
}

/// Checks the magic number at the start of an IDX file and returns the `N` big-endian integers after it.
fn header<const N: usize>(bytes: &[u8], magic: u32) -> io::Result<[usize; N]> {
    let words: Vec<u32> = body(bytes, 0, 4 * (N + 1))?
        .chunks(4)
        .map(|word| u32::from_be_bytes(word.try_into().unwrap()))
        .collect();
    if words[0] != magic {
        return Err(invalid_data("wrong magic number"));
    }
    Ok(std::array::from_fn(|i| words[i + 1] as usize))
}

/// Returns `length` bytes starting at `start`.
fn body(bytes: &[u8], start: usize, length: usize) -> io::Result<&[u8]> {
    bytes
        .get(start..start + length)
        .ok_or_else(|| invalid_data("file is truncated"))
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn image_file(count: u32, rows: u32, columns: u32, pixels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for n in [2051, count, rows, columns] {
            bytes.extend(n.to_be_bytes());
        }
        bytes.extend(pixels);
        bytes
    }

    fn label_file(count: u32, labels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for n in [2049, count] {
            bytes.extend(n.to_be_bytes());
        }
        bytes.extend(labels);
        bytes
    }

    fn load(images: &[u8], labels: &[u8]) -> io::Result<Dataset> {
        let dir = TempDir::new().unwrap();
        let images_path = dir.path().join("images");
        let labels_path = dir.path().join("labels");
        fs::write(&images_path, images).unwrap();
        fs::write(&labels_path, labels).unwrap();
        Dataset::load(&images_path, &labels_path)
    }

    fn assert_invalid_data(result: io::Result<Dataset>) {
        assert_eq!(result.err().unwrap().kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn loads_images_with_scaled_pixels_and_labels() {
        let mut pixels = vec![0u8; 3 * 784];
        pixels[0] = 255;
        pixels[784 + 1] = 51;
        pixels[2 * 784 + 783] = 102;
        let dataset = load(&image_file(3, 28, 28, &pixels), &label_file(3, &[7, 0, 9])).unwrap();
        assert_eq!(dataset.inputs.dim(), (3, 784));
        assert_eq!(dataset.inputs[[0, 0]], 1.0);
        assert_eq!(dataset.inputs[[1, 1]], 0.2);
        assert_eq!(dataset.inputs[[2, 783]], 0.4);
        assert_eq!(dataset.inputs.iter().filter(|&&p| p != 0.0).count(), 3);
        assert_eq!(dataset.labels, vec![7, 0, 9]);
    }

    #[test]
    fn wrong_image_magic_number_is_invalid_data() {
        let mut images = image_file(1, 28, 28, &[0; 784]);
        images[3] = 0;
        assert_invalid_data(load(&images, &label_file(1, &[0])));
    }

    #[test]
    fn wrong_label_magic_number_is_invalid_data() {
        let mut labels = label_file(1, &[0]);
        labels[3] = 0;
        assert_invalid_data(load(&image_file(1, 28, 28, &[0; 784]), &labels));
    }

    #[test]
    fn disagreeing_counts_are_invalid_data() {
        let images = image_file(2, 28, 28, &[0; 2 * 784]);
        assert_invalid_data(load(&images, &label_file(1, &[0])));
    }

    #[test]
    fn dimensions_other_than_28_by_28_are_invalid_data() {
        let images = image_file(1, 27, 29, &[0; 27 * 29]);
        assert_invalid_data(load(&images, &label_file(1, &[0])));
    }

    #[test]
    fn truncated_image_file_is_invalid_data() {
        let images = image_file(2, 28, 28, &[0; 784]);
        assert_invalid_data(load(&images, &label_file(2, &[0, 0])));
    }

    #[test]
    fn truncated_label_file_is_invalid_data() {
        let images = image_file(2, 28, 28, &[0; 2 * 784]);
        assert_invalid_data(load(&images, &label_file(2, &[0])));
    }

    #[test]
    fn truncated_header_is_invalid_data() {
        let images = image_file(1, 28, 28, &[0; 784]);
        assert_invalid_data(load(&images[..10], &label_file(1, &[0])));
    }
}
