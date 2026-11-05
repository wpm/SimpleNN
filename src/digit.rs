use image::ImageResult;
use image::imageops::{self, FilterType};
use ndarray::Array2;
use std::path::Path;

/// Reads an image of a handwritten digit as one row of 784 inputs for a network.
///
/// The image is converted to grayscale, resized to 28×28 and scaled to 0.0–1.0. MNIST digits are
/// light on a dark background, so an image whose mean value is above 0.5 is flipped.
pub fn load_digit_image(path: &Path) -> ImageResult<Array2<f32>> {
    let gray = image::open(path)?.to_luma8();
    let image = imageops::resize(&gray, 28, 28, FilterType::Triangle);
    let mut inputs = Array2::from_shape_vec(
        (1, 784),
        image
            .pixels()
            .map(|pixel| pixel[0] as f32 / 255.0)
            .collect(),
    )
    .unwrap();
    if inputs.mean().unwrap() > 0.5 {
        inputs.mapv_inplace(|value| 1.0 - value);
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use tempfile::TempDir;

    /// Writes a `size`×`size` PNG with `background` everywhere except a `foreground` top-left pixel.
    fn write_png(dir: &TempDir, size: u32, background: u8, foreground: u8) -> std::path::PathBuf {
        let mut image = GrayImage::from_pixel(size, size, Luma([background]));
        image.put_pixel(0, 0, Luma([foreground]));
        let path = dir.path().join("digit.png");
        image.save(&path).unwrap();
        path
    }

    #[test]
    fn white_on_black_is_only_scaled() {
        let dir = TempDir::new().unwrap();
        let inputs = load_digit_image(&write_png(&dir, 28, 0, 255)).unwrap();
        assert_eq!(inputs.dim(), (1, 784));
        assert_eq!(inputs[[0, 0]], 1.0);
        assert!(inputs.iter().skip(1).all(|&value| value == 0.0));
    }

    #[test]
    fn black_on_white_is_flipped() {
        let dir = TempDir::new().unwrap();
        let inputs = load_digit_image(&write_png(&dir, 28, 255, 0)).unwrap();
        assert_eq!(inputs[[0, 0]], 1.0);
        assert!(inputs.iter().skip(1).all(|&value| value == 0.0));
    }

    #[test]
    fn larger_image_is_resized_to_784_values() {
        let dir = TempDir::new().unwrap();
        let inputs = load_digit_image(&write_png(&dir, 100, 0, 255)).unwrap();
        assert_eq!(inputs.dim(), (1, 784));
        assert!(inputs.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn non_image_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("digit.png");
        std::fs::write(&path, "not an image").unwrap();
        assert!(load_digit_image(&path).is_err());
    }
}
