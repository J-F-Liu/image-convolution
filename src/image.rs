use crate::Real;

/// Row major image data
pub struct Image {
    pub data: Vec<Real>,
    pub width: u32,
    pub height: u32,
}

impl Image {
    pub fn new(width: u32, height: u32, value: Real) -> Self {
        let len = (width * height) as usize;
        let data = vec![value; len];
        Image {
            width,
            height,
            data,
        }
    }

    pub fn size(&self) -> u32 {
        self.width * self.height
    }

    pub fn load<P: AsRef<std::path::Path>>(filepath: &P) -> Image {
        let image = image::open(filepath).expect("read image file").into_luma8();
        let (width, height) = image.dimensions();
        let data = image.as_raw().iter().map(|pixel| *pixel as Real).collect();
        Image {
            data,
            width,
            height,
        }
    }

    pub fn save<P: AsRef<std::path::Path>>(&self, filepath: P) {
        let image = image::GrayImage::from_raw(
            self.width,
            self.height,
            self.data.iter().map(|pixel| pixel.abs() as u8).collect(),
        )
        .expect("Create output image");
        image.save(filepath).expect("write image file");
    }
}

impl std::ops::Index<(u32, u32)> for Image {
    type Output = Real;

    #[inline]
    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        let idx = (y * self.width + x) as usize;
        &self.data[idx]
    }
}
