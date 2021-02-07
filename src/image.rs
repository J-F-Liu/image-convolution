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
}

impl std::ops::Index<(u32, u32)> for Image {
    type Output = Real;

    #[inline]
    fn index(&self, (x, y): (u32, u32)) -> &Self::Output {
        let idx = (y * self.width + x) as usize;
        &self.data[idx]
    }
}
