pub type Real = f32;

pub mod convolution;
pub mod gpu_device;
mod image;
pub mod kernels;

pub use crate::image::Image;
pub use kernels::Kernel;

pub enum BorderType {
    Crop,
    Mirror,
    Zero,
}
