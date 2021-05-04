pub type Real = f32;

pub mod convolution;
pub mod gpu_device;
mod image;
pub mod kernels;
mod pipeline;

pub use crate::image::Image;
pub use kernels::Kernel;
pub use pipeline::Pipeline;

pub enum BorderType {
    Crop,
    Mirror,
    Zero,
}
