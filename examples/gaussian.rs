//! Apply gaussian filter to an image will get a blurred image,
//! bigger sigma parameter results bigger blurriness.

use image_convolution::*;

fn main() {
    let file = std::env::args().nth(1).expect("image file name");
    let file = std::path::Path::new(&file);
    let input = Image::load(&file); // RGB image is converted to luma8
    let kernel = kernels::gaussian(0.8);
    dbg!(kernel.size);
    let device = gpu_device::create_gpu_device();
    let output = futures::executor::block_on(convolution::run(&device, &input, &kernel));
    output.save(file.with_extension("result.png"));
}
