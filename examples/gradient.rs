use image_convolution::{kernels::*, Image, Kernel, Pipeline};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "gradient")]
struct Args {
    /// Image file or folder
    image_path: PathBuf,
    /// Sigma of gaussian kernel
    #[structopt(short, long, default_value = "0.8")]
    sigma: f32,
    /// Gradient kernel type
    #[structopt(short, long, default_value = "sobel")]
    gradient: String,
}

fn main() {
    let args = Args::from_args();
    let gaussian_kernel = gaussian(args.sigma);
    let gradient_kernel = match args.gradient.as_str() {
        "roberts" => roberts_operator(),
        "desolneux" => desolneux_operator(),
        "sobel" => sobel_operator(),
        "freichen" => freichen_operator(),
        _ => unimplemented!(),
    };
    if args.image_path.is_file() {
        let (gx, gy) = compute_gradient(&args.image_path, gaussian_kernel, gradient_kernel);
        gx.save(args.image_path.with_extension("gx.png"));
        gy.save(args.image_path.with_extension("gy.png"));
    }
}

fn compute_gradient(
    image_file: &PathBuf,
    gaussian_kernel: Kernel,
    (kx, ky): (Kernel, Kernel),
) -> (Image, Image) {
    let image = Image::load(image_file);
    let mut pipeline = Pipeline::new();

    // chain operations
    let input = pipeline
        .device
        .create_data_buffer("input", bytemuck::cast_slice(&image.data));
    let (smoothed_image, smoothed_image_size) =
        pipeline.chain(&input, &gaussian_kernel, (image.width, image.height));
    let (gx, gx_size) = pipeline.chain(&smoothed_image, &kx, smoothed_image_size);
    let (gy, gy_size) = pipeline.chain(&smoothed_image, &ky, smoothed_image_size);
    let mut gradients: Vec<Vec<f32>> =
        futures::executor::block_on(pipeline.run(&[(&gx, gx_size, 4), (&gy, gy_size, 4)]));
    (
        Image {
            data: gradients.swap_remove(0),
            width: gx_size.0,
            height: gx_size.1,
        },
        Image {
            data: gradients.swap_remove(0),
            width: gy_size.0,
            height: gy_size.1,
        },
    )
}
