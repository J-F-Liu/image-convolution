use crate::gpu_device::*;
use crate::{Image, Kernel, Real};

pub async fn run(gpu_device: &GpuDevice, image: &Image, kernel: &Kernel) -> Image {
    let GpuDevice { device, queue } = gpu_device;

    let crop = kernel.size - 1;
    let mut output = Image {
        data: Vec::new(),
        width: image.width - crop,
        height: image.height - crop,
    };
    let output_size = (output.size() * std::mem::size_of::<Real>() as u32) as u64;
    let params = [image.width, kernel.size];
    let params_data = bytemuck::cast_slice(&params);

    // create input and output buffers
    let input_buffer = create_data_buffer(&device, "input", bytemuck::cast_slice(&image.data));
    let result_buffer = create_buffer(&device, "result", output_size);
    let kernel_buffer = create_data_buffer(&device, "kernel", bytemuck::cast_slice(&kernel.data));
    let params_buffer = create_uniform_buffer(&device, "params", params_data);
    let output_buffer = create_output_buffer(&device, "output", output_size);

    // create bind group and compute pipeline
    let (bind_group, compute_pipeline) = create_compute_pipeline(
        device,
        &[
            (
                &input_buffer,
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            (
                &result_buffer,
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            (
                &kernel_buffer,
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
            ),
            (
                &params_buffer,
                params_data.len() as u64,
                wgpu::BufferBindingType::Uniform,
            ),
        ],
        include_str!("convolution.wgsl"),
    );

    // encode and run commands
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_pipeline(&compute_pipeline);
        cpass.dispatch(output.width, output.height, 1);
    }
    // copy data from input buffer on GPU to output buffer on CPU
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &output_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    // read output_buffer
    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        output.data = bytemuck::cast_slice::<u8, f32>(&data).to_vec();

        // We have to make sure all mapped views are dropped before we unmap the buffer.
        drop(data);
        output_buffer.unmap();

        output
    } else {
        panic!("failed to run compute on gpu!")
    }
}
