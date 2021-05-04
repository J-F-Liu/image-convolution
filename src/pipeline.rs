use crate::gpu_device::*;
use crate::{Image, Kernel, Real};

pub struct Pipeline {
    pub device: GpuDevice,
    encoder: wgpu::CommandEncoder,
}

impl Pipeline {
    pub fn new() -> Self {
        let device = create_gpu_device();
        let encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Pipeline { device, encoder }
    }

    pub fn chain(
        &mut self,
        input_buffer: &wgpu::Buffer,
        kernel: &Kernel,
        image_size: (u32, u32),
    ) -> (wgpu::Buffer, (u32, u32)) {
        let (width, height) = image_size;
        let crop = kernel.size - 1;
        let output = Image {
            data: Vec::new(),
            width: width - crop,
            height: height - crop,
        };
        let output_size = (output.size() * std::mem::size_of::<Real>() as u32) as u64;
        let result_buffer = self.device.create_buffer("result", output_size);
        let kernel_buffer = self
            .device
            .create_data_buffer("kernel", bytemuck::cast_slice(&kernel.data));
        let params = [width, kernel.size];
        let params_data = bytemuck::cast_slice(&params);
        let params_buffer = self.device.create_uniform_buffer("params", params_data);

        // create bind group and compute pipeline
        let (bind_group, compute_pipeline) = self.device.create_compute_pipeline(
            &[
                (
                    &input_buffer,
                    4,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                (
                    &result_buffer,
                    4,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                (
                    &kernel_buffer,
                    4,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                (
                    &params_buffer,
                    params_data.len() as u64,
                    wgpu::BufferBindingType::Uniform,
                ),
            ],
            include_str!("convolution.wgsl"),
        );

        let mut cpass = self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_pipeline(&compute_pipeline);
        cpass.dispatch(output.width, output.height, 1);

        (result_buffer, (output.width, output.height))
    }

    pub async fn run<T: bytemuck::Pod>(
        mut self,
        output_buffers: &[(&wgpu::Buffer, (u32, u32), u32)],
    ) -> Vec<Vec<T>> {
        let mut output_offset_sizes = Vec::with_capacity(output_buffers.len());
        let mut offset = 0;
        for (result, image_size, pixel_size) in output_buffers {
            let size = (image_size.0 * image_size.1 * pixel_size) as u64;
            output_offset_sizes.push((result, offset, size));
            offset += size;
        }
        let output_buffer = self.device.create_output_buffer("output", offset);
        for (result, offset, size) in output_offset_sizes {
            self.encoder
                .copy_buffer_to_buffer(result, 0, &output_buffer, offset, size);
        }
        self.device.queue.submit(Some(self.encoder.finish()));

        // Read output
        let buffer_slice = output_buffer.slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        self.device.device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Ok(()) = buffer_future.await {
            let data = buffer_slice.get_mapped_range();
            let mut output = bytemuck::cast_slice::<u8, T>(&data).to_vec();

            // We have to make sure all mapped views are dropped before we unmap the buffer.
            drop(data);
            output_buffer.unmap();

            let mut outputs = Vec::with_capacity(output_buffers.len());
            for (_, image_size, _) in output_buffers {
                let size = (image_size.0 * image_size.1) as usize;
                let remained_data = output.split_off(size);
                outputs.push(output);
                output = remained_data;
            }
            outputs
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
