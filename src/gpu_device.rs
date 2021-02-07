use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct GpuDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

pub fn create_gpu_device() -> GpuDevice {
    let (device, queue) = futures::executor::block_on(create_device_queue());
    GpuDevice { device, queue }
}

pub async fn create_device_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device")
}

pub fn create_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub fn create_data_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
    })
}

pub fn create_uniform_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsage::UNIFORM,
    })
}

pub fn create_output_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_bind_group(
    device: &wgpu::Device,
    buffers: &[(&wgpu::Buffer, u64, wgpu::BufferBindingType)],
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout_entries = buffers
        .iter()
        .enumerate()
        .map(|(index, (_, size, ty))| wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: *ty,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(*size),
            },
            count: None,
        })
        .collect::<Vec<_>>();
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &layout_entries,
    });
    let group_entries = buffers
        .iter()
        .enumerate()
        .map(|(index, (buffer, _, _))| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &group_entries,
    });
    (bind_group_layout, bind_group)
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    buffers: &[(&wgpu::Buffer, u64, wgpu::BufferBindingType)],
    shader: &str,
) -> (wgpu::BindGroup, wgpu::ComputePipeline) {
    let (bind_group_layout, bind_group) = create_bind_group(device, buffers);

    // create shader module
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
        flags: wgpu::ShaderFlags::VALIDATION,
    });

    // create pipeline for shader
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });
    (bind_group, compute_pipeline)
}
