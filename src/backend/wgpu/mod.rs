pub(crate) mod device;
mod surface;

pub use device::{WgpuDevice, WgpuInstance};
pub use surface::{WgpuSurface, WgpuSwapchain};

pub struct WgpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
}

impl WgpuBuffer {
    pub fn new(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotex-wgpu-buffer"),
            size: size.max(1),
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }
}
