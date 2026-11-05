//! wgpu device, surface, and buffer wrappers used by [`WgpuBridge`](crate::WgpuBridge).
pub(crate) mod device;
mod surface;

pub use device::{WgpuDevice, WgpuInstance};
pub use surface::{WgpuSurface, WgpuSwapchain};

/// GPU buffer and the size requested when it was created.
pub struct WgpuBuffer {
    /// wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// Size in bytes. A request of zero allocates one byte.
    pub size: u64,
}

impl WgpuBuffer {
    /// Creates a buffer of at least one byte with `usage`.
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
