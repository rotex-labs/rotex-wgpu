use super::WgpuDevice;
use crate::error::{Error, ErrorKind};

/// Surface that can configure a swapchain.
pub struct WgpuSurface {
    raw: wgpu::Surface<'static>,
}

/// Surface configuration currently applied to a [`WgpuSurface`].
pub struct WgpuSwapchain {
    /// Configuration last applied to the surface.
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuSurface {
    /// Wraps an existing wgpu surface.
    pub fn new(raw: wgpu::Surface<'static>) -> Self {
        Self { raw }
    }

    /// Configures the surface for a swapchain of at least `width` by `height` pixels.
    pub fn create_swapchain(
        &self,
        device: &WgpuDevice,
        width: u32,
        height: u32,
    ) -> Result<WgpuSwapchain, Error> {
        let caps = self.raw.get_capabilities(&device.adapter);
        let format = *caps
            .formats
            .first()
            .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?;
        let view_formats = if format.is_srgb() {
            let linear = match format {
                wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
                _ => format,
            };
            vec![linear]
        } else {
            let srgb = match format {
                wgpu::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8UnormSrgb,
                _ => format,
            };
            vec![srgb]
        };
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else {
            caps.present_modes
                .first()
                .copied()
                .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?
        };
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes
                .first()
                .copied()
                .ok_or_else(|| Error::fatal(ErrorKind::NoCompatibleDevice))?
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode,
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        self.raw.configure(&device.raw, &config);
        Ok(WgpuSwapchain { config })
    }

    /// Underlying wgpu surface.
    pub fn raw(&self) -> &wgpu::Surface<'static> {
        &self.raw
    }
}
