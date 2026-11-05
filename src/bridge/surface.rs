use crate::error::{Error, ErrorKind};
use rotex_types::{Extent2D, SurfaceDescriptor};

use super::{WgpuBridge, surface_not_attached_error};

pub(super) fn attach_surface(
    bridge: &mut WgpuBridge,
    surface_descriptor: SurfaceDescriptor,
) -> Result<(), Error> {
    let surface = bridge.instance.create_surface(
        surface_descriptor.display_handle,
        surface_descriptor.window_handle,
    )?;
    let extent = surface_descriptor.extent.clamped();
    let swapchain = surface.create_swapchain(&bridge.device, extent.width, extent.height)?;
    bridge.swapchain_format = Some(swapchain.config.format);
    bridge.surface = Some(surface);
    bridge.swapchain = Some(swapchain);
    bridge.rhi_pipeline_cache.clear();
    bridge.depth_target = None;
    Ok(())
}

pub(super) fn resize(bridge: &mut WgpuBridge, extent: Extent2D) -> Result<(), Error> {
    reconfigure_surface(bridge, extent.clamped())
}

pub(super) fn acquire_surface_texture(
    bridge: &mut WgpuBridge,
) -> Result<Option<wgpu::SurfaceTexture>, Error> {
    for _ in 0..2 {
        let surface = bridge
            .surface
            .as_ref()
            .ok_or_else(surface_not_attached_error)?;
        match surface.raw().get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => return Ok(Some(texture)),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let extent = bridge
                    .swapchain
                    .as_ref()
                    .map(|s| Extent2D {
                        width: s.config.width,
                        height: s.config.height,
                    })
                    .ok_or_else(surface_not_attached_error)?;
                reconfigure_surface(bridge, extent)?;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return Ok(None),
        }
    }

    Err(Error::recoverable(ErrorKind::SurfaceOutdated))
}

fn reconfigure_surface(bridge: &mut WgpuBridge, extent: Extent2D) -> Result<(), Error> {
    let surface = bridge
        .surface
        .as_ref()
        .ok_or_else(surface_not_attached_error)?;
    let swapchain = surface.create_swapchain(&bridge.device, extent.width, extent.height)?;
    bridge.swapchain_format = Some(swapchain.config.format);
    bridge.swapchain = Some(swapchain);
    bridge.depth_target = None;
    bridge.rhi_pipeline_cache.clear();
    Ok(())
}
