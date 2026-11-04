use std::collections::HashMap;

use crate::backend::wgpu::{WgpuDevice, WgpuInstance};
use crate::error::Error;

use super::WgpuBridge;
use super::types::ResourceStorage;

pub(crate) fn init_bridge(instance: WgpuInstance, device: WgpuDevice) -> Result<WgpuBridge, Error> {
    Ok(WgpuBridge {
        instance,
        device,
        surface: None,
        swapchain: None,
        resources: ResourceStorage::default(),
        depth_target: None,
        rhi_pipeline_cache: HashMap::new(),
        shader_module_cache: HashMap::new(),
        compute_bind_groups: HashMap::new(),
        recording_encoder: None,
        surface_texture: None,
        swapchain_color_view: None,
        swapchain_format: None,
        current_frame_index: 0,
        active_pass: None,
        render_bundle: None,
        pending_acquire_skips: 0,
        pre_present_hook: None,
    })
}
