use std::collections::HashMap;

use crate::backend::wgpu::{WgpuDevice, WgpuInstance};
use crate::error::Error;

use super::WgpuBridge;
use super::types::ResourceStorage;

pub(crate) fn init_bridge(instance: WgpuInstance, device: WgpuDevice) -> Result<WgpuBridge, Error> {
    let fallback_sampler = device.raw.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("rotex-wgpu-fallback-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        ..Default::default()
    });
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
        compute_pass_open: false,
        active_compute_pipeline: None,
        compute_binds: Vec::new(),
        compute_immediates: None,
        fallback_sampler,
        immediate_uniform: None,
        uniform_graphics: HashMap::new(),
    })
}
