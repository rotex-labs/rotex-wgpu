#![allow(dead_code)]
mod bindings;
mod compute_pipeline_cache;
mod extended;
mod immediate_uniform;
pub(crate) mod init;
mod render;
mod resources;
mod shader_cache;
mod surface;
mod types;

use std::collections::HashMap;

use crate::backend::wgpu::{WgpuDevice, WgpuInstance, WgpuSurface, WgpuSwapchain};
use crate::error::{Error, ErrorKind};
use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuBackend, Severity as CoreSeverity,
};
use rotex_types::{
    BufferId, ComputePipelineId, CreatedResources, Extent2D, ResourceBatchCreate,
    ResourceBatchUpdate, RhiCommand, SurfaceDescriptor, TextureId, TextureReadback,
};

use self::shader_cache::ShaderCacheKey;
use self::types::{DepthTarget, MaterialPipelineKey, ResourceStorage};

/// Translates [`rotex_types::RhiCommand`] values into wgpu calls.
pub struct WgpuBridge {
    pub(crate) instance: WgpuInstance,
    pub(crate) device: WgpuDevice,
    pub(crate) surface: Option<WgpuSurface>,
    pub(crate) swapchain: Option<WgpuSwapchain>,
    pub(crate) resources: ResourceStorage,
    pub(crate) depth_target: Option<DepthTarget>,
    pub(crate) rhi_pipeline_cache:
        HashMap<MaterialPipelineKey, types::WgpuGraphicsPipelineResource>,
    pub(crate) shader_module_cache: HashMap<ShaderCacheKey, wgpu::ShaderModule>,
    pub(crate) compute_bind_groups:
        HashMap<(ComputePipelineId, BufferId), Vec<(u32, wgpu::BindGroup)>>,
    pub(crate) recording_encoder: Option<wgpu::CommandEncoder>,
    pub(crate) surface_texture: Option<wgpu::SurfaceTexture>,
    pub(crate) swapchain_color_view: Option<wgpu::TextureView>,
    pub(crate) swapchain_format: Option<wgpu::TextureFormat>,
    pub(crate) current_frame_index: u32,
    pub(crate) active_pass: Option<render::ActiveGraphicsPass>,
    pub(crate) render_bundle: Option<wgpu::RenderBundle>,
    pub(crate) pending_acquire_skips: u32,
    pub(crate) compute_pass_open: bool,
    pub(crate) active_compute_pipeline: Option<ComputePipelineId>,
    pub(crate) compute_binds: Vec<(u32, rotex_types::BindGroupId, Vec<u32>)>,
    pub(crate) compute_immediates: Option<(u32, Vec<u8>)>,
    pub(crate) fallback_sampler: wgpu::Sampler,
    pub(crate) immediate_uniform: Option<immediate_uniform::ImmediateUniform>,
    /// Graphics pipelines whose push constants were lowered to a group-1 uniform.
    /// The value is true when group 0 is an empty layout inserted for that pipeline.
    pub(crate) uniform_graphics: HashMap<rotex_types::GraphicsPipelineId, bool>,
}

impl WgpuBridge {
    /// Creates a surface and configures a swapchain.
    pub fn attach_surface(&mut self, surface_descriptor: SurfaceDescriptor) -> Result<(), Error> {
        surface::attach_surface(self, surface_descriptor)
    }

    /// Creates the buffers, textures, meshes, and pipelines named in `descriptor`.
    pub fn create_resources(
        &mut self,
        descriptor: ResourceBatchCreate,
    ) -> Result<CreatedResources, Error> {
        resources::create_resources(self, descriptor)
    }

    /// Replaces the contents or parameters of resources the bridge already holds.
    pub fn update_resources(&mut self, descriptor: ResourceBatchUpdate) -> Result<(), Error> {
        resources::update_resources(self, descriptor)
    }

    /// Records `commands` and submits a frame when the list asks for it.
    pub fn execute(&mut self, commands: &[RhiCommand]) -> Result<(), Error> {
        render::execute(self, commands)
    }

    /// Reconfigures the swapchain at `extent`.
    pub fn resize(&mut self, extent: Extent2D) -> Result<(), Error> {
        surface::resize(self, extent)
    }

    /// Copies a texture back to the CPU.
    pub fn read_texture(&mut self, id: TextureId) -> Result<TextureReadback, Error> {
        extended::read_texture(self, id)
    }

    /// Drops the bridge and the GPU resources it owns.
    pub fn destroy(self) {}

    /// Names the RHI features this backend does not implement.
    pub fn unsupported_feature_reporting() -> &'static [&'static str] {
        crate::caps::unsupported_feature_reporting()
    }
}

pub(crate) fn surface_not_attached_error() -> Error {
    Error::fatal(ErrorKind::SurfaceNotAttached)
}

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        crate::error::ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        crate::error::ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        other => CoreErrorKind::Backend(format!("{other:?}")),
    };
    CoreError { kind, severity }
}

impl GpuBackend for WgpuBridge {
    fn capabilities(&self) -> rotex_types::BackendCapabilities {
        crate::caps::capabilities_for(&self.device.raw)
    }

    fn attach_surface(
        &mut self,
        surface_descriptor: rotex_types::SurfaceDescriptor,
    ) -> Result<(), CoreError> {
        WgpuBridge::attach_surface(self, surface_descriptor).map_err(to_core_error)
    }

    fn create_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchCreate,
    ) -> Result<rotex_types::CreatedResources, CoreError> {
        WgpuBridge::create_resources(self, descriptor).map_err(to_core_error)
    }

    fn update_resources(
        &mut self,
        descriptor: rotex_types::ResourceBatchUpdate,
    ) -> Result<(), CoreError> {
        WgpuBridge::update_resources(self, descriptor).map_err(to_core_error)
    }

    fn execute(&mut self, commands: &[rotex_types::RhiCommand]) -> Result<(), CoreError> {
        WgpuBridge::execute(self, commands).map_err(to_core_error)
    }

    fn resize(&mut self, extent: rotex_types::Extent2D) -> Result<(), CoreError> {
        WgpuBridge::resize(self, extent).map_err(to_core_error)
    }

    fn read_texture(
        &mut self,
        id: rotex_types::TextureId,
    ) -> Result<rotex_types::TextureReadback, CoreError> {
        WgpuBridge::read_texture(self, id).map_err(to_core_error)
    }

    fn destroy(self: Box<Self>) {
        WgpuBridge::destroy(*self);
    }

    fn invalidate_command_cache(&mut self) {
        self.render_bundle = None;
    }
}
