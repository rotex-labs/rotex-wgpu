#![allow(dead_code)]
use std::collections::HashMap;

use rotex_types::{
    IndexFormat, MaterialDescriptor, MaterialId, MeshId, TextureFormat, TextureId, VertexAttribute,
    VertexFormat,
};

/// Vertex layout translated for wgpu.
#[derive(Debug, Clone)]
pub struct WgpuVertexLayout {
    /// Bytes from one vertex or instance to the next.
    pub array_stride: u64,
    /// Shader locations sourced from this buffer.
    pub attributes: Vec<wgpu::VertexAttribute>,
    /// Whether the buffer advances per vertex or per instance.
    pub step_mode: wgpu::VertexStepMode,
}

impl WgpuVertexLayout {
    /// Borrows this layout as wgpu vertex-buffer state.
    pub fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

/// Vertex and index buffers for one mesh.
pub struct WgpuMeshResource {
    /// Vertex buffer.
    pub vertex_buffer: wgpu::Buffer,
    /// Index buffer.
    pub index_buffer: wgpu::Buffer,
    /// Width of each index.
    pub index_format: wgpu::IndexFormat,
    /// Number of indices.
    pub index_count: u32,
    /// Hash of [`Self::vertex_layout`], used as a pipeline-cache key.
    pub vertex_layout_id: u64,
    /// Layout of [`Self::vertex_buffer`].
    pub vertex_layout: WgpuVertexLayout,
}

impl WgpuMeshResource {
    /// Returns the layout id, layout, vertex buffer, index buffer, index format, and index count.
    pub fn to_draw_inputs(
        &self,
    ) -> (
        u64,
        WgpuVertexLayout,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::IndexFormat,
        u32,
    ) {
        (
            self.vertex_layout_id,
            self.vertex_layout.clone(),
            self.vertex_buffer.clone(),
            self.index_buffer.clone(),
            self.index_format,
            self.index_count,
        )
    }
}

/// Texture, its default view, and the size it was created with.
pub struct WgpuTextureResource {
    /// GPU texture.
    pub texture: wgpu::Texture,
    /// View of the whole texture.
    pub default_view: wgpu::TextureView,
    /// wgpu format the texture was allocated with.
    pub format: wgpu::TextureFormat,
    /// RHI format requested by the caller.
    pub source_format: TextureFormat,
    /// Width and height in texels.
    pub size: (u32, u32),
}

/// Query set and the kind of queries it stores.
pub struct WgpuQuerySet {
    /// wgpu query set.
    pub query_set: wgpu::QuerySet,
    /// Kind of query stored in [`Self::query_set`].
    pub ty: rotex_types::QueryType,
}

/// Depth texture kept for one swapchain size.
pub struct DepthTarget {
    /// Depth texture. Retained so [`Self::view`] stays valid.
    pub _texture: wgpu::Texture,
    /// View used as a depth attachment.
    pub view: wgpu::TextureView,
    /// Width and height in pixels.
    pub size: (u32, u32),
}

/// Cache key for a graphics pipeline derived from a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialPipelineKey {
    /// Material the pipeline was compiled from.
    pub material_id: MaterialId,
    /// Vertex layout the pipeline was compiled for.
    pub vertex_layout_id: u64,
    /// Whether the pipeline writes and tests depth.
    pub depth_enabled: bool,
}

/// Compute pipeline and the bind-group layouts it was created with.
pub struct WgpuComputePipelineResource {
    /// Descriptor the pipeline was compiled from.
    pub descriptor: rotex_types::resource::ComputePipelineDescriptor,
    /// Compiled compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// Layouts for the pipeline bind groups, in set order.
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    /// Push constants were rewritten to the group-1 uniform.
    pub uniform_immediate: bool,
}

/// Compiled graphics pipeline.
pub struct WgpuGraphicsPipelineResource {
    /// wgpu render pipeline.
    pub pipeline: wgpu::RenderPipeline,
}

/// Width and height that identify a cached depth target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthTargetKey {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// GPU resources created through the bridge, keyed by RHI handle.
#[derive(Default)]
pub struct ResourceStorage {
    /// Meshes.
    pub meshes: HashMap<MeshId, WgpuMeshResource>,
    /// Materials, retained so pipelines can be compiled later.
    pub materials: HashMap<MaterialId, MaterialDescriptor>,
    /// Textures.
    pub textures: HashMap<TextureId, WgpuTextureResource>,
    /// Compute pipelines.
    pub compute_pipelines: HashMap<rotex_types::ComputePipelineId, WgpuComputePipelineResource>,
    /// Storage, uniform, and copy buffers.
    pub buffers: HashMap<rotex_types::BufferId, crate::backend::wgpu::WgpuBuffer>,
    /// Bind-group layouts.
    pub bind_group_layouts: HashMap<rotex_types::BindGroupLayoutId, wgpu::BindGroupLayout>,
    /// Sampler bindings in each layout. Missing group entries use the fallback sampler.
    pub layout_sampler_bindings: HashMap<rotex_types::BindGroupLayoutId, Vec<u32>>,
    /// Bind groups.
    pub bind_groups: HashMap<rotex_types::BindGroupId, wgpu::BindGroup>,
    /// Samplers.
    pub samplers: HashMap<rotex_types::SamplerId, wgpu::Sampler>,
    /// Explicit graphics pipelines.
    pub graphics_pipelines: HashMap<rotex_types::GraphicsPipelineId, wgpu::RenderPipeline>,
    /// Query sets.
    pub query_sets: HashMap<rotex_types::QuerySetId, WgpuQuerySet>,
    /// Shader artifacts loaded for their WGSL payload.
    pub shaders: HashMap<rotex_types::ShaderId, rotex_types::ShaderArtifact>,
}

/// Maps an RHI vertex format to wgpu.
pub fn map_vertex_format(format: VertexFormat) -> wgpu::VertexFormat {
    match format {
        VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
        VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
        VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
    }
}

/// Size of one vertex attribute, in bytes.
pub fn vertex_format_size(format: VertexFormat) -> u64 {
    format.size()
}

/// Maps one RHI vertex attribute to wgpu.
pub fn wgpu_vertex_attribute(attribute: VertexAttribute) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: map_vertex_format(attribute.format),
        offset: attribute.offset,
        shader_location: attribute.location,
    }
}

/// Maps an RHI texture format to wgpu, or `None` when wgpu has no equivalent.
pub fn map_texture_format(format: TextureFormat) -> Option<wgpu::TextureFormat> {
    Some(match format {
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        TextureFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        _ => return None,
    })
}

/// Bytes in one texel of `format`.
pub fn bytes_per_pixel(format: TextureFormat) -> u32 {
    format.info().block_bytes
}

/// Maps an RHI index format to wgpu.
pub fn map_index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

/// Size of one index, in bytes.
pub fn index_format_size(format: IndexFormat) -> usize {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}
