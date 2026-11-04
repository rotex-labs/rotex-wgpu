#![allow(dead_code)]
use std::collections::HashMap;

use rotex_types::{
    IndexFormat, MaterialDescriptor, MaterialId, MeshId, TextureFormat, TextureId, VertexAttribute,
    VertexFormat,
};

#[derive(Debug, Clone)]
pub struct WgpuVertexLayout {
    pub array_stride: u64,
    pub attributes: Vec<wgpu::VertexAttribute>,
    pub step_mode: wgpu::VertexStepMode,
}

impl WgpuVertexLayout {
    pub fn as_wgpu(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.array_stride,
            step_mode: self.step_mode,
            attributes: &self.attributes,
        }
    }
}

pub struct WgpuMeshResource {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_format: wgpu::IndexFormat,
    pub index_count: u32,
    pub vertex_layout_id: u64,
    pub vertex_layout: WgpuVertexLayout,
}

impl WgpuMeshResource {
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

pub struct WgpuTextureResource {
    pub texture: wgpu::Texture,
    pub default_view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
    pub size: (u32, u32),
}

pub struct DepthTarget {
    pub _texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialPipelineKey {
    pub material_id: MaterialId,
    pub vertex_layout_id: u64,
    pub depth_enabled: bool,
}

pub struct WgpuComputePipelineResource {
    pub descriptor: rotex_types::resource::ComputePipelineDescriptor,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

pub struct WgpuGraphicsPipelineResource {
    pub pipeline: wgpu::RenderPipeline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthTargetKey {
    pub width: u32,
    pub height: u32,
}

#[derive(Default)]
pub struct ResourceStorage {
    pub meshes: HashMap<MeshId, WgpuMeshResource>,
    pub materials: HashMap<MaterialId, MaterialDescriptor>,
    pub textures: HashMap<TextureId, WgpuTextureResource>,
    pub compute_pipelines: HashMap<rotex_types::ComputePipelineId, WgpuComputePipelineResource>,
    pub buffers: HashMap<rotex_types::BufferId, crate::backend::wgpu::WgpuBuffer>,
    pub bind_group_layouts: HashMap<rotex_types::BindGroupLayoutId, wgpu::BindGroupLayout>,
    pub bind_groups: HashMap<rotex_types::BindGroupId, wgpu::BindGroup>,
}

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

pub fn vertex_format_size(format: VertexFormat) -> u64 {
    format.size()
}

pub fn wgpu_vertex_attribute(attribute: VertexAttribute) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: map_vertex_format(attribute.format),
        offset: attribute.offset,
        shader_location: attribute.location,
    }
}

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

pub fn bytes_per_pixel(format: TextureFormat) -> u32 {
    format.info().block_bytes
}

pub fn map_index_format(format: IndexFormat) -> wgpu::IndexFormat {
    match format {
        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
    }
}

pub fn index_format_size(format: IndexFormat) -> usize {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}
