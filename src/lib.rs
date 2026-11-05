#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! wgpu backend for explicit command lists, including headless runs.
//!
//! [`WgpuGpuPlatform`] enumerates adapters and [`WgpuBridge`] translates
//! [`rotex_types::RhiCommand`] values through wgpu. The bridge keeps handle
//! tables and compiles pipelines from the shader packages the caller supplies.
//! It does not own scene state.
//!
//! Memory barriers are accepted; wgpu inserts its own hazards. Push constants
//! are recorded when the device supports immediates.

/// Device and surface wrappers.
pub mod backend;
/// Command translation and GPU resource tables.
pub mod bridge;
mod caps;
/// Re-exported RHI vocabulary.
pub mod core;
/// Errors returned by the wgpu backend.
pub mod error;
mod gpu_platform;

pub use backend::wgpu;
pub use bridge::WgpuBridge;
pub use error::{Error, ErrorKind, Severity};
pub use gpu_platform::WgpuGpuPlatform;

impl rotex_core::Supports<rotex_types::IndirectDraw> for WgpuBridge {}

/// Shared RHI vocabulary.
pub mod rotex_types {
    pub use ::rotex_types::*;
}

#[cfg(test)]
mod tests {
    use super::WgpuGpuPlatform;
    use super::rotex_types::{
        AbstractPipelineLayout, CullMode, DeviceDescriptor, GraphicsShaderPackage, IndexFormat,
        MaterialDescriptor, MeshDescriptor, PlatformDescriptor, ResourceBatchCreate,
        ResourceBatchUpdate, ResourceCreateDescriptor, ResourceHandle, ResourceUpdateDescriptor,
        ShaderPackage, ShaderPayload, ShaderStage, ShaderVariantMap, TextureDescriptor,
        TextureFormat, VertexAttribute, VertexBufferLayout, VertexFormat,
    };
    use rotex_core::{GpuPlatform, HardwareFamily};

    #[test]
    fn backend_matches_reported_capabilities() {
        pollster::block_on(async {
            let gpu = match WgpuGpuPlatform::new(PlatformDescriptor::default()).await {
                Ok(gpu) => gpu,
                Err(_) => return,
            };
            let adapters = match gpu.enumerate_adapters().await {
                Ok(adapters) => adapters,
                Err(_) => return,
            };
            let Some(adapter) = adapters.first() else {
                return;
            };
            let mut backend = match gpu
                .create_backend(adapter.id, DeviceDescriptor::default())
                .await
            {
                Ok(backend) => backend,
                Err(_) => return,
            };
            rotex_conformance::run(&mut backend).expect("conformance");
        });
    }

    #[test]
    fn headless_resource_lifecycle_smoke() {
        pollster::block_on(async {
            let gpu = match WgpuGpuPlatform::new(PlatformDescriptor::default()).await {
                Ok(gpu) => gpu,
                Err(_) => return,
            };
            let adapters = match gpu.enumerate_adapters().await {
                Ok(adapters) => adapters,
                Err(_) => return,
            };
            let adapter_id = adapters
                .iter()
                .find(|adapter| adapter.hardware_family == HardwareFamily::DiscreteGpu)
                .or(adapters.first())
                .map(|adapter| adapter.id);
            let adapter_id = match adapter_id {
                Some(id) => id,
                None => return,
            };
            let mut bridge = match gpu
                .create_backend(adapter_id, DeviceDescriptor::default())
                .await
            {
                Ok(bridge) => bridge,
                Err(_) => return,
            };

            let resources = bridge
                .create_resources(ResourceBatchCreate {
                    resources: vec![
                        ResourceCreateDescriptor::Mesh {
                            id: 1,
                            mesh: sample_mesh(
                                [
                                    [0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
                                    [-0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
                                    [0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
                                ],
                                vec![0, 1, 2],
                            ),
                        },
                        ResourceCreateDescriptor::Material {
                            id: 2,
                            material: MaterialDescriptor {
                                shaders: GraphicsShaderPackage {
                                    vertex: ShaderPackage {
                                        source_hash: 0,
                                        stage: ShaderStage::Vertex,
                                        entry_point: "vs_main".to_string(),
                                        layout: AbstractPipelineLayout {
                                            bind_groups: vec![],
                                            push_constants: vec![],
                                        },
                                        variants: ShaderVariantMap {
                                            spirv: Some(ShaderPayload::SpirV(vec![
                                                0x03, 0x02, 0x23, 0x07,
                                            ])),
                                            ..Default::default()
                                        },
                                    },
                                    fragment: ShaderPackage {
                                        source_hash: 0,
                                        stage: ShaderStage::Fragment,
                                        entry_point: "fs_main".to_string(),
                                        layout: AbstractPipelineLayout {
                                            bind_groups: vec![],
                                            push_constants: vec![],
                                        },
                                        variants: ShaderVariantMap {
                                            spirv: Some(ShaderPayload::SpirV(vec![
                                                0x03, 0x02, 0x23, 0x07,
                                            ])),
                                            ..Default::default()
                                        },
                                    },
                                    layout: AbstractPipelineLayout {
                                        bind_groups: vec![],
                                        push_constants: vec![],
                                    },
                                },
                                enable_depth: true,
                                cull_mode: CullMode::Back,
                                texture: None,
                            },
                        },
                        ResourceCreateDescriptor::Texture {
                            id: 3,
                            texture: TextureDescriptor {
                                width: 1,
                                height: 1,
                                format: TextureFormat::Rgba8Unorm,
                                data: vec![255, 255, 255, 255],
                                render_attachment: false,
                            },
                        },
                    ],
                })
                .expect("resource creation should succeed");

            let mut mesh_id = None;
            let mut material_id = None;
            let mut texture_id = None;
            for handle in resources.handles {
                match handle {
                    ResourceHandle::Mesh(id) => mesh_id = Some(id),
                    ResourceHandle::Material(id) => material_id = Some(id),
                    ResourceHandle::Texture(id) => texture_id = Some(id),
                    ResourceHandle::Buffer(_)
                    | ResourceHandle::ComputePipeline(_)
                    | ResourceHandle::BindGroupLayout(_)
                    | ResourceHandle::BindGroup(_)
                    | ResourceHandle::Sampler(_)
                    | ResourceHandle::GraphicsPipeline(_)
                    | ResourceHandle::QuerySet(_) => {}
                    _ => {}
                }
            }

            bridge
                .update_resources(ResourceBatchUpdate {
                    updates: vec![
                        ResourceUpdateDescriptor::Mesh {
                            id: mesh_id.expect("mesh id"),
                            vertex_streams: sample_mesh(
                                [
                                    [0.0, 0.25, 0.0, 1.0, 1.0, 1.0],
                                    [-0.25, -0.25, 0.0, 1.0, 1.0, 1.0],
                                    [0.25, -0.25, 0.0, 1.0, 1.0, 1.0],
                                ],
                                vec![0, 1, 2],
                            )
                            .vertex_streams,
                            index_data: vec![0, 0, 1, 0, 2, 0],
                            index_format: IndexFormat::Uint16,
                            index_count: 3,
                        },
                        ResourceUpdateDescriptor::Material {
                            id: material_id.expect("material id"),
                            enable_depth: Some(false),
                            texture: Some(texture_id),
                        },
                        ResourceUpdateDescriptor::Texture {
                            id: texture_id.expect("texture id"),
                            data: vec![0, 0, 0, 255],
                        },
                    ],
                })
                .expect("resource update should succeed");

            bridge.destroy();
        });
    }

    fn sample_mesh(vertices: [[f32; 6]; 3], indices: Vec<u16>) -> MeshDescriptor {
        let mut vertex_data = Vec::with_capacity(vertices.len() * 6 * std::mem::size_of::<f32>());
        for vertex in vertices {
            for component in vertex {
                vertex_data.extend_from_slice(component.to_le_bytes().as_slice());
            }
        }
        let index_count = indices.len() as u32;
        let mut index_data = Vec::with_capacity(indices.len() * std::mem::size_of::<u16>());
        for index in indices {
            index_data.extend_from_slice(index.to_le_bytes().as_slice());
        }
        MeshDescriptor::single(
            vertex_data,
            default_vertex_layout(),
            index_data,
            IndexFormat::Uint16,
            index_count,
        )
    }

    fn default_vertex_layout() -> VertexBufferLayout {
        VertexBufferLayout {
            array_stride: (6 * std::mem::size_of::<f32>()) as u64,
            step_mode: rotex_types::resource::VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    location: 0,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: (3 * std::mem::size_of::<f32>()) as u64,
                    location: 1,
                },
            ],
        }
    }
}
