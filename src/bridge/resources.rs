use crate::error::{Error, ErrorKind};
use rotex_types::{
    BindGroupEntryDescriptor, BindGroupId, BindGroupLayoutId, BufferDescriptor, BufferId,
    BufferUsage, BufferUsages, ComputePipelineId, CreatedResources, MaterialDescriptor, MaterialId,
    MeshDescriptor, MeshId, ResourceBatchCreate, ResourceBatchUpdate, ResourceCreateDescriptor,
    ResourceHandle, ResourceUpdateDescriptor, TextureDescriptor, TextureFormat, TextureId,
    VertexBufferLayout, VertexFormat, VertexStreamData,
};
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hasher};

use super::WgpuBridge;
use super::types::{
    WgpuMeshResource, WgpuTextureResource, WgpuVertexLayout, bytes_per_pixel, index_format_size,
    map_index_format, map_texture_format, vertex_format_size, wgpu_vertex_attribute,
};

pub(super) fn create_resources(
    bridge: &mut WgpuBridge,
    descriptor: ResourceBatchCreate,
) -> Result<CreatedResources, Error> {
    let mut handles = Vec::with_capacity(descriptor.resources.len());
    for resource in descriptor.resources {
        match resource {
            ResourceCreateDescriptor::Mesh { id, mesh } => {
                let id = MeshId(id);
                ensure_absent(&bridge.resources.meshes, &id)?;
                bridge
                    .resources
                    .meshes
                    .insert(id, create_wgpu_mesh(&bridge.device.raw, &mesh)?);
                handles.push(ResourceHandle::Mesh(id));
            }
            ResourceCreateDescriptor::Material { id, material } => {
                validate_material_descriptor(&material)?;
                let id = MaterialId(id);
                ensure_absent(&bridge.resources.materials, &id)?;
                bridge.resources.materials.insert(id, material);
                handles.push(ResourceHandle::Material(id));
            }
            ResourceCreateDescriptor::Texture { id, texture } => {
                let id = TextureId(id);
                ensure_absent(&bridge.resources.textures, &id)?;
                let gpu_texture = create_wgpu_texture(&bridge.device, texture)?;
                bridge.resources.textures.insert(id, gpu_texture);
                handles.push(ResourceHandle::Texture(id));
            }
            ResourceCreateDescriptor::Buffer { id, buffer: buf } => {
                let id = BufferId(id);
                ensure_absent(&bridge.resources.buffers, &id)?;
                let usage = map_wgpu_buffer_usage(&buf);
                let size = buf.size.max(1);
                let buffer = bridge.device.raw.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rotex-buffer"),
                    size,
                    usage,
                    mapped_at_creation: true,
                });
                if let Some(data) = &buf.initial_data {
                    let len = data.len().min(size as usize);
                    let mut mapped = buffer.slice(0..len as u64).get_mapped_range_mut();
                    mapped.copy_from_slice(&data[..len]);
                }
                buffer.unmap();
                bridge.resources.buffers.insert(
                    id,
                    crate::backend::wgpu::WgpuBuffer {
                        buffer,
                        size: buf.size,
                    },
                );
                handles.push(ResourceHandle::Buffer(id));
            }
            ResourceCreateDescriptor::BindGroupLayout { id, layout } => {
                let id = BindGroupLayoutId(id);
                ensure_absent(&bridge.resources.bind_group_layouts, &id)?;
                let wgpu_layout =
                    super::bindings::create_bind_group_layout(&bridge.device.raw, &layout)?;
                bridge.resources.bind_group_layouts.insert(id, wgpu_layout);
                handles.push(ResourceHandle::BindGroupLayout(id));
            }
            ResourceCreateDescriptor::BindGroup { id, group: bg } => {
                let id = BindGroupId(id);
                ensure_absent(&bridge.resources.bind_groups, &id)?;
                let layout =
                    bridge
                        .resources
                        .bind_group_layouts
                        .get(&bg.layout)
                        .ok_or(Error::fatal(ErrorKind::Unsupported(
                            "bind group layout not found",
                        )))?;
                let mut entries = Vec::new();
                for entry in &bg.entries {
                    match entry {
                        BindGroupEntryDescriptor::Buffer {
                            binding,
                            buffer,
                            offset,
                            size,
                        } => {
                            let buf_res =
                                bridge.resources.buffers.get(buffer).ok_or(Error::fatal(
                                    ErrorKind::Unsupported("buffer not found"),
                                ))?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: *binding,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &buf_res.buffer,
                                    offset: *offset,
                                    size: if *size == 0 {
                                        None
                                    } else {
                                        Some(std::num::NonZeroU64::new(*size).ok_or_else(|| {
                                            Error::fatal(ErrorKind::Unsupported(
                                                "buffer binding size is zero",
                                            ))
                                        })?)
                                    },
                                }),
                            });
                        }
                        BindGroupEntryDescriptor::Texture { binding, texture } => {
                            let tex_res =
                                bridge.resources.textures.get(texture).ok_or_else(|| {
                                    Error::fatal(ErrorKind::Unsupported(
                                        "texture not found for bind group",
                                    ))
                                })?;
                            entries.push(wgpu::BindGroupEntry {
                                binding: *binding,
                                resource: wgpu::BindingResource::TextureView(&tex_res.default_view),
                            });
                        }
                    }
                }
                let wgpu_bg = bridge
                    .device
                    .raw
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rotex-bg"),
                        layout,
                        entries: &entries,
                    });
                bridge.resources.bind_groups.insert(id, wgpu_bg);
                handles.push(ResourceHandle::BindGroup(id));
            }
            ResourceCreateDescriptor::ComputePipeline { id, pipeline } => {
                let id = ComputePipelineId(id);
                ensure_absent(&bridge.resources.compute_pipelines, &id)?;
                let result =
                    super::compute_pipeline_cache::create_compute_pipeline(bridge, &pipeline)?;
                bridge.resources.compute_pipelines.insert(id, result);
                handles.push(ResourceHandle::ComputePipeline(id));
            }
            _ => return Err(Error::unsupported("resource create not implemented")),
        }
    }
    Ok(CreatedResources { handles })
}

pub(super) fn update_resources(
    bridge: &mut WgpuBridge,
    descriptor: ResourceBatchUpdate,
) -> Result<(), Error> {
    for update in descriptor.updates {
        match update {
            ResourceUpdateDescriptor::Mesh {
                id,
                vertex_streams,
                index_data,
                index_format,
                index_count,
            } => {
                let mesh = MeshDescriptor {
                    vertex_streams,
                    index_data,
                    index_format,
                    index_count,
                };
                let gpu_mesh = create_wgpu_mesh(&bridge.device.raw, &mesh)?;
                bridge.resources.meshes.insert(id, gpu_mesh);
            }
            ResourceUpdateDescriptor::Material {
                id,
                enable_depth,
                texture,
            } => {
                let material =
                    bridge.resources.materials.get_mut(&id).ok_or_else(|| {
                        Error::recoverable(ErrorKind::ResourceNotFound("material"))
                    })?;
                if let Some(depth) = enable_depth {
                    material.enable_depth = depth;
                }
                if let Some(texture_update) = texture {
                    if let Some(texture_id) = texture_update {
                        if !bridge.resources.textures.contains_key(&texture_id) {
                            return Err(Error::recoverable(ErrorKind::ResourceNotFound("texture")));
                        }
                    }
                    material.texture = texture_update;
                }
                bridge
                    .rhi_pipeline_cache
                    .retain(|key, _| key.material_id != id);
            }
            ResourceUpdateDescriptor::MeshVertices { .. }
            | ResourceUpdateDescriptor::Buffer { .. } => {}
            ResourceUpdateDescriptor::Destroy { handle } => {
                destroy_resource(bridge, handle)?;
            }
            ResourceUpdateDescriptor::Texture { id, data } => {
                let texture =
                    bridge.resources.textures.get(&id).ok_or_else(|| {
                        Error::recoverable(ErrorKind::ResourceNotFound("texture"))
                    })?;
                write_texture_data(
                    &bridge.device,
                    texture.format,
                    texture.size.0,
                    texture.size.1,
                    &texture.texture,
                    &data,
                )?;
            }
            _ => return Err(Error::unsupported("resource update not implemented")),
        }
    }
    Ok(())
}

fn destroy_resource(bridge: &mut WgpuBridge, handle: ResourceHandle) -> Result<(), Error> {
    match handle {
        ResourceHandle::Mesh(id) => {
            bridge
                .resources
                .meshes
                .remove(&id)
                .ok_or_else(|| Error::unsupported("mesh not found"))?;
        }
        ResourceHandle::Texture(id) => {
            bridge
                .resources
                .textures
                .remove(&id)
                .ok_or_else(|| Error::unsupported("texture not found"))?;
        }
        ResourceHandle::Material(id) => {
            if bridge.resources.materials.remove(&id).is_none() {
                return Err(Error::unsupported("material not found"));
            }
            bridge
                .rhi_pipeline_cache
                .retain(|key, _| key.material_id != id);
        }
        ResourceHandle::Buffer(id) => {
            bridge
                .resources
                .buffers
                .remove(&id)
                .ok_or_else(|| Error::unsupported("buffer not found"))?;
        }
        ResourceHandle::ComputePipeline(id) => {
            bridge
                .resources
                .compute_pipelines
                .remove(&id)
                .ok_or_else(|| Error::unsupported("compute pipeline not found"))?;
        }
        ResourceHandle::BindGroupLayout(id) => {
            bridge
                .resources
                .bind_group_layouts
                .remove(&id)
                .ok_or_else(|| Error::unsupported("bind group layout not found"))?;
        }
        ResourceHandle::BindGroup(id) => {
            bridge
                .resources
                .bind_groups
                .remove(&id)
                .ok_or_else(|| Error::unsupported("bind group not found"))?;
        }
        ResourceHandle::Sampler(_)
        | ResourceHandle::GraphicsPipeline(_)
        | ResourceHandle::QuerySet(_) => {
            return Err(Error::unsupported("resource destroy not implemented"));
        }
        _ => return Err(Error::unsupported("resource destroy not implemented")),
    }
    Ok(())
}

fn ensure_absent<K, V>(map: &std::collections::HashMap<K, V>, key: &K) -> Result<(), Error>
where
    K: Eq + std::hash::Hash,
{
    if map.contains_key(key) {
        Err(Error::unsupported("resource id already exists"))
    } else {
        Ok(())
    }
}

fn create_wgpu_mesh(
    device: &wgpu::Device,
    mesh: &MeshDescriptor,
) -> Result<WgpuMeshResource, Error> {
    if mesh.index_count == 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_missing_indices",
        )));
    }
    if mesh.index_data.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_missing_index_data",
        )));
    }

    let (vertex_data, vertex_layout) = match &mesh.vertex_streams[0].data {
        VertexStreamData::Static(data) => {
            if data.is_empty() {
                return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                    "mesh_missing_vertex_data",
                )));
            }
            (data.as_slice(), &mesh.vertex_streams[0].layout)
        }
        VertexStreamData::External(_) => (&[][..], &mesh.vertex_streams[0].layout),
    };

    let index_stride = index_format_size(mesh.index_format);
    let expected_index_len = (mesh.index_count as usize)
        .checked_mul(index_stride)
        .ok_or_else(|| Error::recoverable(ErrorKind::InvalidDescriptor("index_size_overflow")))?;
    if mesh.index_data.len() != expected_index_len {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "mesh_index_data_size_mismatch",
        )));
    }

    let vertex_layout = translate_vertex_layout(vertex_layout)?;

    let vertex_buffer = create_buffer(
        device,
        wgpu::BufferUsages::VERTEX,
        vertex_data,
        "rotex-wgpu-vertex-buffer",
    );
    let index_buffer = create_buffer(
        device,
        wgpu::BufferUsages::INDEX,
        mesh.index_data.as_slice(),
        "rotex-wgpu-index-buffer",
    );

    Ok(WgpuMeshResource {
        vertex_buffer,
        index_buffer,
        index_format: map_index_format(mesh.index_format),
        index_count: mesh.index_count,
        vertex_layout_id: hash_vertex_layout(&mesh.vertex_streams[0].layout),
        vertex_layout,
    })
}

fn create_buffer<T: Copy>(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    let size = std::mem::size_of_val(data) as u64;
    let aligned_size = align_to_u64(size.max(1), wgpu::COPY_BUFFER_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage,
        mapped_at_creation: true,
    });
    if size > 0 {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size as usize) };
        let mut staged = vec![0_u8; aligned_size as usize];
        staged[..bytes.len()].copy_from_slice(bytes);
        let mut mapped = buffer.slice(..).get_mapped_range_mut();
        mapped.copy_from_slice(staged.as_slice());
    }
    buffer.unmap();
    buffer
}

fn create_wgpu_texture(
    device: &crate::backend::wgpu::WgpuDevice,
    texture: TextureDescriptor,
) -> Result<WgpuTextureResource, Error> {
    let format = map_texture_format(texture.format)
        .ok_or_else(|| Error::unsupported("texture format not implemented"))?;
    let width = texture.width.max(1);
    let height = texture.height.max(1);
    let raw_texture = device.raw.create_texture(&wgpu::TextureDescriptor {
        label: Some("rotex-wgpu-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    write_texture_data(device, format, width, height, &raw_texture, &texture.data)?;

    let default_view = raw_texture.create_view(&wgpu::TextureViewDescriptor::default());

    Ok(WgpuTextureResource {
        texture: raw_texture,
        default_view,
        format,
        size: (width, height),
    })
}

struct TextureUploadPlan {
    height: u32,
    bytes_per_row: u32,
    rows_per_image: u32,
    expected_len: usize,
    extent: wgpu::Extent3d,
}

fn write_texture_data(
    device: &crate::backend::wgpu::WgpuDevice,
    texture_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    texture: &wgpu::Texture,
    data: &[u8],
) -> Result<(), Error> {
    let upload = validate_texture_upload(texture_format, width, height, data)?;
    let src = &data[..upload.expected_len];
    device.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        src,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(upload.bytes_per_row),
            rows_per_image: Some(upload.rows_per_image),
        },
        upload.extent,
    );
    Ok(())
}

fn validate_material_descriptor(material: &MaterialDescriptor) -> Result<(), Error> {
    let has_vert = material.shaders.vertex.spirv_bytes().is_some()
        || material.shaders.vertex.wgsl_source().is_some();
    let has_frag = material.shaders.fragment.spirv_bytes().is_some()
        || material.shaders.fragment.wgsl_source().is_some();
    if !has_vert || !has_frag {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_source_missing",
        )));
    }
    if material.shaders.vertex.entry_point.is_empty()
        || material.shaders.fragment.entry_point.is_empty()
    {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "material_shader_entry_missing",
        )));
    }
    Ok(())
}

fn translate_vertex_layout(layout: &VertexBufferLayout) -> Result<WgpuVertexLayout, Error> {
    if layout.array_stride == 0 {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "vertex_layout_zero_stride",
        )));
    }
    if layout.attributes.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "vertex_layout_missing_attributes",
        )));
    }

    let mut attribute_locations = HashSet::with_capacity(layout.attributes.len());
    let mut translated = Vec::with_capacity(layout.attributes.len());
    for attribute in layout.attributes.iter().cloned() {
        let size = vertex_format_size(attribute.format);
        let end = attribute.offset.checked_add(size).ok_or_else(|| {
            Error::recoverable(ErrorKind::InvalidDescriptor("vertex_attribute_overflow"))
        })?;
        if end > layout.array_stride {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "vertex_attribute_out_of_bounds",
            )));
        }
        if !attribute_locations.insert(attribute.location) {
            return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
                "vertex_attribute_duplicate_location",
            )));
        }
        translated.push(wgpu_vertex_attribute(attribute));
    }

    let step_mode = match layout.step_mode {
        rotex_types::VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        rotex_types::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    };
    Ok(WgpuVertexLayout {
        array_stride: layout.array_stride,
        attributes: translated,
        step_mode,
    })
}

fn hash_vertex_layout(layout: &VertexBufferLayout) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u64(layout.array_stride);
    hasher.write_u32(match layout.step_mode {
        rotex_types::VertexStepMode::Vertex => 0,
        rotex_types::VertexStepMode::Instance => 1,
    });
    for attribute in &layout.attributes {
        hasher.write_u32(attribute.location);
        hasher.write_u64(attribute.offset);
        hasher.write_u8(vertex_format_tag(attribute.format));
    }
    hasher.finish()
}

fn vertex_format_tag(format: VertexFormat) -> u8 {
    match format {
        VertexFormat::Float32 => 0,
        VertexFormat::Float32x2 => 1,
        VertexFormat::Float32x3 => 2,
        VertexFormat::Float32x4 => 3,
        VertexFormat::Uint32 => 4,
        VertexFormat::Unorm8x4 => 5,
        VertexFormat::Float16x2 => 6,
        VertexFormat::Sint32 => 7,
    }
}

fn validate_texture_upload(
    texture_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
) -> Result<TextureUploadPlan, Error> {
    let mapped_format = match texture_format {
        wgpu::TextureFormat::R8Unorm => TextureFormat::R8Unorm,
        wgpu::TextureFormat::Rg8Unorm => TextureFormat::Rg8Unorm,
        wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::R32Float => TextureFormat::R32Float,
        wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba32Float => TextureFormat::Rgba32Float,
        wgpu::TextureFormat::Depth16Unorm => TextureFormat::Depth16Unorm,
        wgpu::TextureFormat::Depth24Plus => TextureFormat::Depth24Plus,
        wgpu::TextureFormat::Depth32Float => TextureFormat::Depth32Float,
        wgpu::TextureFormat::Depth24PlusStencil8 => TextureFormat::Depth24PlusStencil8,
        _ => {
            return Err(Error::recoverable(ErrorKind::Unsupported(
                "texture_format_not_supported",
            )));
        }
    };
    let width = width.max(1);
    let height = height.max(1);
    let bpp = bytes_per_pixel(mapped_format) as u64;
    let expected = (width as u64)
        .checked_mul(height as u64)
        .and_then(|value| value.checked_mul(bpp))
        .ok_or_else(|| Error::recoverable(ErrorKind::InvalidDescriptor("texture_size_overflow")))?;
    let expected_len = usize::try_from(expected)
        .map_err(|_| Error::recoverable(ErrorKind::InvalidDescriptor("texture_size_overflow")))?;
    if data.len() < expected_len {
        return Err(Error::recoverable(ErrorKind::TextureUploadFailed(
            "texture_data_too_small",
        )));
    }

    Ok(TextureUploadPlan {
        height,
        bytes_per_row: (width as u64 * bpp) as u32,
        rows_per_image: height,
        expected_len,
        extent: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    })
}

fn map_wgpu_buffer_usage(desc: &BufferDescriptor) -> wgpu::BufferUsages {
    let usages = desc.effective_usages();
    let mut flags = wgpu::BufferUsages::COPY_DST;
    if usages.contains(BufferUsages::VERTEX) {
        flags |= wgpu::BufferUsages::VERTEX;
    }
    if usages.contains(BufferUsages::INDEX) {
        flags |= wgpu::BufferUsages::INDEX;
    }
    if usages.contains(BufferUsages::UNIFORM) {
        flags |= wgpu::BufferUsages::UNIFORM;
    }
    if usages.contains(BufferUsages::STORAGE) {
        flags |= wgpu::BufferUsages::STORAGE;
    }
    if flags == wgpu::BufferUsages::COPY_DST {
        match desc.usage {
            BufferUsage::Vertex => flags |= wgpu::BufferUsages::VERTEX,
            BufferUsage::Index => flags |= wgpu::BufferUsages::INDEX,
            BufferUsage::Uniform => flags |= wgpu::BufferUsages::UNIFORM,
            BufferUsage::Storage => flags |= wgpu::BufferUsages::STORAGE,
        }
    }
    flags
}

fn align_to_u64(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}
