#![allow(dead_code)]
use rotex_types::{
    AbstractPipelineLayout, BindGroupLayoutDescriptor, BindingType, ShaderStageFlags,
};

pub fn build_set_layouts_from_abstract_layout(
    device: &wgpu::Device,
    layout: &AbstractPipelineLayout,
) -> Result<Vec<wgpu::BindGroupLayout>, crate::error::Error> {
    if layout.bind_groups.is_empty() {
        return Ok(Vec::new());
    }
    let max_set = layout
        .bind_groups
        .iter()
        .map(|group| group.set)
        .max()
        .unwrap_or(0);
    let mut layouts = Vec::with_capacity(max_set as usize + 1);
    for set in 0..=max_set {
        if let Some(desc) = layout.bind_groups.iter().find(|group| group.set == set) {
            layouts.push(create_bind_group_layout(device, desc)?);
        } else {
            layouts.push(create_empty_bind_group_layout(device));
        }
    }
    Ok(layouts)
}

pub fn build_material_set_layouts(
    device: &wgpu::Device,
    layout: &AbstractPipelineLayout,
) -> Result<Vec<wgpu::BindGroupLayout>, crate::error::Error> {
    build_set_layouts_from_abstract_layout(device, layout)
}

pub fn create_bind_group_layout(
    device: &wgpu::Device,
    desc: &BindGroupLayoutDescriptor,
) -> Result<wgpu::BindGroupLayout, crate::error::Error> {
    let entries: Vec<_> = desc
        .entries
        .iter()
        .map(|entry| {
            Ok(wgpu::BindGroupLayoutEntry {
                binding: entry.binding,
                visibility: map_shader_stages(entry.visibility),
                ty: map_binding_type(entry.ty, entry.readonly)?,
                count: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rotex-wgpu-bind-group-layout"),
            entries: &entries,
        }),
    )
}

pub fn create_empty_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rotex-wgpu-empty-bind-group-layout"),
        entries: &[],
    })
}

pub fn map_binding_type(
    ty: BindingType,
    readonly: bool,
) -> Result<wgpu::BindingType, crate::error::Error> {
    Ok(match ty {
        BindingType::UniformBuffer | BindingType::UniformBufferDynamic => {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: matches!(ty, BindingType::UniformBufferDynamic),
                min_binding_size: None,
            }
        }
        BindingType::StorageBuffer => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: readonly,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        BindingType::CombinedImageSampler | BindingType::SampledTexture => {
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        }
        BindingType::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        BindingType::StorageTexture => {
            return Err(crate::error::Error::unsupported(
                "storage texture bindings are not implemented",
            ));
        }
        _ => {
            return Err(crate::error::Error::unsupported(
                "binding type not implemented",
            ));
        }
    })
}

pub fn map_shader_stages(stages: ShaderStageFlags) -> wgpu::ShaderStages {
    let mut flags = wgpu::ShaderStages::empty();
    if stages.contains(ShaderStageFlags::VERTEX) {
        flags |= wgpu::ShaderStages::VERTEX;
    }
    if stages.contains(ShaderStageFlags::FRAGMENT) {
        flags |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.contains(ShaderStageFlags::COMPUTE) {
        flags |= wgpu::ShaderStages::COMPUTE;
    }
    flags
}

pub fn layout_has_dynamic_offset(desc: &BindGroupLayoutDescriptor) -> bool {
    desc.entries
        .iter()
        .any(|entry| matches!(entry.ty, BindingType::UniformBufferDynamic))
}
