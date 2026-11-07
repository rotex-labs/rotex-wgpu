#![allow(dead_code)]
use crate::error::{Error, ErrorKind};

use super::super::WgpuBridge;
use super::super::bindings;
use super::super::shader_cache;
use super::super::types::{MaterialPipelineKey, WgpuGraphicsPipelineResource, WgpuVertexLayout};
use rotex_types::resource::MaterialId;

pub(super) fn rhi_pipeline_for_draw<'a>(
    bridge: &'a mut WgpuBridge,
    material_id: MaterialId,
    vertex_layout_id: u64,
    vertex_layout: &WgpuVertexLayout,
    depth_enabled: bool,
) -> Result<&'a wgpu::RenderPipeline, Error> {
    let material = bridge
        .resources
        .materials
        .get(&material_id)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?;
    let key = MaterialPipelineKey {
        material_id,
        vertex_layout_id,
        depth_enabled: depth_enabled && material.enable_depth,
    };

    if !bridge.rhi_pipeline_cache.contains_key(&key) {
        let pipeline = rhi_build_pipeline(bridge, material_id, vertex_layout, key.depth_enabled)?;
        bridge
            .rhi_pipeline_cache
            .insert(key, WgpuGraphicsPipelineResource { pipeline });
    }

    bridge
        .rhi_pipeline_cache
        .get(&key)
        .map(|r| &r.pipeline)
        .ok_or_else(|| {
            Error::fatal(ErrorKind::PipelineCreationFailed(
                "missing_pipeline_after_insert",
            ))
        })
}

pub(super) fn rhi_build_pipeline(
    bridge: &mut WgpuBridge,
    material_id: MaterialId,
    vertex_layout: &WgpuVertexLayout,
    depth_enabled: bool,
) -> Result<wgpu::RenderPipeline, Error> {
    let (vertex_pkg, fragment_pkg) = {
        let material = bridge
            .resources
            .materials
            .get(&material_id)
            .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?;
        (
            material.shaders.vertex.clone(),
            material.shaders.fragment.clone(),
        )
    };

    let vertex_shader =
        shader_cache::get_or_create_shader_module(bridge, "rotex-wgpu-vertex-shader", &vertex_pkg)?;
    let fragment_shader = shader_cache::get_or_create_shader_module(
        bridge,
        "rotex-wgpu-fragment-shader",
        &fragment_pkg,
    )?;

    let swapchain = bridge
        .swapchain
        .as_ref()
        .ok_or_else(super::super::surface_not_attached_error)?;
    let material = bridge
        .resources
        .materials
        .get(&material_id)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("material")))?;

    let wgpu_layouts = bindings::build_set_layouts_from_abstract_layout(
        &bridge.device.raw,
        &material.shaders.layout,
    )?;
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = wgpu_layouts.iter().map(Some).collect();
    let pipeline_layout =
        bridge
            .device
            .raw
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rotex-wgpu-rhi-pipeline-layout"),
                bind_group_layouts: &layout_refs,
                immediate_size: super::super::extended::immediate_size(
                    &bridge.device.raw,
                    &material.shaders.layout,
                ),
            });

    let wgpu_cull_mode = match material.cull_mode {
        rotex_types::CullMode::None => None,
        rotex_types::CullMode::Front => Some(wgpu::Face::Front),
        rotex_types::CullMode::Back => Some(wgpu::Face::Back),
    };

    let vertex_layout_wgpu = vertex_layout.as_wgpu();

    let pipeline = bridge
        .device
        .raw
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rotex-wgpu-rhi-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: Some(material.shaders.vertex.entry_point.as_str()),
                buffers: &[vertex_layout_wgpu],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu_cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: if depth_enabled {
                Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                })
            } else {
                None
            },
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: Some(material.shaders.fragment.entry_point.as_str()),
                targets: &[Some(wgpu::ColorTargetState {
                    format: swapchain.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

    Ok(pipeline)
}

pub(super) fn shader_uses_uniform(bridge: &WgpuBridge, package: &rotex_types::ShaderPackage) -> bool {
    match &package.variants.wgsl {
        Some(rotex_types::shader::ShaderPayload::Wgsl(source)) => {
            super::super::immediate_uniform::needed(&bridge.device.raw, source)
        }
        _ => false,
    }
}

pub(in super::super) fn create_explicit_pipeline(
    bridge: &mut WgpuBridge,
    descriptor: &rotex_types::GraphicsPipelineDescriptor,
) -> Result<(wgpu::RenderPipeline, Option<bool>), Error> {
    let vertex_shader = shader_cache::get_or_create_shader_module(
        bridge,
        "rotex-wgpu-explicit-vertex",
        &descriptor.shaders.vertex,
    )?;
    let fragment_shader = shader_cache::get_or_create_shader_module(
        bridge,
        "rotex-wgpu-explicit-fragment",
        &descriptor.shaders.fragment,
    )?;
    let mut layouts = bindings::build_set_layouts_from_abstract_layout(
        &bridge.device.raw,
        &descriptor.shaders.layout,
    )?;
    let uniform_immediate = shader_uses_uniform(bridge, &descriptor.shaders.vertex)
        || shader_uses_uniform(bridge, &descriptor.shaders.fragment);
    let empty_group0 = if uniform_immediate {
        super::super::immediate_uniform::append_group(
            &bridge.device.raw,
            &mut layouts,
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        )
    } else {
        false
    };
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
    let immediate_size = if uniform_immediate {
        0
    } else {
        super::super::extended::immediate_size(&bridge.device.raw, &descriptor.shaders.layout)
    };
    let pipeline_layout =
        bridge
            .device
            .raw
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rotex-wgpu-explicit-layout"),
                bind_group_layouts: &layout_refs,
                immediate_size,
            });
    let mut vertex_layouts = Vec::with_capacity(descriptor.vertex_buffers.len());
    for layout in &descriptor.vertex_buffers {
        vertex_layouts.push(super::super::resources::translate_vertex_layout(layout)?);
    }
    let vertex_buffers: Vec<wgpu::VertexBufferLayout<'_>> = vertex_layouts
        .iter()
        .map(|layout| layout.as_wgpu())
        .collect();
    let targets: Vec<wgpu::TextureFormat> = if descriptor.targets.is_empty() {
        vec![
            bridge
                .swapchain_format
                .unwrap_or(wgpu::TextureFormat::Rgba8Unorm),
        ]
    } else {
        descriptor
            .targets
            .iter()
            .map(|format| {
                super::super::types::map_texture_format(*format)
                    .ok_or_else(|| Error::unsupported("pipeline target format not implemented"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let blend = match (descriptor.blend.mode, descriptor.blend.enabled) {
        (rotex_types::BlendMode::Additive, _) => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
        (rotex_types::BlendMode::Alpha, _) => Some(wgpu::BlendState::ALPHA_BLENDING),
        (rotex_types::BlendMode::Replace, true) => Some(wgpu::BlendState::ALPHA_BLENDING),
        (rotex_types::BlendMode::Replace, false) => Some(wgpu::BlendState::REPLACE),
    };
    let color_targets: Vec<Option<wgpu::ColorTargetState>> = targets
        .iter()
        .map(|format| {
            Some(wgpu::ColorTargetState {
                format: *format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    let cull_mode = match descriptor.cull_mode {
        rotex_types::CullMode::None => None,
        rotex_types::CullMode::Front => Some(wgpu::Face::Front),
        rotex_types::CullMode::Back => Some(wgpu::Face::Back),
    };
    let topology = match descriptor.topology {
        rotex_types::PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        rotex_types::PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        rotex_types::PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        rotex_types::PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    };
    Ok((
        bridge
            .device
            .raw
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("rotex-wgpu-explicit-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_shader,
                    entry_point: Some(descriptor.shaders.vertex.entry_point.as_str()),
                    buffers: &vertex_buffers,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: if descriptor.depth_enabled {
                    Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth24Plus,
                        depth_write_enabled: Some(descriptor.depth_write),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    })
                } else {
                    None
                },
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_shader,
                    entry_point: Some(descriptor.shaders.fragment.entry_point.as_str()),
                    targets: &color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            }),
        uniform_immediate.then_some(empty_group0),
    ))
}
