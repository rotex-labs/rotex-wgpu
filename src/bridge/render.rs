#![allow(dead_code)]
use crate::error::{Error, ErrorKind};
use rotex_types::{ColorAttachmentLoad, DepthAttachmentLoad, PassColorTarget, RhiCommand};

use super::WgpuBridge;
use super::bindings;
use super::shader_cache;
use super::surface;
use super::types::{
    DepthTarget, MaterialPipelineKey, WgpuGraphicsPipelineResource, WgpuVertexLayout,
};
use rotex_types::resource::MaterialId;

pub struct ActiveGraphicsPass {
    pub label: String,
}

fn ensure_depth_target(bridge: &mut WgpuBridge) -> Result<&wgpu::TextureView, Error> {
    let swapchain = bridge
        .swapchain
        .as_ref()
        .ok_or_else(super::surface_not_attached_error)?;
    let current_size = (
        swapchain.config.width.max(1),
        swapchain.config.height.max(1),
    );
    let needs_recreate = bridge
        .depth_target
        .as_ref()
        .map(|target| target.size != current_size)
        .unwrap_or(true);

    if needs_recreate {
        let texture = bridge.device.raw.create_texture(&wgpu::TextureDescriptor {
            label: Some("rotex-wgpu-depth"),
            size: wgpu::Extent3d {
                width: current_size.0,
                height: current_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        bridge.depth_target = Some(DepthTarget {
            _texture: texture,
            view,
            size: current_size,
        });
    }

    bridge
        .depth_target
        .as_ref()
        .map(|target| &target.view)
        .ok_or_else(|| {
            Error::fatal(ErrorKind::Unsupported(
                "depth target missing after creation",
            ))
        })
}

fn to_wgpu_color(clear: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: clear[0] as f64,
        g: clear[1] as f64,
        b: clear[2] as f64,
        a: clear[3] as f64,
    }
}

fn clear_color_for_surface(format: wgpu::TextureFormat, clear: [f32; 4]) -> wgpu::Color {
    let converted = if format.is_srgb() {
        clear
    } else {
        [
            linear_to_srgb(clear[0]),
            linear_to_srgb(clear[1]),
            linear_to_srgb(clear[2]),
            clear[3],
        ]
    };
    to_wgpu_color(converted)
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn rhi_pipeline_for_draw<'a>(
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

fn rhi_build_pipeline(
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
        .ok_or_else(super::surface_not_attached_error)?;
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
                immediate_size: super::extended::immediate_size(
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

pub(super) fn execute(bridge: &mut WgpuBridge, commands: &[RhiCommand]) -> Result<(), Error> {
    let mut cmd_iter = commands.iter();

    while let Some(command) = cmd_iter.next() {
        match command {
            RhiCommand::BeginFrame { frame_index: _ } => {
                bridge.current_frame_index += 1;
                let encoder =
                    bridge
                        .device
                        .raw
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("rotex-wgpu-encoder"),
                        });
                bridge.recording_encoder = Some(encoder);
                bridge.active_pass = None;
                if let Some(uniform) = bridge.immediate_uniform.as_mut() {
                    uniform.begin_frame(bridge.current_frame_index);
                }
            }
            RhiCommand::AcquirePresentationTarget => {
                let surface_texture = surface::acquire_surface_texture(bridge)?;
                bridge.surface_texture = surface_texture;
                if bridge.surface_texture.is_some() {
                    let format = bridge
                        .swapchain
                        .as_ref()
                        .ok_or_else(super::surface_not_attached_error)?
                        .config
                        .format;
                    bridge.swapchain_format = Some(format);
                }
            }
            RhiCommand::WriteBuffer {
                buffer,
                offset,
                data,
            } => {
                let buf_res = bridge
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| Error::fatal(ErrorKind::Unsupported("buffer not found")))?;
                bridge
                    .device
                    .queue
                    .write_buffer(&buf_res.buffer, *offset, data);
            }
            RhiCommand::BeginRenderPass { pass } => {
                let mut encoder = bridge.recording_encoder.take().ok_or_else(|| {
                    Error::fatal(ErrorKind::Unsupported(
                        "no command encoder for render pass (call BeginFrame first)",
                    ))
                })?;

                let (color_view, surface_format) = match &pass.color_target {
                    PassColorTarget::Presentation => {
                        let surface_texture = bridge.surface_texture.as_ref().ok_or_else(|| {
                            Error::fatal(ErrorKind::Unsupported(
                                "no surface texture (call AcquirePresentationTarget first)",
                            ))
                        })?;
                        let format = bridge
                            .swapchain
                            .as_ref()
                            .ok_or_else(super::surface_not_attached_error)?
                            .config
                            .format;
                        (
                            surface_texture
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                            format,
                        )
                    }
                    PassColorTarget::Texture(id) => {
                        let texture = bridge.resources.textures.get(id).ok_or_else(|| {
                            Error::fatal(ErrorKind::Unsupported("render target texture not found"))
                        })?;
                        (texture.default_view.clone(), texture.format)
                    }
                };

                let load_color = match pass.color_load {
                    ColorAttachmentLoad::Clear => wgpu::LoadOp::Clear(clear_color_for_surface(
                        surface_format,
                        pass.clear_color,
                    )),
                    ColorAttachmentLoad::Load => wgpu::LoadOp::Load,
                };

                let depth_view_owned = if pass.uses_depth_attachment() {
                    Some(ensure_depth_target(bridge)?.clone())
                } else {
                    None
                };
                let depth_load_op = match pass.depth_load {
                    DepthAttachmentLoad::Clear => wgpu::LoadOp::Clear(pass.clear_depth),
                    DepthAttachmentLoad::Load => wgpu::LoadOp::Load,
                    DepthAttachmentLoad::None => wgpu::LoadOp::Clear(pass.clear_depth),
                };
                let depth_attachment = depth_view_owned.as_ref().map(|depth_view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: depth_load_op,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }
                });

                let occlusion_query = occlusion_query_for_pass(bridge, cmd_iter.clone());
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(pass.name.as_str()),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: load_color,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: depth_attachment,
                    occlusion_query_set: occlusion_query.as_ref(),
                    timestamp_writes: None,
                    multiview_mask: None,
                });

                bridge.active_pass = Some(ActiveGraphicsPass {
                    label: pass.name.clone(),
                });

                if remaining_needs_direct(cmd_iter.clone()) {
                    record_direct_pass(&mut render_pass, bridge, &mut cmd_iter)?;
                    bridge.render_bundle = None;
                } else if let Some(ref bundle) = bridge.render_bundle {
                    render_pass.execute_bundles(std::slice::from_ref(bundle));
                    loop {
                        match cmd_iter.next() {
                            Some(RhiCommand::EndRenderPass) => break,
                            Some(_) => continue,
                            None => {
                                return Err(Error::fatal(ErrorKind::Unsupported(
                                    "missing EndRenderPass command",
                                )));
                            }
                        }
                    }
                } else {
                    let depth_stencil = if pass.uses_depth_attachment() {
                        Some(wgpu::RenderBundleDepthStencil {
                            format: wgpu::TextureFormat::Depth24Plus,
                            depth_read_only: false,
                            stencil_read_only: true,
                        })
                    } else {
                        None
                    };
                    let device = bridge.device.raw.clone();
                    let mut bundle_encoder =
                        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                            label: Some("rotex-wgpu-bundle"),
                            color_formats: &[Some(surface_format)],
                            depth_stencil,
                            sample_count: 1,
                            multiview: None,
                        });

                    struct DrawOp {
                        pipeline: *const wgpu::RenderPipeline,
                        bind_groups: Vec<(u32, *const wgpu::BindGroup)>,
                        vertex_buffer: *const wgpu::Buffer,
                        index_buffer: *const wgpu::Buffer,
                        index_format: wgpu::IndexFormat,
                        index_count: u32,
                        instance_count: u32,
                        first_index: u32,
                        vertex_offset: i32,
                        first_instance: u32,
                        dynamic_offsets: Vec<u32>,
                    }
                    let mut ops: Vec<DrawOp> = Vec::new();
                    loop {
                        match cmd_iter.next() {
                            Some(RhiCommand::EndRenderPass) => break,
                            Some(RhiCommand::BindGroups {
                                first_slot,
                                bind_groups,
                                dynamic_offsets,
                            }) => {
                                for (i, bg_id) in bind_groups.iter().enumerate() {
                                    let bg = bridge.resources.bind_groups.get(bg_id).ok_or_else(
                                        || {
                                            Error::fatal(ErrorKind::Unsupported(
                                                "bind group not found",
                                            ))
                                        },
                                    )?;
                                    if let Some(op) = ops.last_mut() {
                                        op.bind_groups.push((
                                            *first_slot + i as u32,
                                            bg as *const wgpu::BindGroup,
                                        ));
                                        if i == 0 {
                                            op.dynamic_offsets = dynamic_offsets.clone();
                                        }
                                    }
                                }
                            }
                            Some(RhiCommand::SetVertexBuffers {
                                mesh,
                                first_binding: _,
                            }) => {
                                let mesh_id = *mesh;
                                let mesh_res =
                                    bridge.resources.meshes.get(&mesh_id).ok_or_else(|| {
                                        Error::recoverable(ErrorKind::ResourceNotFound("mesh"))
                                    })?;
                                if let Some(op) = ops.last_mut() {
                                    op.vertex_buffer =
                                        &mesh_res.vertex_buffer as *const wgpu::Buffer;
                                }
                            }
                            Some(RhiCommand::SetIndexBuffer { mesh }) => {
                                let mesh_id = *mesh;
                                let mesh_res =
                                    bridge.resources.meshes.get(&mesh_id).ok_or_else(|| {
                                        Error::recoverable(ErrorKind::ResourceNotFound("mesh"))
                                    })?;
                                if let Some(op) = ops.last_mut() {
                                    op.index_buffer = &mesh_res.index_buffer as *const wgpu::Buffer;
                                    op.index_format = mesh_res.index_format;
                                }
                            }
                            Some(RhiCommand::DrawIndexed {
                                index_count,
                                instance_count,
                                first_index,
                                vertex_offset,
                                first_instance,
                            }) => {
                                if let Some(op) = ops.last_mut() {
                                    op.index_count = *index_count;
                                    op.instance_count = *instance_count;
                                    op.first_index = *first_index;
                                    op.vertex_offset = *vertex_offset;
                                    op.first_instance = *first_instance;
                                }
                            }
                            Some(RhiCommand::PushConstants { .. }) => {
                                return Err(Error::unsupported(
                                    "push constants are not implemented",
                                ));
                            }
                            Some(_) => {
                                return Err(Error::unsupported(
                                    "command is not implemented inside a render pass",
                                ));
                            }
                            None => {
                                return Err(Error::fatal(ErrorKind::Unsupported(
                                    "missing EndRenderPass command",
                                )));
                            }
                        }
                    }

                    for op in &ops {
                        // Safety: all pointers reference data in bridge's resource caches,
                        // which outlive the bundle encoder and this function.
                        let p = unsafe { &*op.pipeline };
                        bundle_encoder.set_pipeline(p);
                        for (group_index, (set, bg_ptr)) in op.bind_groups.iter().enumerate() {
                            let bg = unsafe { &**bg_ptr };
                            let offsets = if group_index == 0 {
                                op.dynamic_offsets.as_slice()
                            } else {
                                &[]
                            };
                            bundle_encoder.set_bind_group(*set, bg, offsets);
                        }
                        if !op.vertex_buffer.is_null() {
                            let vb = unsafe { &*op.vertex_buffer };
                            bundle_encoder.set_vertex_buffer(0, vb.slice(..));
                        }
                        if !op.index_buffer.is_null() {
                            let ib = unsafe { &*op.index_buffer };
                            bundle_encoder.set_index_buffer(ib.slice(..), op.index_format);
                        }
                        bundle_encoder.draw_indexed(
                            op.first_index..(op.first_index + op.index_count),
                            op.vertex_offset,
                            op.first_instance..(op.first_instance + op.instance_count),
                        );
                    }

                    let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor {
                        label: Some("rotex-wgpu-bundle"),
                    });
                    bridge.render_bundle = Some(bundle);
                    let cached = bridge.render_bundle.as_ref().ok_or_else(|| {
                        Error::fatal(ErrorKind::Unsupported("render bundle missing after finish"))
                    })?;
                    render_pass.execute_bundles(std::slice::from_ref(cached));
                }

                drop(render_pass);
                bridge.active_pass = None;
                bridge.recording_encoder = Some(encoder);
            }
            RhiCommand::EndRenderPass => {}
            RhiCommand::BeginComputePass { .. } => {
                bridge.compute_pass_open = true;
                bridge.compute_binds.clear();
                bridge.compute_immediates = None;
            }
            RhiCommand::EndComputePass => {
                bridge.compute_pass_open = false;
                bridge.active_compute_pipeline = None;
                bridge.compute_binds.clear();
                bridge.compute_immediates = None;
            }
            RhiCommand::PushConstants { offset, data, .. } if bridge.compute_pass_open => {
                bridge.compute_immediates = Some((*offset, data.clone()));
            }
            RhiCommand::SetComputePipeline { pipeline } => {
                bridge.active_compute_pipeline = Some(*pipeline);
            }
            RhiCommand::Dispatch { workgroup_count } => {
                super::extended::dispatch_compute(bridge, *workgroup_count)?;
            }
            RhiCommand::BindGroups {
                first_slot,
                bind_groups,
                dynamic_offsets,
            } if bridge.compute_pass_open => {
                for (index, group) in bind_groups.iter().enumerate() {
                    let offsets = if index == 0 {
                        dynamic_offsets.clone()
                    } else {
                        Vec::new()
                    };
                    bridge
                        .compute_binds
                        .push((*first_slot + index as u32, *group, offsets));
                }
            }
            RhiCommand::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                super::extended::copy_buffer_to_buffer(
                    bridge,
                    *src,
                    *src_offset,
                    *dst,
                    *dst_offset,
                    *size,
                )?;
            }
            RhiCommand::CopyBufferToTexture {
                src,
                src_offset,
                dst,
                bytes_per_row,
                width,
                height,
            } => {
                super::extended::copy_buffer_to_texture(
                    bridge,
                    *src,
                    *src_offset,
                    *dst,
                    *bytes_per_row,
                    *width,
                    *height,
                )?;
            }
            RhiCommand::CopyTextureToBuffer {
                src,
                dst,
                dst_offset,
                bytes_per_row,
                width,
                height,
            } => {
                super::extended::copy_texture_to_buffer(
                    bridge,
                    *src,
                    *dst,
                    *dst_offset,
                    *bytes_per_row,
                    *width,
                    *height,
                )?;
            }
            RhiCommand::Barrier { .. } => {}
            RhiCommand::WriteTimestamp { query_set, index } => {
                super::extended::write_timestamp(bridge, *query_set, *index)?;
            }
            RhiCommand::ResolveQuerySet {
                query_set,
                first_query,
                query_count,
                destination,
                offset,
            } => {
                super::extended::resolve_query_set(
                    bridge,
                    *query_set,
                    *first_query,
                    *query_count,
                    *destination,
                    *offset,
                )?;
            }
            RhiCommand::SubmitFrame { present } => {
                if let Some(encoder) = bridge.recording_encoder.take() {
                    if let Some(uniform) = bridge.immediate_uniform.as_ref() {
                        uniform.flush(&bridge.device.queue);
                    }
                    bridge.device.queue.submit(Some(encoder.finish()));
                }
                if *present && let Some(surface_texture) = bridge.surface_texture.take() {
                    surface_texture.present();
                }
            }
            _ => return Err(Error::unsupported("command not implemented")),
        }
    }

    Ok(())
}

fn remaining_needs_direct<'a>(commands: impl Iterator<Item = &'a RhiCommand>) -> bool {
    for command in commands {
        match command {
            RhiCommand::EndRenderPass => return false,
            RhiCommand::SetViewport { .. }
            | RhiCommand::SetScissor { .. }
            | RhiCommand::Draw { .. }
            | RhiCommand::Extension { .. }
            | RhiCommand::SetGraphicsPipeline { .. }
            | RhiCommand::BindVertexBuffer { .. }
            | RhiCommand::BindIndexBuffer { .. }
            | RhiCommand::PushConstants { .. }
            | RhiCommand::WriteTimestamp { .. } => return true,
            _ => {}
        }
    }
    false
}

fn occlusion_query_for_pass<'a>(
    bridge: &WgpuBridge,
    commands: impl Iterator<Item = &'a RhiCommand>,
) -> Option<wgpu::QuerySet> {
    for command in commands {
        match command {
            RhiCommand::EndRenderPass => return None,
            RhiCommand::WriteTimestamp { query_set, .. } => {
                if let Some(resource) = bridge.resources.query_sets.get(query_set)
                    && resource.ty == rotex_types::QueryType::Occlusion
                {
                    return Some(resource.query_set.clone());
                }
            }
            _ => {}
        }
    }
    None
}

fn record_direct_pass<'a>(
    pass: &mut wgpu::RenderPass<'_>,
    bridge: &mut WgpuBridge,
    commands: &mut impl Iterator<Item = &'a RhiCommand>,
) -> Result<(), Error> {
    let mut uniform_graphics: Option<bool> = None;
    loop {
        match commands.next() {
            Some(RhiCommand::EndRenderPass) => return Ok(()),
            Some(RhiCommand::BindGroups {
                first_slot,
                bind_groups,
                dynamic_offsets,
            }) => {
                for (index, group_id) in bind_groups.iter().enumerate() {
                    let group = bridge
                        .resources
                        .bind_groups
                        .get(group_id)
                        .ok_or_else(|| {
                            Error::fatal(ErrorKind::Unsupported("bind group not found"))
                        })?
                        .clone();
                    let offsets = if index == 0 {
                        dynamic_offsets.as_slice()
                    } else {
                        &[]
                    };
                    pass.set_bind_group(*first_slot + index as u32, &group, offsets);
                }
            }
            Some(RhiCommand::SetVertexBuffers {
                mesh,
                first_binding,
            }) => {
                let mesh_res = bridge
                    .resources
                    .meshes
                    .get(mesh)
                    .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("mesh")))?;
                let buffer = mesh_res.vertex_buffer.clone();
                pass.set_vertex_buffer(*first_binding, buffer.slice(..));
            }
            Some(RhiCommand::SetIndexBuffer { mesh }) => {
                let mesh_res = bridge
                    .resources
                    .meshes
                    .get(mesh)
                    .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("mesh")))?;
                let buffer = mesh_res.index_buffer.clone();
                let format = mesh_res.index_format;
                pass.set_index_buffer(buffer.slice(..), format);
            }
            Some(RhiCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            }) => {
                pass.draw_indexed(
                    *first_index..(*first_index + *index_count),
                    *vertex_offset,
                    *first_instance..(*first_instance + *instance_count),
                );
            }
            Some(RhiCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            }) => {
                pass.draw(
                    *first_vertex..(*first_vertex + *vertex_count),
                    *first_instance..(*first_instance + *instance_count),
                );
            }
            Some(RhiCommand::Extension {
                feature,
                opcode,
                payload,
            }) => {
                use rotex_core::Feature;
                use rotex_types::IndirectDraw;
                if *feature != IndirectDraw::ID || *opcode != 1 || payload.len() < 16 {
                    return Err(Error::unsupported("extension command is not implemented"));
                }
                let mut buffer_bytes = [0u8; 8];
                let mut offset_bytes = [0u8; 8];
                buffer_bytes.copy_from_slice(&payload[0..8]);
                offset_bytes.copy_from_slice(&payload[8..16]);
                let buffer_id = rotex_types::BufferId(u64::from_le_bytes(buffer_bytes));
                let offset = u64::from_le_bytes(offset_bytes);
                let buffer = bridge
                    .resources
                    .buffers
                    .get(&buffer_id)
                    .ok_or_else(|| Error::unsupported("indirect buffer not found"))?
                    .buffer
                    .clone();
                pass.draw_indexed_indirect(&buffer, offset);
            }
            Some(RhiCommand::SetViewport {
                x,
                y,
                width,
                height,
                min_depth,
                max_depth,
            }) => {
                pass.set_viewport(*x, *y, *width, *height, *min_depth, *max_depth);
            }
            Some(RhiCommand::SetScissor {
                x,
                y,
                width,
                height,
            }) => {
                pass.set_scissor_rect(*x, *y, *width, *height);
            }
            Some(RhiCommand::SetGraphicsPipeline { pipeline }) => {
                uniform_graphics = bridge.uniform_graphics.get(pipeline).copied();
                let pipeline = bridge
                    .resources
                    .graphics_pipelines
                    .get(pipeline)
                    .ok_or_else(|| Error::unsupported("graphics pipeline not found"))?
                    .clone();
                pass.set_pipeline(&pipeline);
            }
            Some(RhiCommand::BindVertexBuffer {
                buffer,
                binding,
                offset,
            }) => {
                let buffer = bridge
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| Error::unsupported("vertex buffer not found"))?
                    .buffer
                    .clone();
                pass.set_vertex_buffer(*binding, buffer.slice(*offset..));
            }
            Some(RhiCommand::BindIndexBuffer {
                buffer,
                offset,
                format,
            }) => {
                let buffer = bridge
                    .resources
                    .buffers
                    .get(buffer)
                    .ok_or_else(|| Error::unsupported("index buffer not found"))?
                    .buffer
                    .clone();
                pass.set_index_buffer(
                    buffer.slice(*offset..),
                    super::types::map_index_format(*format),
                );
            }
            Some(RhiCommand::PushConstants { offset, data, .. }) => {
                if bridge
                    .device
                    .raw
                    .features()
                    .contains(wgpu::Features::IMMEDIATES)
                {
                    pass.set_immediates(*offset, data);
                } else if let Some(empty_group0) = uniform_graphics {
                    let device = bridge.device.raw.clone();
                    let frame_index = bridge.current_frame_index;
                    let bound = super::immediate_uniform::bind_on(
                        &mut bridge.immediate_uniform,
                        &device,
                        frame_index,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        data,
                        empty_group0,
                    )?;
                    if let Some(empty) = bound.empty_group0 {
                        pass.set_bind_group(0, &empty, &[]);
                    }
                    pass.set_bind_group(1, &bound.group, &[bound.offset]);
                } else {
                    return Err(Error::unsupported("push constants are unavailable"));
                }
            }
            Some(RhiCommand::WriteTimestamp { query_set, index }) => {
                let resource = bridge
                    .resources
                    .query_sets
                    .get(query_set)
                    .ok_or_else(|| Error::unsupported("query set not found"))?;
                let kind = resource.ty;
                let query_set = resource.query_set.clone();
                match kind {
                    rotex_types::QueryType::Timestamp => {
                        if !bridge
                            .device
                            .raw
                            .features()
                            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
                        {
                            return Err(Error::unsupported("timestamp queries are unavailable"));
                        }
                        pass.write_timestamp(&query_set, *index);
                    }
                    rotex_types::QueryType::Occlusion => {
                        pass.begin_occlusion_query(*index);
                        pass.end_occlusion_query();
                    }
                }
            }
            Some(_) => {
                return Err(Error::unsupported(
                    "command is not implemented inside a render pass",
                ));
            }
            None => {
                return Err(Error::fatal(ErrorKind::Unsupported(
                    "missing EndRenderPass command",
                )));
            }
        }
    }
}

fn shader_uses_uniform(bridge: &WgpuBridge, package: &rotex_types::ShaderPackage) -> bool {
    match &package.variants.wgsl {
        Some(rotex_types::shader::ShaderPayload::Wgsl(source)) => {
            super::immediate_uniform::needed(&bridge.device.raw, source)
        }
        _ => false,
    }
}

pub(super) fn create_explicit_pipeline(
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
        super::immediate_uniform::append_group(
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
        super::extended::immediate_size(&bridge.device.raw, &descriptor.shaders.layout)
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
        vertex_layouts.push(super::resources::translate_vertex_layout(layout)?);
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
                super::types::map_texture_format(*format)
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
