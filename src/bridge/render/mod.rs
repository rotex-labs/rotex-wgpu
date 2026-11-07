#![allow(dead_code)]
use crate::error::{Error, ErrorKind};
use rotex_types::{ColorAttachmentLoad, DepthAttachmentLoad, PassColorTarget, RhiCommand};

use super::WgpuBridge;
use super::surface;
use super::types::DepthTarget;

pub struct ActiveGraphicsPass {
    pub label: String,
}

mod direct;
mod pipeline;

use direct::{occlusion_query_for_pass, record_direct_pass, remaining_needs_direct};
pub(super) use pipeline::create_explicit_pipeline;

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

pub(super) fn to_wgpu_color(clear: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: clear[0] as f64,
        g: clear[1] as f64,
        b: clear[2] as f64,
        a: clear[3] as f64,
    }
}

pub(super) fn clear_color_for_surface(format: wgpu::TextureFormat, clear: [f32; 4]) -> wgpu::Color {
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

pub(super) fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
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
