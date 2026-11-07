#![allow(dead_code)]
use crate::error::{Error, ErrorKind};
use rotex_types::RhiCommand;

use super::super::WgpuBridge;

pub(super) fn remaining_needs_direct<'a>(commands: impl Iterator<Item = &'a RhiCommand>) -> bool {
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

pub(super) fn occlusion_query_for_pass<'a>(
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

pub(super) fn record_direct_pass<'a>(
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
                    super::super::types::map_index_format(*format),
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
                    let bound = super::super::immediate_uniform::bind_on(
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
