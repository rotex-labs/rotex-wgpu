//! Copies, readback, split compute dispatch, and query resolve.
//! Barriers are accepted and ignored: wgpu inserts its own hazards.

use std::sync::mpsc;

use crate::error::{Error, ErrorKind};
use rotex_types::{QueryType, TextureId, TextureReadback};

use super::WgpuBridge;
use super::types::bytes_per_pixel;

pub(super) fn read_texture(
    bridge: &mut WgpuBridge,
    id: TextureId,
) -> Result<TextureReadback, Error> {
    let texture = bridge
        .resources
        .textures
        .get(&id)
        .ok_or_else(|| Error::unsupported("texture not found"))?;
    let width = texture.size.0.max(1);
    let height = texture.size.1.max(1);
    let format = texture.source_format;
    let gpu_texture = texture.texture.clone();
    let bytes_per_row = aligned_bytes_per_row(0, width, format)?;
    let padded_len = (bytes_per_row as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| Error::unsupported("texture readback size overflow"))?;
    let buffer = bridge.device.raw.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rotex-wgpu-readback"),
        size: padded_len.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = bridge
        .device
        .raw
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rotex-wgpu-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    bridge.device.queue.submit(Some(encoder.finish()));
    bridge
        .device
        .raw
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| Error::fatal(ErrorKind::Unsupported("texture readback wait failed")))?;
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    bridge
        .device
        .raw
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| Error::fatal(ErrorKind::Unsupported("texture readback map failed")))?;
    receiver
        .recv()
        .map_err(|_| Error::fatal(ErrorKind::Unsupported("texture readback map cancelled")))?
        .map_err(|_| Error::fatal(ErrorKind::Unsupported("texture readback map failed")))?;
    let mapped = buffer.slice(..).get_mapped_range();
    let row_bytes = (width as u64)
        .saturating_mul(bytes_per_pixel(format) as u64)
        .min(bytes_per_row as u64) as usize;
    let mut data = Vec::with_capacity(row_bytes.saturating_mul(height as usize));
    for row in 0..height as usize {
        let start = row * bytes_per_row as usize;
        let end = start + row_bytes;
        if end > mapped.len() {
            break;
        }
        data.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    buffer.unmap();
    Ok(TextureReadback {
        data,
        format,
        width,
        height,
    })
}

pub(super) fn copy_buffer_to_buffer(
    bridge: &mut WgpuBridge,
    src: rotex_types::BufferId,
    src_offset: u64,
    dst: rotex_types::BufferId,
    dst_offset: u64,
    size: u64,
) -> Result<(), Error> {
    let src = bridge
        .resources
        .buffers
        .get(&src)
        .ok_or_else(|| Error::unsupported("copy source buffer not found"))?
        .buffer
        .clone();
    let dst = bridge
        .resources
        .buffers
        .get(&dst)
        .ok_or_else(|| Error::unsupported("copy destination buffer not found"))?
        .buffer
        .clone();
    let mut encoder = take_encoder(bridge);
    encoder.copy_buffer_to_buffer(&src, src_offset, &dst, dst_offset, size);
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn copy_buffer_to_texture(
    bridge: &mut WgpuBridge,
    src: rotex_types::BufferId,
    src_offset: u64,
    dst: TextureId,
    bytes_per_row: u32,
    width: u32,
    height: u32,
) -> Result<(), Error> {
    let format = bridge
        .resources
        .textures
        .get(&dst)
        .ok_or_else(|| Error::unsupported("copy destination texture not found"))?
        .source_format;
    let bytes_per_row = aligned_bytes_per_row(bytes_per_row, width, format)?;
    let src = bridge
        .resources
        .buffers
        .get(&src)
        .ok_or_else(|| Error::unsupported("copy source buffer not found"))?
        .buffer
        .clone();
    let dst = bridge
        .resources
        .textures
        .get(&dst)
        .ok_or_else(|| Error::unsupported("copy destination texture not found"))?
        .texture
        .clone();
    let mut encoder = take_encoder(bridge);
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &src,
            layout: wgpu::TexelCopyBufferLayout {
                offset: src_offset,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn copy_texture_to_buffer(
    bridge: &mut WgpuBridge,
    src: TextureId,
    dst: rotex_types::BufferId,
    dst_offset: u64,
    bytes_per_row: u32,
    width: u32,
    height: u32,
) -> Result<(), Error> {
    let format = bridge
        .resources
        .textures
        .get(&src)
        .ok_or_else(|| Error::unsupported("copy source texture not found"))?
        .source_format;
    let bytes_per_row = aligned_bytes_per_row(bytes_per_row, width, format)?;
    let src = bridge
        .resources
        .textures
        .get(&src)
        .ok_or_else(|| Error::unsupported("copy source texture not found"))?
        .texture
        .clone();
    let dst = bridge
        .resources
        .buffers
        .get(&dst)
        .ok_or_else(|| Error::unsupported("copy destination buffer not found"))?
        .buffer
        .clone();
    let mut encoder = take_encoder(bridge);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &dst,
            layout: wgpu::TexelCopyBufferLayout {
                offset: dst_offset,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn dispatch_compute(bridge: &mut WgpuBridge, groups: [u32; 3]) -> Result<(), Error> {
    let pipeline_id = bridge
        .active_compute_pipeline
        .ok_or_else(|| Error::unsupported("no compute pipeline bound"))?;
    let (pipeline, uniform_immediate) = {
        let resource = bridge
            .resources
            .compute_pipelines
            .get(&pipeline_id)
            .ok_or_else(|| Error::unsupported("compute pipeline not found"))?;
        (resource.pipeline.clone(), resource.uniform_immediate)
    };
    let mut bind_groups = Vec::with_capacity(bridge.compute_binds.len());
    for (slot, id, offsets) in &bridge.compute_binds {
        let group = bridge
            .resources
            .bind_groups
            .get(id)
            .ok_or_else(|| Error::unsupported("compute bind group not found"))?
            .clone();
        bind_groups.push((*slot, group, offsets.clone()));
    }
    let mut encoder = take_encoder(bridge);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rotex-wgpu-compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        for (slot, group, offsets) in &bind_groups {
            pass.set_bind_group(*slot, group, offsets);
        }
        if uniform_immediate {
            if let Some((_, data)) = bridge.compute_immediates.clone() {
                let device = bridge.device.raw.clone();
                let frame_index = bridge.current_frame_index;
                let bound = super::immediate_uniform::bind_on(
                    &mut bridge.immediate_uniform,
                    &device,
                    frame_index,
                    wgpu::ShaderStages::COMPUTE,
                    &data,
                    false,
                )?;
                pass.set_bind_group(1, &bound.group, &[bound.offset]);
            }
        } else if let Some((offset, data)) = &bridge.compute_immediates {
            pass.set_immediates(*offset, data);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn write_timestamp(
    bridge: &mut WgpuBridge,
    query_set: rotex_types::QuerySetId,
    index: u32,
) -> Result<(), Error> {
    let resource = bridge
        .resources
        .query_sets
        .get(&query_set)
        .ok_or_else(|| Error::unsupported("query set not found"))?;
    if resource.ty != QueryType::Timestamp {
        return Err(Error::unsupported(
            "occlusion queries are written inside a render pass",
        ));
    }
    if !bridge
        .device
        .raw
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    {
        return Err(Error::unsupported("timestamp queries are unavailable"));
    }
    let query_set = resource.query_set.clone();
    let mut encoder = take_encoder(bridge);
    encoder.write_timestamp(&query_set, index);
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn resolve_query_set(
    bridge: &mut WgpuBridge,
    query_set: rotex_types::QuerySetId,
    first_query: u32,
    query_count: u32,
    destination: rotex_types::BufferId,
    offset: u64,
) -> Result<(), Error> {
    let query_set = bridge
        .resources
        .query_sets
        .get(&query_set)
        .ok_or_else(|| Error::unsupported("query set not found"))?
        .query_set
        .clone();
    let destination = bridge
        .resources
        .buffers
        .get(&destination)
        .ok_or_else(|| Error::unsupported("query destination buffer not found"))?
        .buffer
        .clone();
    let mut encoder = take_encoder(bridge);
    encoder.resolve_query_set(
        &query_set,
        first_query..(first_query + query_count),
        &destination,
        offset,
    );
    bridge.recording_encoder = Some(encoder);
    Ok(())
}

pub(super) fn immediate_size(
    device: &wgpu::Device,
    layout: &rotex_types::AbstractPipelineLayout,
) -> u32 {
    if !device.features().contains(wgpu::Features::IMMEDIATES) {
        return 0;
    }
    let needed = layout
        .push_constants
        .iter()
        .map(|range| range.offset.saturating_add(range.size))
        .max()
        .unwrap_or(0);
    needed.min(device.limits().max_immediate_size)
}

fn take_encoder(bridge: &mut WgpuBridge) -> wgpu::CommandEncoder {
    bridge.recording_encoder.take().unwrap_or_else(|| {
        bridge
            .device
            .raw
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rotex-wgpu-encoder"),
            })
    })
}

fn aligned_bytes_per_row(
    bytes_per_row: u32,
    width: u32,
    format: rotex_types::TextureFormat,
) -> Result<u32, Error> {
    let packed = if bytes_per_row == 0 {
        width.saturating_mul(bytes_per_pixel(format))
    } else {
        bytes_per_row
    };
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    Ok(packed.div_ceil(alignment) * alignment)
}
