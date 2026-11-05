//! Push constants for backends that cannot enable immediates.
//!
//! wgpu 29's browser backend reports an immediate-data limit of zero and
//! panics in `set_immediates`. Slang still emits `var<immediate>`. Those
//! variables are rewritten to a group-1 uniform, and each push is stored in
//! its own aligned slot of one buffer.

use std::collections::HashMap;

use crate::error::{Error, ErrorKind};

const FRAMES: u32 = 3;
const SLOTS_PER_FRAME: u32 = 8;

pub(crate) struct ImmediateUniform {
    buffer: wgpu::Buffer,
    stride: u64,
    cpu: Vec<u8>,
    frame_index: u32,
    cursor: u32,
    groups: HashMap<u32, wgpu::BindGroup>,
    empty_group: wgpu::BindGroup,
}

pub(super) struct UniformBind {
    pub group: wgpu::BindGroup,
    pub offset: u32,
    pub empty_group0: Option<wgpu::BindGroup>,
}

pub(super) fn needed(device: &wgpu::Device, wgsl: &str) -> bool {
    !device.features().contains(wgpu::Features::IMMEDIATES) && wgsl.contains("var<immediate>")
}

pub(super) fn prepare_source(source: &str, uniform: bool) -> String {
    let stripped = source
        .lines()
        .filter(|line| line.trim() != "requires immediate_address_space;")
        .collect::<Vec<_>>()
        .join("\n");
    if !uniform {
        return stripped;
    }
    let Some(type_name) = immediate_type_name(&stripped) else {
        return stripped;
    };
    let padded = pad_struct(&stripped, &type_name);
    padded.replace("var<immediate>", "@group(1) @binding(0) var<uniform>")
}

/// Adds the uniform at group 1. Returns whether group 0 was an empty layout
/// inserted so the group index stays 1.
pub(super) fn bind_on(
    slot: &mut Option<ImmediateUniform>,
    device: &wgpu::Device,
    frame_index: u32,
    stages: wgpu::ShaderStages,
    data: &[u8],
    empty_group0: bool,
) -> Result<UniformBind, Error> {
    if slot.is_none() {
        *slot = Some(ImmediateUniform::new(device, frame_index)?);
    }
    let uniform = slot.as_mut().ok_or_else(|| {
        Error::fatal(ErrorKind::Unsupported(
            "immediate uniform buffer is missing",
        ))
    })?;
    uniform.bind(device, stages, data, empty_group0)
}

pub(super) fn append_group(
    device: &wgpu::Device,
    layouts: &mut Vec<wgpu::BindGroupLayout>,
    stages: wgpu::ShaderStages,
) -> bool {
    let inserted_empty = layouts.is_empty();
    if inserted_empty {
        layouts.push(empty_layout(device));
    }
    if layouts.len() == 1 {
        layouts.push(uniform_layout(device, stages));
    }
    inserted_empty
}

impl ImmediateUniform {
    pub(super) fn new(device: &wgpu::Device, frame_index: u32) -> Result<Self, Error> {
        let stride = u64::from(device.limits().min_uniform_buffer_offset_alignment).max(256);
        let slots = FRAMES * SLOTS_PER_FRAME;
        let size = stride * u64::from(slots);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rotex-wgpu-immediate-uniform"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let empty_layout = empty_layout(device);
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rotex-wgpu-empty-bind-group"),
            layout: &empty_layout,
            entries: &[],
        });
        Ok(Self {
            buffer,
            stride,
            cpu: vec![0; size as usize],
            frame_index,
            cursor: 0,
            groups: HashMap::new(),
            empty_group,
        })
    }

    pub(super) fn begin_frame(&mut self, frame_index: u32) {
        self.frame_index = frame_index;
        self.cursor = 0;
    }

    pub(super) fn bind(
        &mut self,
        device: &wgpu::Device,
        stages: wgpu::ShaderStages,
        data: &[u8],
        empty_group0: bool,
    ) -> Result<UniformBind, Error> {
        if self.cursor >= SLOTS_PER_FRAME {
            return Err(Error::fatal(ErrorKind::Unsupported(
                "too many push constants in one frame",
            )));
        }
        let slot = (self.frame_index % FRAMES) * SLOTS_PER_FRAME + self.cursor;
        self.cursor += 1;
        let start = slot as usize * self.stride as usize;
        let end = start + self.stride as usize;
        self.cpu[start..end].fill(0);
        let copy = data.len().min(self.stride as usize);
        self.cpu[start..start + copy].copy_from_slice(&data[..copy]);
        let group = self.group(device, stages)?;
        Ok(UniformBind {
            group,
            offset: u32::try_from(slot as u64 * self.stride).unwrap_or(0),
            empty_group0: empty_group0.then(|| self.empty_group.clone()),
        })
    }

    pub(super) fn flush(&self, queue: &wgpu::Queue) {
        if self.cursor == 0 {
            return;
        }
        let start = u64::from((self.frame_index % FRAMES) * SLOTS_PER_FRAME) * self.stride;
        let bytes = (SLOTS_PER_FRAME as usize) * self.stride as usize;
        let src = start as usize;
        queue.write_buffer(&self.buffer, start, &self.cpu[src..src + bytes]);
    }

    fn group(
        &mut self,
        device: &wgpu::Device,
        stages: wgpu::ShaderStages,
    ) -> Result<wgpu::BindGroup, Error> {
        if let Some(group) = self.groups.get(&stages.bits()) {
            return Ok(group.clone());
        }
        let layout = uniform_layout(device, stages);
        let size = std::num::NonZeroU64::new(self.stride)
            .ok_or_else(|| Error::fatal(ErrorKind::Unsupported("uniform stride is zero")))?;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rotex-wgpu-immediate-bind-group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.buffer,
                    offset: 0,
                    size: Some(size),
                }),
            }],
        });
        self.groups.insert(stages.bits(), group.clone());
        Ok(group)
    }
}

fn immediate_type_name(source: &str) -> Option<String> {
    let marker = "var<immediate>";
    let start = source.find(marker)?;
    let after = source[start + marker.len()..].trim_start();
    let ty = after.split(':').nth(1)?.trim().trim_end_matches(';').trim();
    if ty.is_empty() {
        None
    } else {
        Some(ty.to_string())
    }
}

fn pad_struct(source: &str, type_name: &str) -> String {
    let header = format!("struct {type_name}");
    let Some(at) = source.find(&header) else {
        return source.to_string();
    };
    let Some(open_rel) = source[at..].find('{') else {
        return source.to_string();
    };
    let open = at + open_rel;
    let Some(close_rel) = source[open..].find('}') else {
        return source.to_string();
    };
    let close = open + close_rel;
    let mut output = String::new();
    output.push_str(&source[..close]);
    output.push_str("\n    @align(16) _rotex_uniform_pad : u32,\n");
    output.push_str(&source[close..]);
    output
}

fn empty_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rotex-wgpu-empty-bind-group-layout"),
        entries: &[],
    })
}

fn uniform_layout(device: &wgpu::Device, stages: wgpu::ShaderStages) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rotex-wgpu-immediate-uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}
