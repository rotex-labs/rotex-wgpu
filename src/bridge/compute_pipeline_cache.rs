use std::collections::BTreeMap;

use crate::error::{Error, ErrorKind};
use rotex_types::resource::{ComputeBindingLayout, ComputePipelineDescriptor};

use super::WgpuBridge;
use super::types::WgpuComputePipelineResource;

pub(super) fn create_compute_pipeline(
    bridge: &WgpuBridge,
    descriptor: &ComputePipelineDescriptor,
) -> Result<WgpuComputePipelineResource, Error> {
    if descriptor.shader_spv.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "compute_shader_bytes_missing",
        )));
    }
    if !descriptor.shader_spv.len().is_multiple_of(4) {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "compute_shader_bytes_not_word_aligned",
        )));
    }
    if descriptor.entry_point.is_empty() {
        return Err(Error::recoverable(ErrorKind::InvalidDescriptor(
            "compute_shader_entry_missing",
        )));
    }

    let bind_group_layouts = build_bind_group_layouts(&bridge.device.raw, &descriptor.bindings)?;
    let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
        bind_group_layouts.iter().map(Some).collect();
    let pipeline_layout =
        bridge
            .device
            .raw
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rotex-wgpu-compute-pipeline-layout"),
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });
    let shader = bridge
        .device
        .raw
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rotex-wgpu-compute-shader"),
            source: wgpu::util::make_spirv(&descriptor.shader_spv),
        });
    let pipeline = bridge
        .device
        .raw
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rotex-wgpu-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(descriptor.entry_point.as_str()),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    Ok(WgpuComputePipelineResource {
        descriptor: descriptor.clone(),
        pipeline,
        bind_group_layouts,
    })
}

pub(super) fn record_compute_dispatch(
    bridge: &WgpuBridge,
    encoder: &mut wgpu::CommandEncoder,
    pass: &rotex_types::ComputePassDescriptor,
) -> Result<(), Error> {
    let pipeline_resource = bridge
        .resources
        .compute_pipelines
        .get(&pass.pipeline)
        .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("compute_pipeline")))?;

    let intents_by_set: BTreeMap<u32, Vec<&rotex_types::BufferUsageIntent>> = pass
        .buffer_intents
        .iter()
        .fold(BTreeMap::new(), |mut map, intent| {
            map.entry(intent.set).or_default().push(intent);
            map
        });

    let mut bind_groups = Vec::new();
    for (set_index, intents) in intents_by_set {
        let layout = pipeline_resource
            .bind_group_layouts
            .get(set_index as usize)
            .ok_or_else(|| {
                Error::recoverable(ErrorKind::InvalidDescriptor(
                    "compute_bind_group_layout_missing",
                ))
            })?;
        let mut entries = Vec::with_capacity(intents.len());
        for intent in intents {
            let buffer = bridge
                .resources
                .buffers
                .get(&intent.buffer)
                .ok_or_else(|| Error::recoverable(ErrorKind::ResourceNotFound("buffer")))?;
            let size = if intent.size == 0 {
                buffer.size
            } else {
                intent.size
            };
            entries.push(wgpu::BindGroupEntry {
                binding: intent.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer.buffer,
                    offset: intent.offset,
                    size: std::num::NonZeroU64::new(size),
                }),
            });
        }
        let bind_group = bridge
            .device
            .raw
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rotex-wgpu-compute-bind-group"),
                layout,
                entries: &entries,
            });
        bind_groups.push((set_index, bind_group));
    }

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("rotex-wgpu-compute-pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline_resource.pipeline);
    for (set_index, bind_group) in bind_groups {
        compute_pass.set_bind_group(set_index, &bind_group, &[]);
    }
    compute_pass.dispatch_workgroups(
        pass.workgroup_count[0],
        pass.workgroup_count[1],
        pass.workgroup_count[2],
    );
    Ok(())
}

fn build_bind_group_layouts(
    device: &wgpu::Device,
    bindings: &[ComputeBindingLayout],
) -> Result<Vec<wgpu::BindGroupLayout>, Error> {
    if bindings.is_empty() {
        return Ok(Vec::new());
    }
    let max_set = bindings
        .iter()
        .map(|binding| binding.set)
        .max()
        .unwrap_or(0);
    let mut layouts = Vec::with_capacity(max_set as usize + 1);
    for set in 0..=max_set {
        let set_bindings: Vec<_> = bindings
            .iter()
            .filter(|binding| binding.set == set)
            .collect();
        let entries: Vec<_> = set_bindings
            .iter()
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding.readonly {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        layouts.push(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rotex-wgpu-compute-bind-group-layout"),
                entries: &entries,
            }),
        );
    }
    Ok(layouts)
}
