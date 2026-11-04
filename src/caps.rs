use rotex_core::StaticCaps;
use rotex_types::{BackendCapabilities, CoreCaps, FeatureSet, Limits, Support};

use crate::bridge::WgpuBridge;

pub(crate) const CAPS: CoreCaps = CoreCaps {
    compute: Support::Native,
    storage_buffers: Support::Native,
    push_constants: Support::Absent,
    barriers: Support::Absent,
    queries: Support::Absent,
    copies: Support::Absent,
    readback: Support::Absent,
    render_to_texture: Support::Absent,
    dynamic_offsets: Support::Native,
    explicit_pipelines: Support::Absent,
};

impl StaticCaps for WgpuBridge {
    const CAPS: CoreCaps = CAPS;
}

pub(crate) fn limits_from_wgpu(limits: &wgpu::Limits) -> Limits {
    Limits {
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_bind_groups: limits.max_bind_groups,
        max_push_constant_bytes: limits.max_immediate_size,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size as u32,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u32,
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as u32,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment as u32,
        max_vertex_buffers: limits.max_vertex_buffers,
        max_vertex_attributes: limits.max_vertex_attributes,
        max_color_attachments: limits.max_color_attachments,
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
        max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    }
}

pub(crate) fn capabilities_for(limits: Limits) -> BackendCapabilities {
    BackendCapabilities {
        core: CAPS,
        limits,
        extensions: FeatureSet::empty(),
    }
}

pub(crate) fn unsupported_feature_reporting() -> &'static [&'static str] {
    &[
        "push constants",
        "barriers",
        "queries",
        "copies",
        "readback",
        "render to texture",
        "explicit graphics pipelines",
    ]
}
