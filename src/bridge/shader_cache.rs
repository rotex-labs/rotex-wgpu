use std::hash::{Hash, Hasher};

use rotex_types::shader::{ShaderPackage, ShaderPayload};

use crate::error::{Error, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShaderCacheKey(u64);

pub(super) fn get_or_create_shader_module(
    bridge: &mut super::WgpuBridge,
    label: &'static str,
    package: &ShaderPackage,
) -> Result<wgpu::ShaderModule, Error> {
    let ShaderPayload::Wgsl(source) = package.variants.wgsl.as_ref().ok_or_else(|| {
        Error::fatal(ErrorKind::Unsupported(
            "wgpu requires a WGSL shader payload",
        ))
    })?
    else {
        return Err(Error::fatal(ErrorKind::Unsupported(
            "wgpu requires a WGSL shader payload",
        )));
    };
    let uniform = super::immediate_uniform::needed(&bridge.device.raw, source);
    // Vertex and fragment packages from one artifact share `source_hash`.
    // Key the module on the stage source so the fragment lookup cannot
    // return the vertex module.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    package.stage.hash(&mut hasher);
    package.entry_point.hash(&mut hasher);
    source.hash(&mut hasher);
    uniform.hash(&mut hasher);
    let key = ShaderCacheKey(hasher.finish());

    if let Some(module) = bridge.shader_module_cache.get(&key) {
        return Ok(module.clone());
    }

    let module = bridge
        .device
        .raw
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(
                super::immediate_uniform::prepare_source(source, uniform).into(),
            ),
        });

    bridge.shader_module_cache.insert(key, module.clone());
    Ok(module)
}
