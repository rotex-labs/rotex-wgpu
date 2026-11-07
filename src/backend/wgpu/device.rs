use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use wgpu::{
    DeviceDescriptor as WgpuDeviceDescriptor, InstanceDescriptor as WgpuInstanceDescriptor,
    SurfaceTargetUnsafe,
};

use super::surface::WgpuSurface;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    AdapterCapabilities, AdapterId, AdapterInfo, AdapterQueueSupport, DebugMode, DeviceDescriptor,
    FeatureSet, GpuApiBackend, HardwareFamily, PlatformDescriptor,
};

use crate::caps::{self, CAPS};

/// wgpu instance used to create surfaces.
pub struct WgpuInstance {
    pub(crate) raw: wgpu::Instance,
}

/// Adapter, device, and queue used by [`crate::WgpuBridge`].
pub struct WgpuDevice {
    /// Adapter this device was opened from.
    pub adapter: wgpu::Adapter,
    /// wgpu device.
    pub raw: wgpu::Device,
    /// Queue used for submission.
    pub queue: wgpu::Queue,
}

impl WgpuInstance {
    pub(crate) fn from_raw(raw: wgpu::Instance) -> Self {
        Self { raw }
    }

    /// Creates a surface from raw display and window handles.
    pub fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<WgpuSurface, Error> {
        let surface = unsafe {
            self.raw
                .create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: Some(display_handle),
                    raw_window_handle: window_handle,
                })
        }
        .map_err(|_| Error::fatal(ErrorKind::Backend("surface_create_failed")))?;

        Ok(WgpuSurface::new(surface))
    }
}

pub(crate) fn validate_device_descriptor(descriptor: &DeviceDescriptor) -> Result<(), Error> {
    if descriptor
        .required_features
        .contains(FeatureSet::SAMPLER_ANISOTROPY)
    {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "sampler_anisotropy_not_supported",
        )));
    }
    if descriptor
        .required_features
        .contains(FeatureSet::WIDE_LINES)
    {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "wide_lines_not_supported",
        )));
    }
    if !descriptor.work_queues.needs_graphics {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "graphics_queue_required",
        )));
    }
    Ok(())
}

pub(crate) async fn request_device_from_adapter(
    adapter: &wgpu::Adapter,
    descriptor: &DeviceDescriptor,
) -> Result<WgpuDevice, Error> {
    validate_device_descriptor(descriptor)?;

    let mut requested_features = wgpu::Features::empty();
    if descriptor
        .required_features
        .contains(FeatureSet::FILL_MODE_NON_SOLID)
    {
        requested_features.insert(wgpu::Features::POLYGON_MODE_LINE);
    }
    let available = adapter.features();
    let adapter_limits = adapter.limits();
    let mut required_limits = wgpu::Limits::default();
    if available.contains(wgpu::Features::IMMEDIATES) && adapter_limits.max_immediate_size > 0 {
        requested_features.insert(wgpu::Features::IMMEDIATES);
        required_limits.max_immediate_size = adapter_limits.max_immediate_size;
    }
    if available.contains(wgpu::Features::TIMESTAMP_QUERY) {
        requested_features.insert(wgpu::Features::TIMESTAMP_QUERY);
    }
    if available.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
        requested_features.insert(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
    }
    if available.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
        requested_features.insert(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    }
    if available.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
        requested_features.insert(wgpu::Features::INDIRECT_FIRST_INSTANCE);
    }
    if !adapter.features().contains(requested_features) {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "required_device_features_unavailable",
        )));
    }

    let wgpu_descriptor = WgpuDeviceDescriptor {
        required_features: requested_features,
        required_limits,
        ..Default::default()
    };
    let (raw, queue) = adapter
        .request_device(&wgpu_descriptor)
        .await
        .map_err(|_| Error::fatal(ErrorKind::NoCompatibleDevice))?;

    Ok(WgpuDevice {
        adapter: adapter.clone(),
        raw,
        queue,
    })
}

pub(crate) fn create_wgpu_instance(
    descriptor: &PlatformDescriptor,
) -> Result<wgpu::Instance, Error> {
    let mut instance = WgpuInstanceDescriptor::new_without_display_handle();
    instance.flags = match descriptor.debug {
        DebugMode::Validation => wgpu::InstanceFlags::debugging(),
        DebugMode::Off => wgpu::InstanceFlags::empty(),
    };
    Ok(wgpu::Instance::new(instance))
}

pub(crate) fn adapter_info(adapter: &wgpu::Adapter, id: AdapterId) -> AdapterInfo {
    let info = adapter.get_info();
    let features = adapter.features();
    AdapterInfo {
        id,
        name: info.name,
        hardware_family: map_hardware_family(info.device_type),
        vendor_id: info.vendor,
        device_id: info.device,
        driver_name: info.driver,
        driver_info: info.driver_info,
        api_backend: map_api_backend(info.backend),
        capabilities: AdapterCapabilities {
            supports_window_presentation: true,
            queue_support: AdapterQueueSupport {
                graphics_queue_count: 1,
                compute_queue_count: 1,
                transfer_queue_count: 1,
            },
            features: map_available_features(features),
            limits: caps::limits_from_wgpu(&adapter.limits()),
            core: CAPS,
        },
    }
}

fn map_hardware_family(device_type: wgpu::DeviceType) -> HardwareFamily {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => HardwareFamily::DiscreteGpu,
        wgpu::DeviceType::IntegratedGpu => HardwareFamily::IntegratedGpu,
        wgpu::DeviceType::VirtualGpu => HardwareFamily::VirtualGpu,
        wgpu::DeviceType::Cpu => HardwareFamily::Cpu,
        _ => HardwareFamily::Other,
    }
}

fn map_api_backend(backend: wgpu::Backend) -> GpuApiBackend {
    match backend {
        wgpu::Backend::Vulkan => GpuApiBackend::Vulkan,
        wgpu::Backend::Metal => GpuApiBackend::Metal,
        wgpu::Backend::Dx12 => GpuApiBackend::Dx12,
        wgpu::Backend::Gl => GpuApiBackend::OpenGl,
        wgpu::Backend::BrowserWebGpu => GpuApiBackend::WebGpu,
        _ => GpuApiBackend::Unknown,
    }
}

fn map_available_features(features: wgpu::Features) -> FeatureSet {
    let mut set = FeatureSet::INDIRECT_DRAW;
    if features.contains(wgpu::Features::POLYGON_MODE_LINE) {
        set = set.union(FeatureSet::FILL_MODE_NON_SOLID);
    }
    set
}
