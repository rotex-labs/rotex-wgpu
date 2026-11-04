use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use wgpu::{
    DeviceDescriptor as WgpuDeviceDescriptor, InstanceDescriptor as WgpuInstanceDescriptor,
    SurfaceTargetUnsafe,
};

use super::surface::WgpuSurface;
use crate::error::{Error, ErrorKind};
use rotex_types::{
    AdapterCapabilities, AdapterId, AdapterInfo, AdapterQueueSupport, DeviceDescriptor,
    DeviceFeatures, FeatureSet, GpuApiBackend, HardwareFamily, PlatformDescriptor,
};

use crate::caps::{self, CAPS};

pub struct WgpuInstance {
    pub(crate) raw: wgpu::Instance,
}

pub struct WgpuDevice {
    pub adapter: wgpu::Adapter,
    pub raw: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuInstance {
    pub(crate) fn from_raw(raw: wgpu::Instance) -> Self {
        Self { raw }
    }

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
    if descriptor.required_features.sampler_anisotropy {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "sampler_anisotropy_not_supported",
        )));
    }
    if descriptor.required_features.wide_lines {
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
    if descriptor.required_features.fill_mode_non_solid {
        requested_features.insert(wgpu::Features::POLYGON_MODE_LINE);
    }
    let available = adapter.features();
    if available.contains(wgpu::Features::IMMEDIATES) && adapter.limits().max_immediate_size > 0 {
        requested_features.insert(wgpu::Features::IMMEDIATES);
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
    if !adapter.features().contains(requested_features) {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "required_device_features_unavailable",
        )));
    }

    let mut wgpu_descriptor = WgpuDeviceDescriptor::default();
    wgpu_descriptor.required_features = requested_features;
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

pub(crate) async fn create_wgpu_instance(
    descriptor: &PlatformDescriptor,
) -> Result<wgpu::Instance, Error> {
    if !descriptor.platform_extensions.is_empty() {
        return Err(Error::recoverable(ErrorKind::Unsupported(
            "platform_extensions_not_supported",
        )));
    }
    Ok(wgpu::Instance::new(
        WgpuInstanceDescriptor::new_without_display_handle(),
    ))
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
            available_features: map_available_features(features),
            limits: caps::limits_from_wgpu(&adapter.limits()),
            extensions: FeatureSet::empty(),
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

fn map_available_features(features: wgpu::Features) -> DeviceFeatures {
    DeviceFeatures {
        sampler_anisotropy: false,
        fill_mode_non_solid: features.contains(wgpu::Features::POLYGON_MODE_LINE),
        wide_lines: false,
    }
}
