use rotex_core::{
    Error as CoreError, ErrorKind as CoreErrorKind, GpuPlatform, Severity as CoreSeverity,
};
use rotex_types::{AdapterId, AdapterInfo, DeviceDescriptor, PlatformDescriptor};

use crate::backend::wgpu::WgpuInstance;
use crate::backend::wgpu::device::{
    adapter_info, create_wgpu_instance, request_device_from_adapter,
};
use crate::bridge::WgpuBridge;
use crate::bridge::init::init_bridge;
use crate::error::{Error, ErrorKind};

/// WGPU implementation of [`GpuPlatform`].
///
/// Prefer the backend-free helpers in `rotex-core` (`create_backend_with_selection`, etc.)
/// with a type alias at the app layer. Hold [`WgpuBridge`] directly for backend-internal access.
pub struct WgpuGpuPlatform {
    instance: wgpu::Instance,
    adapters: Vec<wgpu::Adapter>,
}

#[async_trait::async_trait]
impl GpuPlatform for WgpuGpuPlatform {
    type Backend = WgpuBridge;

    async fn new(platform_descriptor: PlatformDescriptor) -> Result<Self, CoreError> {
        let instance = create_wgpu_instance(&platform_descriptor).map_err(to_core_error)?;
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        Ok(Self { instance, adapters })
    }

    async fn enumerate_adapters(&self) -> Result<Vec<AdapterInfo>, CoreError> {
        Ok(self
            .adapters
            .iter()
            .enumerate()
            .map(|(index, adapter)| adapter_info(adapter, AdapterId(index as u32)))
            .collect())
    }

    async fn create_backend(
        self,
        adapter_id: AdapterId,
        device_descriptor: DeviceDescriptor,
    ) -> Result<Self::Backend, CoreError> {
        let adapter = self
            .adapters
            .get(adapter_id.0 as usize)
            .ok_or_else(|| CoreError::fatal(CoreErrorKind::NoCompatibleDevice))?;
        let device = request_device_from_adapter(adapter, &device_descriptor)
            .await
            .map_err(to_core_error)?;
        let instance = WgpuInstance::from_raw(self.instance);
        init_bridge(instance, device).map_err(to_core_error)
    }
}

fn to_core_error(error: Error) -> CoreError {
    let severity = match error.severity {
        crate::error::Severity::Fatal => CoreSeverity::Fatal,
        crate::error::Severity::Info
        | crate::error::Severity::Warning
        | crate::error::Severity::Recoverable => CoreSeverity::Warning,
    };
    let kind = match error.kind {
        ErrorKind::NoCompatibleDevice => CoreErrorKind::NoCompatibleDevice,
        ErrorKind::Unsupported(message) => CoreErrorKind::Unsupported(message),
        other => CoreErrorKind::Backend(format!("{other:?}")),
    };
    CoreError { kind, severity }
}
