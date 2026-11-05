use std::fmt::{Display, Formatter};

/// How serious a backend error is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Diagnostic that does not change control flow.
    Info,
    /// Recoverable condition worth reporting.
    Warning,
    /// The call failed and the device can still be used.
    Recoverable,
    /// The call failed and the device should be torn down.
    Fatal,
}

/// Why a wgpu backend call failed.
#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// The operation is not available. The string says what was missing.
    Unsupported(&'static str),
    /// No adapter matched the request.
    NoCompatibleDevice,
    /// A presentation command ran before a surface was attached.
    SurfaceNotAttached,
    /// The swapchain must be recreated before another frame can be acquired.
    SurfaceOutdated,
    /// The surface can no longer be used.
    SurfaceLost,
    /// Acquiring a surface image timed out.
    SurfaceTimeout,
    /// The surface is occluded and cannot present.
    SurfaceOccluded,
    /// A command named a resource the bridge does not hold.
    ResourceNotFound(&'static str),
    /// A resource descriptor failed validation.
    InvalidDescriptor(&'static str),
    /// Texture bytes could not be uploaded.
    TextureUploadFailed(&'static str),
    /// A pipeline could not be compiled.
    PipelineCreationFailed(&'static str),
    /// wgpu reported a failure. The string names the call.
    Backend(&'static str),
}

/// Error returned by the wgpu backend.
#[derive(Debug, Clone)]
pub struct Error {
    /// Why the call failed.
    pub kind: ErrorKind,
    /// Whether the device can keep running.
    pub severity: Severity,
}

impl Error {
    /// Builds an error with an explicit severity.
    pub const fn new(kind: ErrorKind, severity: Severity) -> Self {
        Self { kind, severity }
    }

    /// Builds an error with [`Severity::Fatal`].
    pub const fn fatal(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Fatal)
    }

    /// Builds an error with [`Severity::Recoverable`].
    pub const fn recoverable(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Recoverable)
    }

    /// Fatal error for an operation this backend cannot perform.
    pub const fn unsupported(message: &'static str) -> Self {
        Self::fatal(ErrorKind::Unsupported(message))
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:?}", self.severity, self.kind)
    }
}

impl std::error::Error for Error {}
