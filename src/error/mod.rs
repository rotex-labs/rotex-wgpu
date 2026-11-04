use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Recoverable,
    Fatal,
}

#[derive(Debug, Clone)]
pub enum ErrorKind {
    Unsupported(&'static str),
    NoCompatibleDevice,
    SurfaceNotAttached,
    SurfaceOutdated,
    SurfaceLost,
    SurfaceTimeout,
    SurfaceOccluded,
    ResourceNotFound(&'static str),
    InvalidDescriptor(&'static str),
    TextureUploadFailed(&'static str),
    PipelineCreationFailed(&'static str),
    Backend(&'static str),
}

#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub severity: Severity,
}

impl Error {
    pub const fn new(kind: ErrorKind, severity: Severity) -> Self {
        Self { kind, severity }
    }

    pub const fn fatal(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Fatal)
    }

    pub const fn recoverable(kind: ErrorKind) -> Self {
        Self::new(kind, Severity::Recoverable)
    }

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
