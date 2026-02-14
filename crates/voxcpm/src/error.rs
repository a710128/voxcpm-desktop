use core::fmt;

/// Crate error type.
#[derive(Debug)]
pub enum VoxCpmError {
    /// A stubbed API was called.
    Unimplemented(&'static str),
    /// Invalid user input.
    InvalidArg(String),

    /// Generation was cancelled by the caller.
    Cancelled,
    /// IO error.
    Io(std::io::Error),

    /// JSON parse error.
    Json(serde_json::Error),

    /// Tokenizer error.
    Tokenizer(tokenizers::Error),

    /// Candle error.
    Candle(candle_core::Error),
}

impl VoxCpmError {
    pub fn unimplemented(what: &'static str) -> Self {
        Self::Unimplemented(what)
    }
}

impl fmt::Display for VoxCpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unimplemented(what) => write!(f, "{what} is not implemented"),
            Self::InvalidArg(msg) => write!(f, "invalid argument: {msg}"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::Tokenizer(err) => write!(f, "tokenizer error: {err}"),
            Self::Candle(err) => write!(f, "candle error: {err}"),
        }
    }
}

impl std::error::Error for VoxCpmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Json(err) => Some(err),
            Self::Tokenizer(err) => Some(err.as_ref()),
            Self::Candle(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VoxCpmError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<candle_core::Error> for VoxCpmError {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}

pub type Result<T> = core::result::Result<T, VoxCpmError>;
