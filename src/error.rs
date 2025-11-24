use std::fmt;

#[derive(Debug)]
pub enum FftCorrelationError {
    FftProcessing(String),
}

impl fmt::Display for FftCorrelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FftCorrelationError::FftProcessing(msg) => write!(f, "FFT processing error: {}", msg),
        }
    }
}

impl std::error::Error for FftCorrelationError {}

pub type Result<T> = std::result::Result<T, FftCorrelationError>;
