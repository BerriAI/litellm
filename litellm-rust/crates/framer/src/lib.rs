mod error;

pub use error::Error;

pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[cfg(feature = "aws")]
pub mod aws_event_stream;
#[cfg(feature = "sse")]
pub mod sse;
