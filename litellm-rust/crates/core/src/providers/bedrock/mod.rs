//! User-directed exception: this base provider owns AWS auth I/O for parity
//! with Python's `BaseAWSLLM`; the broader core purity guidance is reconciled
//! separately.

#[cfg(feature = "bedrock-auth")]
pub mod audio_transcription;
pub mod aws_base;
mod constants;
