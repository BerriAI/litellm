//! Audio routes: speech (TTS) and transcription (STT).
//!
//! This module provides the core functionality for audio generation and transcription,
//! following the same pattern as other routes (chat_completions, messages, embeddings, images).

mod client;
mod handler;
mod prepare;
mod transformation;
pub mod types;

pub use handler::{speech, transcription};
