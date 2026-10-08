//! Wire types for each provider API format (OpenAI Chat Completions,
//! Anthropic Messages, Responses, ...), one submodule per format.

pub mod audio_transcription;
pub mod batches;
pub mod chat_completions;
pub mod messages;
pub mod ocr;
pub mod responses;
