//! Audio types for speech (TTS) and transcription (STT).

use serde::{Deserialize, Serialize};

/// Request for text-to-speech (TTS) generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRequest {
    /// The text to generate audio for.
    pub input: String,
    /// The model to use for generation (e.g., "tts-1", "tts-1-hd").
    pub model: String,
    /// The voice to use (e.g., "alloy", "echo", "fable", "onyx", "nova", "shimmer").
    pub voice: String,
    /// The format to return audio in (e.g., "mp3", "opus", "aac", "flac").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// The speed of the generated audio (0.25 to 4.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

/// Response from text-to-speech (TTS) generation.
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    /// The generated audio data.
    pub audio_data: Vec<u8>,
    /// The content type of the audio (e.g., "audio/mpeg").
    pub content_type: String,
}

/// Request for speech-to-text (STT) transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// The audio file to transcribe (as bytes).
    pub file: Vec<u8>,
    /// The model to use for transcription (e.g., "whisper-1").
    pub model: String,
    /// The language of the input audio (ISO-639-1 code).
    pub language: Option<String>,
    /// An optional prompt to guide the transcription.
    pub prompt: Option<String>,
    /// The format of the transcript output (e.g., "json", "text", "srt", "verbose_json", "vtt").
    pub response_format: Option<String>,
    /// The sampling temperature (0 to 1).
    pub temperature: Option<f32>,
}

/// Response from speech-to-text (STT) transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
}
