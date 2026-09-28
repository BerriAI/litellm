pub mod audio_transcription;
pub mod llms;
pub mod messages;
pub mod recognized;
pub mod responses;
pub mod utils;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Completion,
    Responses,
    Messages,
    Ocr,
}
