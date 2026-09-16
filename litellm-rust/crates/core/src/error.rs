#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Ocr(#[from] crate::ocr::Error),
    #[error(transparent)]
    Messages(#[from] crate::messages::Error),
    #[error(transparent)]
    ChatCompletions(#[from] crate::chat_completions::Error),
    #[error(transparent)]
    AudioTranscription(#[from] crate::audio_transcription::Error),
    #[error(transparent)]
    Responses(#[from] crate::responses::Error),
    #[error(transparent)]
    Realtime(#[from] crate::realtime::Error),
}
