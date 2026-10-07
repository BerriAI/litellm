use litellm_llms::base_llm::ocr::error::Error as OcrError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Ocr(#[from] OcrError),
    #[error(transparent)]
    Messages(#[from] crate::messages::Error),
    #[error(transparent)]
    ChatCompletions(#[from] crate::chat_completions::Error),
    #[error(transparent)]
    AudioTranscription(#[from] crate::audio_transcription::Error),
    #[error(transparent)]
    Responses(#[from] crate::responses::Error),
}
