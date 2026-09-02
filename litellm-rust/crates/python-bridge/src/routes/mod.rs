use pyo3::prelude::*;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;
mod runtime;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)
}
