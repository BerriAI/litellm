use pyo3::prelude::*;

#[macro_use]
mod definition;

mod audio_transcription;
mod chat_completions;
mod messages;
mod messages_stream;
mod ocr;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    messages_stream::register(module)?;
    chat_completions::register(module)
}
