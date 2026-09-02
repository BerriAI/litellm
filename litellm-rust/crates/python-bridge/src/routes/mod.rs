use pyo3::prelude::*;

#[macro_use]
mod definition;
mod runtime;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;
mod receiver;
mod streaming;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)?;
    streaming::register(module)
}
