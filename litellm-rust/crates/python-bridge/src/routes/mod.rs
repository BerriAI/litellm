use pyo3::prelude::*;

#[macro_use]
mod definition;

mod chat_completions;
mod embeddings;
mod image_edit;
mod image_generation;
mod messages;
mod moderation;
mod ocr;
mod rerank;
mod responses;
mod speech;
mod token_counter;
mod transcription;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)?;
    embeddings::register(module)?;
    image_edit::register(module)?;
    image_generation::register(module)?;
    moderation::register(module)?;
    rerank::register(module)?;
    responses::register(module)?;
    speech::register(module)?;
    token_counter::register(module)?;

    #[cfg(feature = "trace-parity")]
    {
        let trace = PyModule::new(module.py(), "_trace")?;
        ocr::register_trace(&trace)?;
        transcription::register_trace(&trace)?;
        messages::register_trace(&trace)?;
        chat_completions::register_trace(&trace)?;
        module.add_submodule(&trace)?;
    }
    Ok(())
}
