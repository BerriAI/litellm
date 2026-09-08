use pyo3::prelude::*;

mod definition;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;
mod responses_websocket;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)?;
    responses_websocket::register(module)?;
    #[cfg(feature = "trace-parity")]
    {
        let trace = PyModule::new(module.py(), "_trace")?;
        ocr::register_trace(&trace)?;
        audio_transcription::register_trace(&trace)?;
        messages::register_trace(&trace)?;
        chat_completions::register_trace(&trace)?;
        crate::trace_parity::register_gateway(&trace)?;
        module.add_submodule(&trace)?;
    }
    Ok(())
}
