use pyo3::prelude::*;

#[macro_use]
mod definition;

#[cfg(feature = "trace-parity")]
mod gateway_messages;

mod audio_transcription;
mod chat_completions;
mod messages;
mod ocr;

#[cfg(feature = "bench")]
mod benchmarks;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "bench")]
    benchmarks::register_namespace(module)?;
    ocr::register(module)?;
    audio_transcription::register(module)?;
    messages::register(module)?;
    chat_completions::register(module)?;
    #[cfg(feature = "trace-parity")]
    {
        let trace = PyModule::new(module.py(), "_trace")?;
        ocr::register_trace(&trace)?;
        audio_transcription::register_trace(&trace)?;
        messages::register_trace(&trace)?;
        chat_completions::register_trace(&trace)?;
        gateway_messages::register_trace(&trace)?;
        module.add_submodule(&trace)?;
    }
    Ok(())
}
