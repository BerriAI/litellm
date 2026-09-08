#![recursion_limit = "256"]

mod diagnostics;
mod driver;
mod errors;
mod marshal;
mod retained;
mod routes;
#[cfg(feature = "trace-parity")]
mod trace_parity;

use pyo3::prelude::*;

#[pymodule(gil_used = true)]
mod _native {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::errors::register(module)?;
        super::routes::register(module)?;
        super::diagnostics::register(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = pyo3::wrap_pymodule!(_native)(py).into_bound(py);

            let expected = [
                "RustBridgeDeclined",
                "RustUpstreamError",
                "RustBridgeDriverError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions",
                "achat_completions",
                "chat_completions_decline",
                "responses_websocket",
            ];

            let public_names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            assert_eq!(public_names, expected);

            #[cfg(not(feature = "trace-parity"))]
            assert!(!module.hasattr("_trace").expect("module lookup should work"));

            #[cfg(feature = "trace-parity")]
            {
                let trace = module
                    .getattr("_trace")
                    .expect("trace build should expose its diagnostic namespace");
                let trace_names: Vec<String> = trace
                    .cast::<PyModule>()
                    .expect("trace namespace should be a module")
                    .dict()
                    .keys()
                    .extract::<Vec<String>>()
                    .expect("trace names should be strings")
                    .into_iter()
                    .filter(|name| !name.starts_with("__"))
                    .collect();
                assert_eq!(
                    trace_names,
                    [
                        "ocr",
                        "aocr",
                        "transcription",
                        "atranscription",
                        "messages",
                        "amessages",
                        "chat_completions",
                        "achat_completions",
                        "chat_completions_decline",
                        "gateway_messages",
                    ]
                );
            }
        });
    }
}
