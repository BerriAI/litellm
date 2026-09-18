mod credentials;
mod diagnostics;
mod errors;
mod marshal;
mod routes;
mod token_counter;

#[pymodule(gil_used = true)]
mod _native {
    #[cfg(feature = "panic-test")]
    #[pymodule_export]
    use crate::diagnostics::_panic_for_test;
    #[pymodule_export]
    use crate::diagnostics::gil_stats;
    #[pymodule_export]
    use crate::errors::{RustBridgeDeclined, RustUpstreamError};
    #[pymodule_export]
    use crate::routes::audio_transcription::{atranscription, transcription};
    #[pymodule_export]
    use crate::routes::ocr::{aocr, ocr};
    #[pymodule_export]
    use crate::routes::responses::ResponsesWebSocketConnection;
    #[pymodule_export]
    use crate::token_counter::TokenCounter;
}

use pyo3::prelude::*;

#[cfg(test)]
pub(crate) fn native_module(py: Python<'_>) -> Bound<'_, PyModule> {
    pyo3::wrap_pymodule!(_native)(py).into_bound(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let mut expected = vec![
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "ResponsesWebSocketConnection",
                "TokenCounter",
                "gil_stats",
            ];
            expected.sort_unstable();

            let mut public_names: Vec<String> = native_module(py)
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            public_names.sort_unstable();
            assert_eq!(public_names, expected);
        });
    }
}
