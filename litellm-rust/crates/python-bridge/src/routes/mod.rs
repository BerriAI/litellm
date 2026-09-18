pub(crate) mod audio_transcription;
pub(crate) mod ocr;
pub(crate) mod responses;

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    const SIGNATURE: &str = "(model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None)";

    fn text_signature(module: &Bound<'_, PyModule>, name: &str) -> String {
        module
            .getattr(name)
            .and_then(|function| function.getattr("__text_signature__"))
            .and_then(|signature| signature.extract())
            .expect("route signature should be available")
    }

    #[test]
    fn sync_and_async_route_signatures_match_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            assert_eq!(text_signature(&module, "transcription"), SIGNATURE);
            assert_eq!(text_signature(&module, "atranscription"), SIGNATURE);
        });
    }

    #[test]
    fn route_arguments_that_fail_to_convert_raise_value_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class Broken:
    def __index__(self):
        raise LookupError('conversion failed')
value = Broken()
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .expect("helper class should define");
            let broken = locals
                .get_item("value")
                .expect("locals should be readable")
                .expect("helper value should exist");

            for name in ["transcription", "atranscription"] {
                let error = module
                    .getattr(name)
                    .and_then(|function| function.call1(("model", &broken)))
                    .expect_err("route should reject a value it cannot convert");

                assert!(
                    error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                    "{name} surfaced {error} instead of ValueError"
                );
            }
        });
    }

    #[test]
    fn sync_and_async_routes_apply_the_same_input_validation() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let audio = PyDict::new(py);
            let invalid = PyList::empty(py);

            for (argument, expected) in [
                ("extra_headers", "ValueError: extra_headers must be a dict"),
                (
                    "optional_params",
                    "ValueError: optional_params must be a dict",
                ),
            ] {
                let kwargs = PyDict::new(py);
                kwargs
                    .set_item(argument, &invalid)
                    .expect("kwargs should accept the argument");
                let sync_error = module
                    .getattr("transcription")
                    .and_then(|function| function.call(("model", &audio), Some(&kwargs)))
                    .expect_err("sync route should reject a non-dict argument");
                let async_error = module
                    .getattr("atranscription")
                    .and_then(|function| function.call(("model", &audio), Some(&kwargs)))
                    .expect_err("async route should reject a non-dict argument");

                assert_eq!(sync_error.to_string(), expected);
                assert_eq!(async_error.to_string(), sync_error.to_string());
            }
        });
    }

    #[test]
    fn route_input_validation_preserves_left_to_right_order() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let invalid = PyList::empty(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            kwargs
                .set_item("optional_params", &invalid)
                .expect("kwargs should accept optional_params");

            let invalid_payload =
                PyModule::new(py, "invalid_payload").expect("invalid payload should be created");
            let error = module
                .getattr("transcription")
                .and_then(|function| function.call(("model", &invalid_payload), Some(&kwargs)))
                .expect_err("payload should be validated before headers");
            assert!(!error.to_string().contains("extra_headers"));

            let audio = PyDict::new(py);
            let error = module
                .getattr("transcription")
                .and_then(|function| function.call(("model", &audio), Some(&kwargs)))
                .expect_err("headers should be validated before optional_params");
            assert_eq!(
                error.to_string(),
                "ValueError: extra_headers must be a dict"
            );
        });
    }

    #[test]
    fn missing_and_explicit_none_extra_headers_share_the_next_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let audio = PyDict::new(py);
            let invalid = PyList::empty(py);
            let omitted = PyDict::new(py);
            omitted
                .set_item("optional_params", &invalid)
                .expect("kwargs should accept optional_params");
            let explicit = PyDict::new(py);
            explicit
                .set_item("extra_headers", py.None())
                .expect("kwargs should accept extra_headers");
            explicit
                .set_item("optional_params", &invalid)
                .expect("kwargs should accept optional_params");

            let omitted_error = module
                .getattr("transcription")
                .and_then(|function| function.call(("model", &audio), Some(&omitted)))
                .expect_err("omitted extra_headers should reach optional_params validation");
            let explicit_error = module
                .getattr("transcription")
                .and_then(|function| function.call(("model", &audio), Some(&explicit)))
                .expect_err("None extra_headers should reach optional_params validation");
            assert_eq!(
                omitted_error.to_string(),
                "ValueError: optional_params must be a dict"
            );
            assert_eq!(explicit_error.to_string(), omitted_error.to_string());
        });
    }
}
