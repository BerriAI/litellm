pub(crate) mod audio_transcription;
pub(crate) mod chat_completions;
pub(crate) mod messages;
pub(crate) mod ocr;
pub(crate) mod responses;

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    #[test]
    fn sync_and_async_route_signatures_match_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let routes = [
                (
                    "transcription",
                    "atranscription",
                    "(model, audio, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None)",
                ),
                (
                    "messages",
                    "amessages",
                    "(model, body, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None)",
                ),
                (
                    "chat_completions",
                    "achat_completions",
                    "(model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None)",
                ),
            ];

            for (sync_name, async_name, expected) in routes {
                let sync_signature: String = module
                    .getattr(sync_name)
                    .and_then(|function| function.getattr("__text_signature__"))
                    .and_then(|signature| signature.extract())
                    .expect("sync signature should be available");
                let async_signature: String = module
                    .getattr(async_name)
                    .and_then(|function| function.getattr("__text_signature__"))
                    .and_then(|signature| signature.extract())
                    .expect("async signature should be available");

                assert_eq!(sync_signature, expected);
                assert_eq!(async_signature, expected);
            }
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

            for name in ["chat_completions", "achat_completions"] {
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

            let invalid_messages = PyDict::new(py);
            let sync_chat_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call1(("model", &invalid_messages)))
                .expect_err("sync chat should reject a non-list messages value");
            let async_chat_error = module
                .getattr("achat_completions")
                .and_then(|function| function.call1(("model", &invalid_messages)))
                .expect_err("async chat should reject a non-list messages value");

            assert_eq!(
                sync_chat_error.to_string(),
                "ValueError: messages must be a list"
            );
            assert_eq!(async_chat_error.to_string(), sync_chat_error.to_string());

            let invalid_body = PyList::empty(py);
            let sync_messages_error = module
                .getattr("messages")
                .and_then(|function| function.call1(("model", &invalid_body)))
                .expect_err("sync Messages should reject a non-dict body");
            let async_messages_error = module
                .getattr("amessages")
                .and_then(|function| function.call1(("model", &invalid_body)))
                .expect_err("async Messages should reject a non-dict body");

            assert_eq!(
                sync_messages_error.to_string(),
                "ValueError: body must be a dict"
            );
            assert_eq!(
                async_messages_error.to_string(),
                sync_messages_error.to_string()
            );

            let invalid_headers = PyList::empty(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("extra_headers", &invalid_headers)
                .expect("kwargs should accept extra_headers");
            let audio = PyDict::new(py);

            let sync_error = module
                .getattr("transcription")
                .and_then(|function| function.call(("model", &audio), Some(&kwargs)))
                .expect_err("sync route should reject non-dict extra_headers");
            let async_error = module
                .getattr("atranscription")
                .and_then(|function| function.call(("model", &audio), Some(&kwargs)))
                .expect_err("async route should reject non-dict extra_headers");

            assert_eq!(
                sync_error.to_string(),
                "ValueError: extra_headers must be a dict"
            );
            assert_eq!(async_error.to_string(), sync_error.to_string());
        });
    }

    #[test]
    fn route_input_validation_preserves_left_to_right_order() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let invalid = PyList::empty(py);

            let chat_kwargs = PyDict::new(py);
            chat_kwargs
                .set_item("optional_params", &invalid)
                .expect("kwargs should accept optional_params");
            chat_kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            let invalid_messages = PyDict::new(py);
            let error = module
                .getattr("chat_completions")
                .and_then(|function| {
                    function.call(("model", &invalid_messages), Some(&chat_kwargs))
                })
                .expect_err("messages should be validated first");
            assert_eq!(error.to_string(), "ValueError: messages must be a list");

            let valid_messages = PyList::empty(py);
            let error = module
                .getattr("chat_completions")
                .and_then(|function| function.call(("model", &valid_messages), Some(&chat_kwargs)))
                .expect_err("optional_params should be validated before headers");
            assert_eq!(
                error.to_string(),
                "ValueError: optional_params must be a dict"
            );

            let headers_kwargs = PyDict::new(py);
            headers_kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            let invalid_body = PyList::empty(py);
            let error = module
                .getattr("messages")
                .and_then(|function| function.call(("model", &invalid_body), Some(&headers_kwargs)))
                .expect_err("body should be validated before headers");
            assert_eq!(error.to_string(), "ValueError: body must be a dict");

            let invalid_payload =
                PyModule::new(py, "invalid_payload").expect("invalid payload should be created");
            let error = module
                .getattr("transcription")
                .and_then(|function| {
                    function.call(("model", &invalid_payload), Some(&headers_kwargs))
                })
                .expect_err("payload should be validated before headers");
            assert!(!error.to_string().contains("extra_headers"));
        });
    }

    #[test]
    fn missing_and_explicit_none_optional_params_share_the_next_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let messages = PyList::empty(py);
            let headers = PyList::empty(py);
            let omitted = PyDict::new(py);
            omitted
                .set_item("extra_headers", &headers)
                .expect("kwargs should accept extra_headers");
            let explicit = PyDict::new(py);
            explicit
                .set_item("optional_params", py.None())
                .expect("kwargs should accept optional_params");
            explicit
                .set_item("extra_headers", &headers)
                .expect("kwargs should accept extra_headers");

            let omitted_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call(("model", &messages), Some(&omitted)))
                .expect_err("omitted optional_params should reach header validation");
            let explicit_error = module
                .getattr("chat_completions")
                .and_then(|function| function.call(("model", &messages), Some(&explicit)))
                .expect_err("None optional_params should reach header validation");
            assert_eq!(
                omitted_error.to_string(),
                "ValueError: extra_headers must be a dict"
            );
            assert_eq!(explicit_error.to_string(), omitted_error.to_string());
        });
    }
}
