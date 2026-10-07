pub(crate) mod audio_transcription;
pub(crate) mod chat_completions;
pub(crate) mod embeddings;
mod inference;
pub(crate) mod messages;
pub(crate) mod ocr;
pub(crate) mod responses;
pub(crate) mod token_counter;
pub(crate) mod traces;

use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_callbacks_legacy_python::{LegacyLogging, PublicCall};
use litellm_host::{call::HostedCompletion, machine::Machine, protocol::Protocol};
use litellm_host_python::{HookChain, PythonBinding, PythonCallHooks, PythonHostCalls};
use pyo3::{
    prelude::*,
    types::{PyDict, PyMapping, PyTuple},
};

struct NativeCall<'py> {
    args: Bound<'py, PyTuple>,
    kwargs: Bound<'py, PyDict>,
    bound: Bound<'py, PyDict>,
}

impl<'py> NativeCall<'py> {
    fn extract(call: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            args: call.getattr("args")?.cast_into()?,
            kwargs: mapping_dict(&call.getattr("kwargs")?)?,
            bound: mapping_dict(&call.getattr("bound")?)?,
        })
    }
}

fn mapping_dict<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(dict.clone());
    }
    let mapping = value.cast::<PyMapping>()?;
    let dict = PyDict::new(value.py());
    dict.update(mapping)?;
    Ok(dict)
}

fn call_hooks(
    py: Python<'_>,
    operation: LoggingOperation,
    request: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: &Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<(Py<PyDict>, impl PythonCallHooks + use<>)> {
    let call = PublicCall::capture(request, args, kwargs)?;
    let arguments = call.arguments(py);
    Ok((
        arguments,
        LegacyLogging::new(py, operation, call, asynchronous),
    ))
}

fn run_public_call<H, M>(
    py: Python<'_>,
    arguments: Py<PyDict>,
    start: impl FnOnce(
        Python<'_>,
        &Bound<'_, PyDict>,
        <H::Protocol as Protocol>::Request,
    ) -> PyResult<M>
    + Send
    + Sync
    + 'static,
    host: H,
    hooks: impl PythonCallHooks + 'static,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: PythonBinding + PythonHostCalls<H::Protocol> + 'static,
    M: Machine<Protocol = H::Protocol> + 'static,
    M::Complete: Into<HostedCompletion<<H::Protocol as Protocol>::Response>>,
{
    litellm_host_python::run_call(
        py,
        move |py, arguments, request| {
            start(py, arguments, request).map(crate::logger::LoggedMachine::new)
        },
        host,
        HookChain::new()
            .with(hooks)
            .with(crate::preflight::SdkPolicy),
        arguments,
        crate::lifecycle::call_options(asynchronous),
    )
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyList},
    };

    fn value_call<'py>(
        py: Python<'py>,
        payload_name: &str,
        payload: &Bound<'py, PyAny>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> Bound<'py, PyAny> {
        let fields = PyDict::new(py);
        fields.set_item("model", "model").unwrap();
        fields.set_item(payload_name, payload).unwrap();
        if let Some(kwargs) = kwargs {
            fields.update(kwargs.as_mapping()).unwrap();
        }
        let attributes = PyDict::new(py);
        attributes
            .set_item("args", pyo3::types::PyTuple::empty(py))
            .unwrap();
        attributes.set_item("kwargs", &fields).unwrap();
        attributes.set_item("bound", &fields).unwrap();
        py.import("types")
            .unwrap()
            .getattr("SimpleNamespace")
            .unwrap()
            .call((), Some(&attributes))
            .unwrap()
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
                    .and_then(|function| {
                        function.call1((value_call(py, "messages", &broken, None),))
                    })
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
                .and_then(|function| {
                    function.call1((value_call(py, "messages", &invalid_messages, None),))
                })
                .expect_err("sync chat should reject a non-list messages value");
            let async_chat_error = module
                .getattr("achat_completions")
                .and_then(|function| {
                    function.call1((value_call(py, "messages", &invalid_messages, None),))
                })
                .expect_err("async chat should reject a non-list messages value");

            assert_eq!(
                sync_chat_error.to_string(),
                "ValueError: messages must be a list"
            );
            assert_eq!(async_chat_error.to_string(), sync_chat_error.to_string());

            let invalid_headers = PyList::empty(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("extra_headers", &invalid_headers)
                .expect("kwargs should accept extra_headers");
            let audio = PyDict::new(py);

            let sync_error = module
                .getattr("transcription")
                .and_then(|function| {
                    function.call1((value_call(py, "audio", &audio, Some(&kwargs)),))
                })
                .expect_err("sync route should reject non-dict extra_headers");
            let async_error = module
                .getattr("atranscription")
                .and_then(|function| {
                    function.call1((value_call(py, "audio", &audio, Some(&kwargs)),))
                })
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
                    function.call1((value_call(
                        py,
                        "messages",
                        &invalid_messages,
                        Some(&chat_kwargs),
                    ),))
                })
                .expect_err("messages should be validated first");
            assert_eq!(error.to_string(), "ValueError: messages must be a list");

            let valid_messages = PyList::empty(py);
            let error = module
                .getattr("chat_completions")
                .and_then(|function| {
                    function.call1((value_call(
                        py,
                        "messages",
                        &valid_messages,
                        Some(&chat_kwargs),
                    ),))
                })
                .expect_err("optional_params should be validated before headers");
            assert_eq!(
                error.to_string(),
                "ValueError: optional_params must be a dict"
            );

            let headers_kwargs = PyDict::new(py);
            headers_kwargs
                .set_item("extra_headers", &invalid)
                .expect("kwargs should accept extra_headers");
            let invalid_payload =
                PyModule::new(py, "invalid_payload").expect("invalid payload should be created");
            let error = module
                .getattr("transcription")
                .and_then(|function| {
                    function.call1((value_call(
                        py,
                        "audio",
                        &invalid_payload,
                        Some(&headers_kwargs),
                    ),))
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
                .and_then(|function| {
                    function.call1((value_call(py, "messages", &messages, Some(&omitted)),))
                })
                .expect_err("omitted optional_params should reach header validation");
            let explicit_error = module
                .getattr("chat_completions")
                .and_then(|function| {
                    function.call1((value_call(py, "messages", &messages, Some(&explicit)),))
                })
                .expect_err("None optional_params should reach header validation");
            assert_eq!(
                omitted_error.to_string(),
                "ValueError: extra_headers must be a dict"
            );
            assert_eq!(explicit_error.to_string(), omitted_error.to_string());
        });
    }
}
