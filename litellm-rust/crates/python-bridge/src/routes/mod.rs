pub(crate) mod audio_transcription;
pub(crate) mod chat_completions;
pub(crate) mod embeddings;
mod inference;
pub(crate) mod messages;
pub(crate) mod ocr;
pub(crate) mod responses;
pub(crate) mod token_counter;
pub(crate) mod traces;

use litellm_callbacks_legacy_python::{LegacyLogging, LoggingOperation, PublicCall};
use litellm_host::{call::HostedCompletion, machine::Machine, protocol::Protocol};
use litellm_host_python::{
    HookChain, PythonBinding, PythonCallHooks, PythonHostCalls, effective_py_args,
};
use litellm_host_python::{missing_state, present};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyMapping, PyTuple},
};

/// The public call as Python bound it: `base` holds the positional arguments by name plus
/// the signature defaults, `kwargs` the caller's keyword dict.
#[derive(FromPyObject)]
pub(crate) struct NativeCall<'py> {
    #[pyo3(attribute)]
    args: Bound<'py, PyTuple>,
    #[pyo3(attribute, from_py_with = mapping_dict)]
    kwargs: Bound<'py, PyDict>,
    #[pyo3(attribute, from_py_with = mapping_dict)]
    base: Bound<'py, PyDict>,
}

impl<'py> NativeCall<'py> {
    /// The call before any hook ran, for the reads that admit or decline it.
    fn resolved(&self) -> PyResult<Bound<'py, PyDict>> {
        effective_py_args(&self.base, &self.kwargs)
    }

    /// The view a route host reads its arguments from.
    fn view(&self) -> PyResult<RequestView> {
        RequestView::new(&self.base, &self.kwargs)
    }
}

/// The keyword view a route host reads: the signature base under the caller's keywords,
/// laid again once the hooks have rewritten them so a keyword a hook deleted stays deleted
/// instead of falling back to the caller's value.
pub(crate) struct RequestView(Option<RequestDicts>);

struct RequestDicts {
    base: Py<PyDict>,
    request: Py<PyDict>,
}

impl RequestView {
    pub(crate) fn new(base: &Bound<'_, PyDict>, kwargs: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self(Some(RequestDicts {
            base: base.clone().unbind(),
            request: effective_py_args(base, kwargs)?.unbind(),
        })))
    }

    fn dicts(&self) -> PyResult<&RequestDicts> {
        self.0.as_ref().ok_or_else(missing_state)
    }

    pub(crate) fn resolve(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let dicts = self.0.as_mut().ok_or_else(missing_state)?;
        dicts.request = effective_py_args(dicts.base.bind(py), arguments)?.unbind();
        Ok(())
    }

    pub(crate) fn request<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyDict>> {
        Ok(self.dicts()?.request.bind(py))
    }

    pub(crate) fn argument<'py>(
        &self,
        py: Python<'py>,
        arguments: &Bound<'py, PyDict>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        present(arguments, self.request(py)?, name)
    }

    pub(crate) fn close(&mut self) {
        self.0 = None;
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(dicts) = &self.0 else {
            return Ok(());
        };
        visit.call(&dicts.base)?;
        visit.call(&dicts.request)
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
    call: &NativeCall<'_>,
    asynchronous: bool,
) -> PyResult<(Py<PyDict>, impl PythonCallHooks + use<>)> {
    let call = PublicCall::capture(&call.base, &call.args, &call.kwargs)?;
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
    use rstest::rstest;

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
        attributes.set_item("base", PyDict::new(py)).unwrap();
        py.import("types")
            .unwrap()
            .getattr("SimpleNamespace")
            .unwrap()
            .call((), Some(&attributes))
            .unwrap()
    }

    fn dict<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyDict> {
        py.eval(&std::ffi::CString::new(source).unwrap(), None, None)
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest]
    #[case::deleted_keyword_stays_deleted("{'system': None}", "{'system': 'caller'}", "{}", None)]
    #[case::deleted_keyword_without_default("{}", "{'system': 'caller'}", "{}", None)]
    #[case::rewritten_keyword_wins(
        "{'system': None}",
        "{'system': 'caller'}",
        "{'system': 'hook'}",
        Some("hook")
    )]
    #[case::signature_default_survives("{'system': 'default'}", "{}", "{}", Some("default"))]
    fn request_view_reads_the_hook_rewritten_keywords_over_the_base(
        #[case] base: &str,
        #[case] kwargs: &str,
        #[case] prepared: &str,
        #[case] expected: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let mut view = super::RequestView::new(&dict(py, base), &dict(py, kwargs)).unwrap();
            let prepared = dict(py, prepared);
            view.resolve(py, &prepared).unwrap();
            let value = view
                .argument(py, &prepared, "system")
                .unwrap()
                .map(|value| value.extract::<String>().unwrap());
            assert_eq!(value.as_deref(), expected);
        });
    }

    #[rstest]
    #[case::sync("transcription")]
    #[case::asynchronous("atranscription")]
    fn route_arguments_that_fail_to_convert_raise_value_error(#[case] name: &str) {
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

            let error = module
                .getattr(name)
                .and_then(|function| function.call1((value_call(py, "audio", &broken, None),)))
                .expect_err("route should reject a value it cannot convert");

            assert!(
                error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                "{name} surfaced {error} instead of ValueError"
            );
        });
    }

    #[rstest]
    fn sync_and_async_routes_apply_the_same_input_validation() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);

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

    #[rstest]
    fn missing_and_explicit_none_optional_params_share_the_next_error() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let audio = PyDict::new(py);
            let transcribe = |optional_params: Option<Bound<'_, PyAny>>| {
                let kwargs = PyDict::new(py);
                if let Some(optional_params) = optional_params {
                    kwargs.set_item("optional_params", optional_params).unwrap();
                }
                module
                    .getattr("transcription")
                    .and_then(|function| {
                        function.call1((value_call(py, "audio", &audio, Some(&kwargs)),))
                    })
                    .expect_err("model 'model' has no provider")
                    .to_string()
            };

            let omitted = transcribe(None);
            assert_eq!(transcribe(Some(py.None().into_bound(py))), omitted);
            assert_ne!(omitted, "ValueError: optional_params must be a dict");
            assert_eq!(
                transcribe(Some(PyList::empty(py).into_any())),
                "ValueError: optional_params must be a dict"
            );
        });
    }

    #[rstest]
    fn route_input_validation_preserves_left_to_right_order() {
        Python::initialize();
        Python::attach(|py| {
            let module = crate::native_module(py);
            let invalid = PyList::empty(py);

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
}
