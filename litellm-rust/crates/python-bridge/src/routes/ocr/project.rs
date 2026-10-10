use litellm_auth::SecretValue;
use litellm_auth::VertexParams;
use litellm_host_python::{from_py, present};
use litellm_inference_ocr::{
    types::{LiteLLMOcrRequest, OcrDocumentInput},
    wire::{OcrWireRequest, consumed_optional_params, decode_document, decode_request_input},
};
use litellm_llms::base_llm::ocr::error::Error;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

use super::{document::FileDocumentInput, errors::to_pyerr as ocr_error_to_pyerr};
use crate::{
    coercion::FieldSpec,
    credentials::{self, CallerTokenProvider},
    marshal::{project_optional_fields, python_timeout_seconds, request_input_sources},
    python_settings::{PythonSettings, Snapshot},
};

/// What the host keeps after projection: the caller's token callable that answers the
/// token operation, and the provider name the failure mapping reports.
pub(super) struct OcrHostHandles {
    pub azure_ad_token_provider: Option<CallerTokenProvider>,
    pub provider: &'static str,
}

struct OcrArguments<'a, 'py> {
    bound: &'a Bound<'py, PyDict>,
    kwargs: &'a Bound<'py, PyDict>,
}

impl<'py> OcrArguments<'_, 'py> {
    fn lookup(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        litellm_host_python::lookup(self.kwargs, self.bound, name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing argument: {name}")))
    }

    fn model(&self) -> PyResult<String> {
        self.lookup("model")?.extract()
    }

    fn custom_llm_provider(&self) -> PyResult<Option<String>> {
        self.lookup("custom_llm_provider")?.extract()
    }

    fn document(&self) -> PyResult<Bound<'py, PyAny>> {
        self.lookup("document")
    }

    fn api_key(&self) -> PyResult<Option<SecretValue>> {
        Ok(self
            .lookup("api_key")?
            .extract::<Option<String>>()?
            .map(SecretValue::new))
    }

    fn api_base(&self) -> PyResult<Option<String>> {
        self.lookup("api_base")?.extract()
    }

    fn extra_headers(&self) -> PyResult<Option<Map<String, Value>>> {
        self.lookup("extra_headers")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| from_py(value.bind(self.bound.py())))
            .transpose()
    }

    fn timeout_seconds(&self) -> PyResult<Option<f64>> {
        Ok(self
            .lookup("timeout")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| python_timeout_seconds(self.bound.py(), value))
            .transpose()?
            .flatten())
    }
}

enum ProjectedDocument {
    File(FileDocumentInput),
    Other(Value),
}

impl ProjectedDocument {
    fn project(document: &Bound<'_, PyAny>) -> PyResult<Self> {
        let kind: String = document
            .get_item("type")
            .and_then(|value| value.extract())
            .map_err(|error| {
                let py = document.py();
                if error.is_instance_of::<pyo3::exceptions::PyKeyError>(py)
                    || error.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                {
                    ocr_error_to_pyerr(Error::RequestField {
                        path: "document.type".into(),
                    })
                } else {
                    error
                }
            })?;
        if kind != "file" {
            return Ok(Self::Other(from_py(document)?));
        }
        Ok(Self::File(document.extract()?))
    }

    /// Reads a file-like document now, so it runs after every other argument was read.
    fn resolve(self, py: Python<'_>) -> PyResult<OcrDocumentInput> {
        match self {
            Self::File(file) => file.resolve(py),
            Self::Other(wire) => Ok(decode_document(wire).map_err(ocr_error_to_pyerr)?.into()),
        }
    }
}

pub(super) fn project_request(
    bound: &Bound<'_, PyDict>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<(LiteLLMOcrRequest<OcrDocumentInput>, OcrHostHandles)> {
    let arguments = OcrArguments { bound, kwargs };
    let model = arguments.model()?;
    let custom_llm_provider = arguments.custom_llm_provider()?;
    let document = ProjectedDocument::project(&arguments.document()?)?;
    let api_key = arguments.api_key()?;
    let specs = consumed_optional_params(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let names = specs.iter().map(|spec| spec.name).collect::<Vec<_>>();
    let optional_params =
        project_optional_fields(names.iter().copied(), |name| present(kwargs, bound, name))?;
    let globals = module_globals_to_read(&optional_params, &names);
    let defaults = if globals.is_empty() {
        None
    } else {
        PythonSettings::ProviderDefaults.read_or_unset(bound.py())?
    };
    let optional_params = match defaults {
        Some(defaults) => fold_module_globals(bound.py(), optional_params, &globals, &defaults)?,
        None => optional_params,
    };
    let input_sources = request_input_sources(
        kwargs,
        names
            .iter()
            .copied()
            .chain(["api_key", "api_base", "extra_headers"]),
    )?;
    let azure_ad_token_provider = credentials::azure_ad_token_provider(kwargs)?;
    let api_base = arguments.api_base()?;
    let extra_headers = arguments.extra_headers()?;
    let timeout_seconds = arguments.timeout_seconds()?;
    let wire = OcrWireRequest {
        model,
        document: document.resolve(bound.py())?,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        input_sources,
        timeout_seconds,
    };
    let request = decode_request_input(wire).map_err(ocr_error_to_pyerr)?;
    let provider = request.provider_name();
    Ok((
        request,
        OcrHostHandles {
            azure_ad_token_provider,
            provider,
        },
    ))
}

/// Python's `VertexBase.get_vertex_ai_project` and `get_vertex_ai_location` fall back to a
/// `litellm.<name>` module global when a call names neither spelling: the globals of the
/// specs this route consumes whose wire names the call left out.
fn module_globals_to_read(
    optional_params: &Map<String, Value>,
    consumed: &[&str],
) -> Vec<&'static str> {
    VertexParams::SPECS
        .iter()
        .filter(|spec| spec.wire.iter().all(|name| consumed.contains(name)))
        .filter(|spec| {
            !spec
                .wire
                .iter()
                .any(|name| optional_params.contains_key(*name))
        })
        .filter_map(|spec| spec.module_global)
        .collect()
}

/// Folds each module global in under the name its spec declares, skipping falsy values the
/// way Python's `or` chain does.
fn fold_module_globals(
    py: Python<'_>,
    optional_params: Map<String, Value>,
    globals: &[&'static str],
    defaults: &Snapshot<'_>,
) -> PyResult<Map<String, Value>> {
    let read = globals
        .iter()
        .map(|name| {
            let spec = FieldSpec::new(name, |field| field.falsy_optional_string());
            Ok(defaults
                .read(&spec)
                .map_err(|error| settings_error(py, error.into()))?
                .map(|value| (name.to_string(), Value::String(value))))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(optional_params
        .into_iter()
        .chain(read.into_iter().flatten())
        .collect())
}

/// A misconfigured litellm setting is the operator's error, not the request's or the
/// provider's, so the host raises it as is instead of mapping it onto a provider failure.
pub(super) const SETTINGS_ERROR_MARKER: &str = "native_settings_error";

fn settings_error(py: Python<'_>, error: PyErr) -> PyErr {
    error.value(py).setattr(SETTINGS_ERROR_MARKER, true).ok();
    error
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::ocr::OcrDocument;
    use pyo3::exceptions::PyValueError;

    use super::*;

    #[rstest::rstest]
    #[case::global_fills_a_missing_project(
        serde_json::json!({"vertex_location": "us-east5"}),
        "vertex_project='from-global', vertex_location='ignored'",
        serde_json::json!({"vertex_location": "us-east5", "vertex_project": "from-global"}),
    )]
    #[case::either_spelling_on_the_call_wins(
        serde_json::json!({"vertex_ai_project": "from-call"}),
        "vertex_project='from-global', vertex_location=None",
        serde_json::json!({"vertex_ai_project": "from-call"}),
    )]
    #[case::falsy_globals_are_absent(
        serde_json::json!({}),
        "vertex_project=[], vertex_location=''",
        serde_json::json!({}),
    )]
    fn module_globals_fill_the_vertex_params_the_call_leaves_out(
        #[case] params: Value,
        #[case] defaults: &str,
        #[case] expected: Value,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let namespace = py
                .eval(
                    &std::ffi::CString::new(format!(
                        "__import__('types').SimpleNamespace({defaults})"
                    ))
                    .unwrap(),
                    None,
                    None,
                )
                .unwrap();
            let snapshot = PythonSettings::ProviderDefaults.snapshot(namespace);
            let consumed = VertexParams::fields().collect::<Vec<_>>();
            let params = params.as_object().cloned().unwrap();
            let globals = module_globals_to_read(&params, &consumed);

            let folded = fold_module_globals(py, params, &globals, &snapshot).unwrap();

            assert_eq!(Value::Object(folded), expected);
        });
    }

    #[rstest::rstest]
    fn a_bad_global_is_a_marked_settings_error() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = py
                .eval(
                    c"__import__('types').SimpleNamespace(vertex_project=1)",
                    None,
                    None,
                )
                .unwrap();
            let snapshot = PythonSettings::ProviderDefaults.snapshot(namespace);

            let error =
                fold_module_globals(py, Map::new(), &["vertex_project"], &snapshot).unwrap_err();

            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(
                error
                    .to_string()
                    .contains("provider_defaults.vertex_project")
            );
            assert!(
                error
                    .value(py)
                    .getattr_opt(SETTINGS_ERROR_MARKER)
                    .unwrap()
                    .is_some()
            );
        });
    }

    #[rstest::rstest]
    #[case::not_consumed(&["pages"], serde_json::json!({}), &[])]
    #[case::consumed_and_absent(
        &["vertex_project", "vertex_ai_project", "vertex_location", "vertex_ai_location"],
        serde_json::json!({}),
        &["vertex_project", "vertex_location"]
    )]
    #[case::consumed_and_present_in_either_spelling(
        &["vertex_project", "vertex_ai_project", "vertex_location", "vertex_ai_location"],
        serde_json::json!({"vertex_ai_project": "p"}),
        &["vertex_location"]
    )]
    fn only_the_globals_of_consumed_absent_params_are_read(
        #[case] consumed: &[&str],
        #[case] params: Value,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            module_globals_to_read(params.as_object().unwrap(), consumed),
            expected
        );
    }

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn dict<'py>(locals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyDict> {
        locals
            .get_item(name)
            .unwrap()
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
    }

    fn arguments<'a, 'py>(
        bound: &'a Bound<'py, PyDict>,
        kwargs: &'a Bound<'py, PyDict>,
    ) -> OcrArguments<'a, 'py> {
        OcrArguments { bound, kwargs }
    }

    fn project_document(document: &Bound<'_, PyAny>) -> PyResult<OcrDocumentInput> {
        ProjectedDocument::project(document)?.resolve(document.py())
    }

    fn url_document(url: &str) -> OcrDocumentInput {
        OcrDocument::DocumentUrl {
            document_url: url.into(),
            extra_fields: Default::default(),
        }
        .into()
    }

    fn stub_timeout_conversion(py: Python<'_>) {
        eval(
            py,
            c"
import sys
import types
timeouts = types.ModuleType('litellm.rust_bridge.timeouts')
timeouts.timeout_to_seconds = lambda timeout: None if timeout is None else float(timeout)
sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm.rust_bridge', types.ModuleType('litellm.rust_bridge'))
sys.modules['litellm.rust_bridge.timeouts'] = timeouts
",
        );
    }

    #[test]
    fn kwargs_override_bound_values_including_explicit_none() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
bound = {'model': 'from-bound', 'custom_llm_provider': 'mistral'}
kwargs = {'model': 'from-kwargs', 'custom_llm_provider': None}
",
            );
            let (bound, kwargs) = (dict(&locals, "bound"), dict(&locals, "kwargs"));
            let arguments = arguments(&bound, &kwargs);
            assert_eq!(arguments.model().unwrap(), "from-kwargs");
            assert_eq!(arguments.custom_llm_provider().unwrap(), None);
        });
    }

    /// A reader that rewrites the bound arguments while it runs shows which arguments
    /// projection read before it and which after: every other argument is read first, and
    /// the read happens exactly once.
    #[test]
    fn document_readers_are_read_once_after_every_other_argument() {
        Python::initialize();
        Python::attach(|py| {
            stub_timeout_conversion(py);
            let locals = eval(
                py,
                c"
class Reader:
    reads = 0
    def read(self):
        Reader.reads += 1
        bound['api_base'] = 'https://mutated.example.com'
        bound['extra_headers'] = {'x-source': 'mutated'}
        bound['timeout'] = 9
        return b'abc'
bound = {
    'model': 'mistral/mistral-ocr-latest',
    'custom_llm_provider': None,
    'api_key': None,
    'api_base': 'https://original.example.com',
    'extra_headers': {'x-source': 'original'},
    'timeout': 1,
    'document': {'type': 'file', 'file': Reader(), 'mime_type': 'application/pdf'},
}
kwargs = {}
",
            );
            let (projected, _) =
                project_request(&dict(&locals, "bound"), &dict(&locals, "kwargs")).unwrap();
            assert_eq!(
                py.eval(c"Reader.reads", Some(&locals), Some(&locals))
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                projected.document,
                OcrDocumentInput::Bytes {
                    bytes: b"abc".as_slice().into(),
                    file_name: None,
                    mime_type: Some("application/pdf".into()),
                }
            );
            assert_eq!(
                projected
                    .credentials
                    .api_base
                    .as_ref()
                    .map(|base| base.value().as_str()),
                Some("https://original.example.com")
            );
            assert_eq!(
                projected.transport.extra_headers,
                [("x-source".to_string(), "original".to_string())]
            );
            assert_eq!(
                projected.transport.timeout,
                Some(std::time::Duration::from_secs(1))
            );
        });
    }

    #[test]
    fn file_documents_become_typed_inputs_and_other_documents_decode() {
        Python::initialize();
        Python::attach(|py| {
            let file = py
                .eval(
                    c"{'type': 'file', 'file': b'%PDF-1.4', 'mime_type': 'application/pdf'}",
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(
                project_document(&file).unwrap(),
                OcrDocumentInput::Bytes {
                    bytes: b"%PDF-1.4".as_slice().into(),
                    file_name: None,
                    mime_type: Some("application/pdf".into()),
                }
            );

            let original = py
                .eval(
                    c"{'type': 'document_url', 'document_url': 'https://example.com/a.pdf'}",
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(
                project_document(&original).unwrap(),
                url_document("https://example.com/a.pdf")
            );
        });
    }

    #[test]
    fn unknown_document_types_reach_existing_core_validation() {
        Python::initialize();
        Python::attach(|py| {
            let document = py
                .eval(c"{'type': 'mystery', 'mystery': 'x'}", None, None)
                .unwrap();
            let error = project_document(&document).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("document"));
        });
    }

    #[test]
    fn document_discriminator_errors_are_validation_errors_and_preserve_custom_failures() {
        Python::initialize();
        Python::attach(|py| {
            let missing = py.eval(c"{}", None, None).unwrap();
            assert!(
                project_document(&missing)
                    .unwrap_err()
                    .is_instance_of::<PyValueError>(py)
            );

            let non_string = py.eval(c"{'type': 1}", None, None).unwrap();
            assert!(
                project_document(&non_string)
                    .unwrap_err()
                    .is_instance_of::<PyValueError>(py)
            );

            let locals = eval(
                py,
                c"
failure = RuntimeError('type lookup failed')
class Document:
    def __getitem__(self, key):
        raise failure
document = Document()
",
            );
            let error =
                project_document(&locals.get_item("document").unwrap().unwrap()).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[rstest::rstest]
    #[case::missing(c"{}")]
    #[case::non_string(c"{'type': 1}")]
    #[case::list(c"[]")]
    fn malformed_document_discriminators_are_bad_requests_naming_the_field(
        #[case] document: &std::ffi::CStr,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = project_document(&py.eval(document, None, None).unwrap()).unwrap_err();
            let value = error.value(py);
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                value.to_string(),
                "invalid OCR request field: document.type"
            );
            assert_eq!(
                value
                    .getattr("status_code")
                    .unwrap()
                    .extract::<u16>()
                    .unwrap(),
                400
            );
        });
    }

    fn bound_and_kwargs<'py>(
        py: Python<'py>,
        kwargs: &std::ffi::CStr,
    ) -> (Bound<'py, PyDict>, Bound<'py, PyDict>) {
        let locals = eval(
            py,
            c"
bound = {
    'model': 'mistral/mistral-ocr-latest',
    'custom_llm_provider': 'mistral',
    'document': {'type': 'document_url', 'document_url': 'https://example.com/bound.pdf'},
    'api_key': None,
    'api_base': 'https://bound.example.com',
    'extra_headers': {'x-source': 'bound'},
    'timeout': 1,
}
",
        );
        py.run(kwargs, Some(&locals), Some(&locals)).unwrap();
        (dict(&locals, "bound"), dict(&locals, "kwargs"))
    }

    #[test]
    fn unconsumed_kwargs_stay_out_of_optional_params_and_response_limit_goes_to_transport() {
        Python::initialize();
        Python::attach(|py| {
            stub_timeout_conversion(py);
            let (bound, kwargs) = bound_and_kwargs(
                py,
                c"
kwargs = {
    'model': 'mistral/mistral-ocr-latest',
    'custom_llm_provider': None,
    'pages': [0],
    'max_response_bytes': 1234,
    'metadata': {'user_api_key_auth': 'auth'},
    'ocr_cost_per_page': 0.05,
    'shared_session': object(),
    'guardrails': ['guard'],
    'opaque': object(),
}
",
            );
            let (projected, _) = project_request(&bound, &kwargs).unwrap();
            assert_eq!(
                projected.optional_params.keys().collect::<Vec<_>>(),
                ["pages"]
            );
            assert_eq!(projected.transport.max_response_bytes, 1234);
        });
    }

    #[rstest::rstest]
    #[case::explicit_none_is_unset(c"bound['pages'] = [1]\nkwargs = {'pages': None}", None)]
    #[case::bound_fallback(c"bound['pages'] = [1]\nkwargs = {}", Some(serde_json::json!([1])))]
    #[case::keyword_wins(c"bound['pages'] = [1]\nkwargs = {'pages': [0]}", Some(serde_json::json!([0])))]
    fn optional_params_read_through_bound_and_drop_none(
        #[case] script: &std::ffi::CStr,
        #[case] expected: Option<Value>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            stub_timeout_conversion(py);
            let (bound, kwargs) = bound_and_kwargs(py, script);
            let (projected, _) = project_request(&bound, &kwargs).unwrap();
            assert_eq!(projected.optional_params.get("pages").cloned(), expected);
        });
    }

    #[test]
    fn replacement_kwargs_project_provider_connection_and_timeout() {
        Python::initialize();
        Python::attach(|py| {
            stub_timeout_conversion(py);
            let (bound, kwargs) = bound_and_kwargs(
                py,
                c"
kwargs = {
    'model': 'mistral-ocr-latest',
    'custom_llm_provider': 'azure_ai',
    'document': {'type': 'document_url', 'document_url': 'https://example.com/kwargs.pdf'},
    'api_base': 'https://kwargs.example.com',
    'extra_headers': {'x-source': 'kwargs'},
    'timeout': 5,
}
",
            );
            let (projected, handles) = project_request(&bound, &kwargs).unwrap();
            assert_eq!(handles.provider, "azure_ai");
            assert_eq!(projected.model, "mistral-ocr-latest");
            assert_eq!(
                projected.document,
                url_document("https://example.com/kwargs.pdf")
            );
            assert_eq!(
                projected.credentials.api_base.unwrap().value(),
                "https://kwargs.example.com"
            );
            assert_eq!(
                projected.transport.extra_headers,
                [("x-source".to_string(), "kwargs".to_string())]
            );
            assert_eq!(
                projected.transport.timeout,
                Some(std::time::Duration::from_secs(5))
            );
        });
    }

    #[test]
    fn document_classification_happens_once() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Document(dict):
    def __init__(self):
        super().__init__({'file': b'abc'})
        self.reads = []
    def __getitem__(self, key):
        self.reads.append(key)
        if key == 'type':
            return 'file' if self.reads.count('type') == 1 else 'document_url'
        return super().__getitem__(key)
document = Document()
",
            );
            let document = locals.get_item("document").unwrap().unwrap();
            let input = project_document(&document).unwrap();
            assert!(matches!(input, OcrDocumentInput::Bytes { .. }));
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type", "mime_type", "file"]);
        });
    }
}
