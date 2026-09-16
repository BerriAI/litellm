use std::sync::Arc;

use litellm_core::ocr::wire::{OcrWireRequest, consumed_optional_params, decode_request};
use litellm_core::ocr::{LiteLLMOcrRequest, NativeOutcome, OcrCall};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::lifecycle::BridgeOcrHooks;
use crate::auth::{AZURE_AD_TOKEN_PROVIDER, PythonTokenProvider};
use crate::errors::RustBridgeDeclined;
use crate::marshal::{project_optional_fields, python_timeout_seconds, request_input_sources};

pub(super) struct ProjectedOcrFields {
    pub boundary_request: Py<PyAny>,
    pub document: Py<PyAny>,
    pub api_key: Py<PyAny>,
    pub azure_ad_token_provider: Option<PythonTokenProvider>,
    pub provider: &'static str,
    pub secret_fields: Vec<&'static str>,
}

pub(super) struct ProjectedOcrCall {
    pub request: LiteLLMOcrRequest,
    pub fields: ProjectedOcrFields,
}

struct OcrArguments<'a, 'py> {
    request: &'a Bound<'py, PyAny>,
    kwargs: &'a Bound<'py, PyDict>,
}

impl<'py> OcrArguments<'_, 'py> {
    fn lookup(&self, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.kwargs.get_item(name)? {
            Some(value) => Ok(value),
            None => self.request.getattr(name),
        }
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

    fn api_key(&self) -> PyResult<Bound<'py, PyAny>> {
        self.lookup("api_key")
    }

    fn api_base(&self) -> PyResult<Option<String>> {
        self.lookup("api_base")?.extract()
    }

    fn extra_headers(&self) -> PyResult<Option<Map<String, Value>>> {
        self.lookup("extra_headers")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| from_py(value.bind(self.request.py())))
            .transpose()
    }

    fn timeout_seconds(&self) -> PyResult<Option<f64>> {
        Ok(self
            .lookup("timeout")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| python_timeout_seconds(self.request.py(), value))
            .transpose()?
            .flatten())
    }
}

enum ProjectedDocument {
    File { wire: Value, retained: Py<PyAny> },
    Other { wire: Value, retained: Py<PyAny> },
}

impl ProjectedDocument {
    fn project(py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<Self> {
        let kind: String = document.get_item("type")?.extract()?;
        if kind != "file" {
            return Ok(Self::Other {
                wire: from_py(document)?,
                retained: document.clone().unbind(),
            });
        }
        let input = document.extract()?;
        let encoded = super::document::file_document(py, input)?;
        let wire = serde_json::to_value(encoded)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self::File {
            retained: to_py(py, &wire)?,
            wire,
        })
    }

    fn into_parts(self) -> (Value, Py<PyAny>) {
        match self {
            Self::File { wire, retained } | Self::Other { wire, retained } => (wire, retained),
        }
    }
}

pub(super) fn project_request(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<ProjectedOcrCall> {
    let boundary_request = request.clone().unbind();
    let arguments = OcrArguments { request, kwargs };
    let model = arguments.model()?;
    let custom_llm_provider = arguments.custom_llm_provider()?;
    let (wire_document, retained_document) =
        ProjectedDocument::project(py, &arguments.document()?)?.into_parts();
    let api_key = arguments.api_key()?;
    let specs = consumed_optional_params(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let names = specs.iter().map(|spec| spec.name).collect::<Vec<_>>();
    let optional_params = project_optional_fields(kwargs, &names)?;
    let input_sources = request_input_sources(
        kwargs,
        names
            .iter()
            .copied()
            .chain(["api_key", "api_base", "extra_headers"]),
    )?;
    let azure_ad_token_provider = kwargs
        .get_item("azure_ad_token_provider")?
        .and_then(|provider| PythonTokenProvider::select(provider, AZURE_AD_TOKEN_PROVIDER));
    let wire = OcrWireRequest {
        model,
        document: wire_document,
        api_key: api_key.extract()?,
        api_base: arguments.api_base()?,
        custom_llm_provider,
        extra_headers: arguments.extra_headers()?,
        optional_params,
        input_sources,
        timeout_seconds: arguments.timeout_seconds()?,
    };
    let request = decode_request(wire).map_err(ocr_error_to_pyerr)?;
    let provider = request.provider_name();
    Ok(ProjectedOcrCall {
        request: request.with_host_hooks(Arc::new(BridgeOcrHooks), None),
        fields: ProjectedOcrFields {
            boundary_request,
            document: retained_document,
            api_key: api_key.unbind(),
            azure_ad_token_provider,
            provider,
            secret_fields: specs
                .into_iter()
                .filter(|spec| spec.secret)
                .map(|spec| spec.name)
                .collect(),
        },
    })
}

pub(super) fn admitted_call(outcome: NativeOutcome<OcrCall>) -> PyResult<OcrCall> {
    match outcome {
        NativeOutcome::Completed(call) => Ok(call),
        NativeOutcome::Declined(reason) => Err(RustBridgeDeclined::new_err(format!(
            "native OCR admission declined: {reason:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use litellm_core::Error;
    use litellm_core::ocr::OcrDecline;
    use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn arguments<'a, 'py>(
        request: &'a Bound<'py, PyAny>,
        kwargs: &'a Bound<'py, PyDict>,
    ) -> OcrArguments<'a, 'py> {
        OcrArguments { request, kwargs }
    }

    fn project_document(
        py: Python<'_>,
        document: &Bound<'_, PyAny>,
    ) -> PyResult<(Value, Py<PyAny>)> {
        ProjectedDocument::project(py, document).map(ProjectedDocument::into_parts)
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
    fn typed_initial_decline_uses_bridge_decline_contract() {
        Python::initialize();
        Python::attach(|py| {
            let Err(error) = admitted_call(NativeOutcome::Declined(OcrDecline::HostOperations))
            else {
                panic!("unsupported host operations should decline admission");
            };
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn post_admission_error_does_not_use_bridge_decline_contract() {
        Python::initialize();
        Python::attach(|py| {
            let error = ocr_error_to_pyerr(Error::InvalidRequest("callback result".into()));
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(!error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn kwargs_override_request_attributes_including_explicit_none() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Request:
    def __init__(self):
        self.accesses = []
    def __getattribute__(self, name):
        if name != 'accesses':
            object.__getattribute__(self, 'accesses').append(name)
        return object.__getattribute__(self, name)
request = Request()
request.model = 'from-request'
request.custom_llm_provider = 'mistral'
kwargs = {'model': 'from-kwargs', 'custom_llm_provider': None}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let arguments = arguments(&request, &kwargs);
            assert_eq!(arguments.model().unwrap(), "from-kwargs");
            assert_eq!(arguments.custom_llm_provider().unwrap(), None);
            let accesses: Vec<String> = request.getattr("accesses").unwrap().extract().unwrap();
            assert_eq!(accesses, Vec::<String>::new());
        });
    }

    #[test]
    fn missing_kwargs_read_the_request_property_once() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Request:
    def __init__(self):
        self.reads = 0
    @property
    def model(self):
        self.reads += 1
        return 'mistral-ocr-latest'
request = Request()
kwargs = {}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            assert_eq!(
                arguments(&request, &kwargs).model().unwrap(),
                "mistral-ocr-latest"
            );
            assert_eq!(
                request.getattr("reads").unwrap().extract::<i32>().unwrap(),
                1
            );
        });
    }

    #[test]
    fn request_property_exceptions_keep_their_identity() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
failure = LookupError('model failed')
class Request:
    @property
    def model(self):
        raise failure
request = Request()
kwargs = {}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let error = arguments(&request, &kwargs).model().unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn unused_raising_property_is_never_inspected() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Request:
    @property
    def unused(self):
        raise RuntimeError('unused')
    model = 'mistral-ocr-latest'
    custom_llm_provider = None
request = Request()
kwargs = {}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let arguments = arguments(&request, &kwargs);
            assert_eq!(arguments.model().unwrap(), "mistral-ocr-latest");
            assert_eq!(arguments.custom_llm_provider().unwrap(), None);
        });
    }

    #[test]
    fn document_reader_mutations_are_visible_to_later_field_reads() {
        Python::initialize();
        Python::attach(|py| {
            stub_timeout_conversion(py);
            let locals = eval(
                py,
                c"
class Request:
    api_base = 'original'
    timeout = 1
    @property
    def document(self):
        return document
class Reader:
    def read(self):
        Request.api_base = 'mutated'
        Request.timeout = 9
        return b'abc'
document = {'type': 'file', 'file': Reader()}
request = Request()
kwargs = {}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let arguments = arguments(&request, &kwargs);
            let document = arguments.document().unwrap();
            project_document(py, &document).unwrap();
            assert_eq!(arguments.api_base().unwrap().as_deref(), Some("mutated"));
            assert_eq!(arguments.timeout_seconds().unwrap(), Some(9.0));
        });
    }

    #[test]
    fn captured_api_key_keeps_the_original_python_object() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
key = object()
class Request:
    api_key = None
request = Request()
kwargs = {'api_key': key}
",
            );
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let captured = arguments(&request, &kwargs).api_key().unwrap();
            assert!(
                captured
                    .unbind()
                    .bind(py)
                    .is(locals.get_item("key").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn file_documents_are_encoded_and_other_documents_keep_the_python_object() {
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
                project_document(py, &file).unwrap().0,
                serde_json::json!({
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64,JVBERi0xLjQ=",
                })
            );

            let original = py
                .eval(
                    c"{'type': 'document_url', 'document_url': 'https://example.com/a.pdf'}",
                    None,
                    None,
                )
                .unwrap();
            let (wire, retained) = project_document(py, &original).unwrap();
            assert_eq!(
                wire,
                serde_json::json!({
                    "type": "document_url",
                    "document_url": "https://example.com/a.pdf",
                })
            );
            assert!(retained.bind(py).is(&original));
        });
    }

    #[test]
    fn unknown_document_types_reach_existing_downstream_validation() {
        Python::initialize();
        Python::attach(|py| {
            let document = py
                .eval(c"{'type': 'mystery', 'mystery': 'x'}", None, None)
                .unwrap();
            let wire_document = project_document(py, &document).unwrap().0;
            assert_eq!(
                wire_document,
                serde_json::json!({"type": "mystery", "mystery": "x"})
            );
            let error = match decode_request(OcrWireRequest {
                model: "mistral/mistral-ocr-latest".into(),
                document: wire_document,
                api_key: None,
                api_base: None,
                custom_llm_provider: None,
                extra_headers: None,
                optional_params: Map::new(),
                input_sources: Default::default(),
                timeout_seconds: None,
            }) {
                Ok(_) => panic!("unknown discriminators belong to core validation"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("document"));
        });
    }

    #[test]
    fn document_discriminator_errors_keep_their_existing_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let missing = py.eval(c"{}", None, None).unwrap();
            assert!(
                project_document(py, &missing)
                    .unwrap_err()
                    .is_instance_of::<PyKeyError>(py)
            );

            let non_string = py.eval(c"{'type': 1}", None, None).unwrap();
            assert!(
                project_document(py, &non_string)
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
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
                project_document(py, &locals.get_item("document").unwrap().unwrap()).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
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
            let (wire, retained) = project_document(py, &document).unwrap();
            assert_eq!(wire["type"], "document_url");
            assert!(!retained.bind(py).is(&document));
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type", "mime_type", "file"]);
        });
    }
}
