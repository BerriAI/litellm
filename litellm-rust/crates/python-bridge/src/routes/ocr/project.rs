use litellm_auth::SecretValue;
use litellm_host_python::from_py;
use litellm_inference_ocr::{
    types::{LiteLLMOcrRequest, OcrDocumentInput},
    wire::{OcrWireRequest, consumed_optional_param_names, decode_document, decode_request_input},
};
use litellm_llms::base_llm::ocr::error::Error;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

use super::{document::FileDocumentInput, errors::to_pyerr as ocr_error_to_pyerr};
use crate::{
    credentials::{self, CallerTokenProvider},
    marshal::{project_optional_fields, request_input_sources},
    routes::{
        codec::connection_options,
        parameters::{connection_fields, field, merged_request},
    },
};

const ROUTE_HOST_MODULE: &str = "litellm.rust_bridge.ocr.route_host";

/// What the host keeps after projection: the caller's token callable that answers the
/// token operation, and the provider name the failure mapping reports.
pub(super) struct OcrHostHandles {
    pub azure_ad_token_provider: Option<CallerTokenProvider>,
    pub provider: &'static str,
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
    request: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<(LiteLLMOcrRequest<OcrDocumentInput>, OcrHostHandles)> {
    let py = request.py();
    let request = merged_request(request.cast::<PyDict>()?, kwargs)?;
    let options = connection_options(py, ROUTE_HOST_MODULE, &request)?;
    let document = ProjectedDocument::project(
        &field(&request, "document")?
            .ok_or_else(|| PyValueError::new_err("document is required"))?,
    )?;
    let names =
        consumed_optional_param_names(&options.model, options.custom_llm_provider.as_deref())
            .map_err(ocr_error_to_pyerr)?;
    let optional_params: Map<String, Value> =
        project_optional_fields(names.iter().copied(), |name| field(&request, name))?
            .into_iter()
            .chain(connection_fields(&request)?)
            .collect();
    let input_sources = request_input_sources(
        &request,
        optional_params
            .keys()
            .map(String::as_str)
            .chain(["api_key", "api_base", "extra_headers"]),
    )?;
    let azure_ad_token_provider = credentials::azure_ad_token_provider(&request)?;
    let wire = OcrWireRequest {
        model: options.model,
        document: document.resolve(py)?,
        api_key: options.api_key.map(SecretValue::new),
        api_base: options.api_base,
        custom_llm_provider: options.custom_llm_provider,
        extra_headers: options.extra_headers,
        optional_params,
        input_sources,
        timeout_seconds: options.timeout.map(|timeout| timeout.as_secs_f64()),
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

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::ocr::OcrDocument;
    use pyo3::exceptions::PyValueError;

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
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

    fn stub_python_modules(py: Python<'_>) {
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
sys.modules['litellm.rust_bridge.ocr.route_host'] = types.ModuleType('litellm.rust_bridge.ocr.route_host')
",
        );
    }

    fn project(
        py: Python<'_>,
        bound: &str,
        hooked: &str,
    ) -> PyResult<LiteLLMOcrRequest<OcrDocumentInput>> {
        stub_python_modules(py);
        let locals = PyDict::new(py);
        locals
            .set_item(
                "base",
                py.eval(
                    c"{'model': 'mistral/mistral-ocr-latest', 'document': {'type': 'document_url', 'document_url': 'https://doc.example.com/a.pdf'}}",
                    None,
                    None,
                )
                .unwrap(),
            )
            .unwrap();
        let eval_dict = |literal: &str| {
            py.eval(
                &std::ffi::CString::new(literal).unwrap(),
                Some(&locals),
                Some(&locals),
            )
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap()
        };
        let request = eval_dict(&format!("{{**base, **{bound}}}"));
        project_request(request.as_any(), &eval_dict(hooked)).map(|(request, _)| request)
    }

    fn api_base(request: &LiteLLMOcrRequest<OcrDocumentInput>) -> Option<&str> {
        request
            .credentials
            .api_base
            .as_ref()
            .map(|base| base.value().as_str())
    }

    #[rstest::rstest]
    #[case::bound_value(
        "{'api_base': 'https://bound.example.com'}",
        "{}",
        Some("https://bound.example.com")
    )]
    #[case::hook_rewrite_wins(
        "{'api_base': 'https://bound.example.com'}",
        "{'api_base': 'https://hooked.example.com'}",
        Some("https://hooked.example.com")
    )]
    #[case::hook_none_clears_a_bound_value(
        "{'api_base': 'https://bound.example.com'}",
        "{'api_base': None}",
        None
    )]
    #[case::base_url_is_the_fallback(
        "{'base_url': 'https://base-url.example.com'}",
        "{}",
        Some("https://base-url.example.com")
    )]
    #[case::empty_api_base_is_absent(
        "{'api_base': '', 'base_url': 'https://base-url.example.com'}",
        "{}",
        Some("https://base-url.example.com")
    )]
    fn connection_options_come_from_the_shared_request_view(
        #[case] bound: &str,
        #[case] hooked: &str,
        #[case] expected: Option<&str>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(api_base(&project(py, bound, hooked).unwrap()), expected);
        });
    }

    #[rstest::rstest]
    #[case::empty_string("''", false)]
    #[case::none("None", false)]
    #[case::value("'sk-ocr'", true)]
    fn only_a_non_empty_api_key_is_a_caller_key(#[case] api_key: &str, #[case] present: bool) {
        Python::initialize();
        Python::attach(|py| {
            let request = project(py, &format!("{{'api_key': {api_key}}}"), "{}").unwrap();
            assert_eq!(request.credentials.api_key.is_some(), present);
        });
    }

    #[test]
    fn a_none_optional_param_is_absent() {
        Python::initialize();
        Python::attach(|py| {
            let request = project(py, "{'pages': [0]}", "{'pages': None}").unwrap();
            assert!(!request.optional_params.contains_key("pages"));
            let request = project(py, "{}", "{'pages': [0]}").unwrap();
            assert_eq!(
                request.optional_params.get("pages"),
                Some(&serde_json::json!([0]))
            );
        });
    }

    #[rstest::rstest]
    #[case::missing_model("{'model': None}", "model is required")]
    #[case::missing_document("{'document': None}", "document is required")]
    fn missing_required_arguments_are_value_errors(#[case] bound: &str, #[case] message: &str) {
        Python::initialize();
        Python::attach(|py| {
            let Err(error) = project(py, bound, "{}") else {
                panic!("projection accepted a request without {message}");
            };
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains(message), "{error}");
        });
    }

    /// A reader that rewrites the request while it runs shows which arguments projection
    /// read before it and which after: every other argument is read first, and the read
    /// happens exactly once.
    #[test]
    fn document_readers_are_read_once_after_every_other_argument() {
        Python::initialize();
        Python::attach(|py| {
            stub_python_modules(py);
            let locals = eval(
                py,
                c"
class Reader:
    reads = 0
    def read(self):
        Reader.reads += 1
        request['api_base'] = 'https://mutated.example.com'
        request['extra_headers'] = {'x-source': 'mutated'}
        request['timeout'] = 9
        return b'abc'
request = {
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
            let request = locals.get_item("request").unwrap().unwrap();
            let kwargs = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let (projected, _) = project_request(&request, &kwargs).unwrap();
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
            assert_eq!(api_base(&projected), Some("https://original.example.com"));
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

    fn request_and_kwargs<'py>(
        py: Python<'py>,
        kwargs: &std::ffi::CStr,
    ) -> (Bound<'py, PyAny>, Bound<'py, PyDict>) {
        let locals = eval(
            py,
            c"
request = {
    'model': 'mistral/mistral-ocr-latest',
    'custom_llm_provider': 'mistral',
    'document': {'type': 'document_url', 'document_url': 'https://example.com/request.pdf'},
    'api_key': None,
    'api_base': 'https://request.example.com',
    'extra_headers': {'x-source': 'request'},
    'timeout': 1,
}
",
        );
        py.run(kwargs, Some(&locals), Some(&locals)).unwrap();
        (
            locals.get_item("request").unwrap().unwrap(),
            locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap(),
        )
    }

    #[test]
    fn unconsumed_kwargs_stay_out_of_optional_params_and_response_limit_goes_to_transport() {
        Python::initialize();
        Python::attach(|py| {
            stub_python_modules(py);
            let (request, kwargs) = request_and_kwargs(
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
            let (projected, _) = project_request(&request, &kwargs).unwrap();
            assert_eq!(
                projected.optional_params.keys().collect::<Vec<_>>(),
                ["pages"]
            );
            assert_eq!(projected.transport.max_response_bytes, 1234);
        });
    }

    #[test]
    fn replacement_kwargs_project_provider_connection_and_timeout() {
        Python::initialize();
        Python::attach(|py| {
            stub_python_modules(py);
            let (request, kwargs) = request_and_kwargs(
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
            let (projected, handles) = project_request(&request, &kwargs).unwrap();
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
