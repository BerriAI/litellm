use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use litellm_auth::InputSource;
use litellm_core::ocr::{
    LiteLLMOcrRequest, NativeOutcome, OcrCall, OcrCredentialInputs, OcrDocument,
    consumed_optional_params,
};
use litellm_python_interop::from_py_preserving_errors as from_py;

use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::lifecycle::BridgeOcrHooks;
use crate::auth::{AZURE_AD_TOKEN_PROVIDER, PythonTokenProvider};
use crate::errors::RustBridgeDeclined;
use crate::lifecycle::BoundArguments;
use crate::marshal::{project_optional_fields, python_timeout_seconds, request_input_sources};

const BOUND_FIELDS: &[&str] = &["model", "document", "timeout", "input_sources"];

pub(super) struct ProjectedOcrCall {
    pub request: LiteLLMOcrRequest,
    pub azure_ad_token_provider: Option<PythonTokenProvider>,
    pub secret_fields: Vec<&'static str>,
}

fn project_document(py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<Value> {
    let kind: String = document.get_item("type")?.extract()?;
    if kind != "file" {
        return from_py(document);
    }
    let encoded = super::document::file_document(py, document.extract()?)?;
    serde_json::to_value(encoded)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn header_pairs(headers: Option<Map<String, Value>>) -> Result<Vec<(String, String)>, litellm_core::ocr::Error> {
    headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            value
                .as_str()
                .map(|value| (name.clone(), value.to_string()))
                .ok_or_else(|| litellm_core::ocr::Error::RequestField {
                    path: format!("extra_headers.{name}"),
                })
        })
        .collect()
}

fn source_for(sources: &BTreeMap<String, InputSource>, name: &str) -> InputSource {
    sources.get(name).copied().unwrap_or_default()
}

pub(super) fn project(py: Python<'_>, arguments: &BoundArguments<'_>) -> PyResult<ProjectedOcrCall> {
    let kwargs: &Bound<'_, PyDict> = arguments.kwargs();
    let model: String = arguments.extract("model")?;
    let custom_llm_provider: Option<String> = arguments.optional("custom_llm_provider")?;
    let document = project_document(py, &arguments.required("document")?)?;
    let api_key: Option<String> = arguments.optional("api_key")?;
    let specs = consumed_optional_params(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let optional_params = project_optional_fields(kwargs, &specs, BOUND_FIELDS)?;
    let input_sources = request_input_sources(
        kwargs,
        optional_params
            .keys()
            .map(String::as_str)
            .chain(["api_key", "api_base", "extra_headers"]),
    )?;
    let azure_ad_token_provider = kwargs
        .get_item("azure_ad_token_provider")?
        .and_then(|provider| PythonTokenProvider::select(provider, AZURE_AD_TOKEN_PROVIDER));
    let api_base: Option<String> = arguments.optional("api_base")?;
    let extra_headers: Option<Map<String, Value>> = arguments
        .get("extra_headers")?
        .filter(|value| !value.is_none())
        .map(|value| from_py(&value))
        .transpose()?;
    let timeout = arguments
        .get("timeout")?
        .filter(|value| !value.is_none())
        .map(|value| python_timeout_seconds(py, value.unbind()))
        .transpose()?
        .flatten()
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds).map_err(|_| {
                litellm_core::ocr::Error::RequestField {
                    path: "timeout_seconds".into(),
                }
            })
        })
        .transpose()
        .map_err(ocr_error_to_pyerr)?;

    let request = (|| {
        let core = LiteLLMOcrRequest::new(
            model,
            OcrDocument::try_from(document)?,
            custom_llm_provider.as_deref(),
            optional_params.into(),
        )?;
        let transport = core.transport.clone().with_overrides(
            header_pairs(extra_headers)?,
            source_for(&input_sources, "extra_headers"),
            timeout,
        );
        let credentials = OcrCredentialInputs::new(
            api_key,
            source_for(&input_sources, "api_key"),
            api_base,
            source_for(&input_sources, "api_base"),
        );
        Ok::<_, litellm_core::ocr::Error>(
            core.with_connection_inputs(credentials, transport, input_sources)
                .with_host_hooks(Arc::new(BridgeOcrHooks), None),
        )
    })()
    .map_err(ocr_error_to_pyerr)?;

    Ok(ProjectedOcrCall {
        request,
        azure_ad_token_provider,
        secret_fields: specs
            .into_iter()
            .filter(|spec| spec.secret)
            .map(|spec| spec.name)
            .collect(),
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
    use litellm_core::ocr::Error;
    use litellm_core::ocr::OcrDecline;
    use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};

    use super::*;

    #[test]
    fn projection_selects_consumed_values_without_serializing_host_objects() {
        let fields = consumed_optional_params("mistral/model", None).unwrap();
        use litellm_core::call_arguments::should_project;
        assert!(should_project("future_option", &fields, BOUND_FIELDS));
        assert!(should_project("extra_body", &fields, BOUND_FIELDS));
        assert!(should_project("id", &fields, BOUND_FIELDS));
        assert!(!should_project("metadata", &fields, BOUND_FIELDS));
        assert!(!should_project("callbacks", &fields, BOUND_FIELDS));
        assert!(!should_project("api_key", &fields, BOUND_FIELDS));
        assert!(!should_project("document", &fields, BOUND_FIELDS));
    }

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
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
    fn file_documents_are_encoded_and_other_documents_pass_through() {
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
                project_document(py, &file).unwrap(),
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
            assert_eq!(
                project_document(py, &original).unwrap(),
                serde_json::json!({
                    "type": "document_url",
                    "document_url": "https://example.com/a.pdf",
                })
            );
        });
    }

    #[test]
    fn unknown_document_types_reach_existing_downstream_validation() {
        Python::initialize();
        Python::attach(|py| {
            let document = py
                .eval(c"{'type': 'mystery', 'mystery': 'x'}", None, None)
                .unwrap();
            let wire_document = project_document(py, &document).unwrap();
            assert_eq!(
                wire_document,
                serde_json::json!({"type": "mystery", "mystery": "x"})
            );
            let error = OcrDocument::try_from(wire_document).unwrap_err();
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
            let wire = project_document(py, &document).unwrap();
            assert_eq!(wire["type"], "document_url");
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type", "mime_type", "file"]);
        });
    }
}
