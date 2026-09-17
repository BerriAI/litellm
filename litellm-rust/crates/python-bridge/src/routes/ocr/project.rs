//! `ProjectRequest` host step: read the bound `ocr()` arguments and hand core a
//! typed [`LiteLLMOcrRequest`]. Generic argument handling lives in
//! [`crate::marshal::BoundRouteInputs`]; this module only owns what is specific
//! to OCR: the `document` argument and the core constructor call.

use pyo3::prelude::*;

use litellm_core::ocr::{
    LiteLLMOcrRequest, OcrConnectionInputs, OcrDocument, consumed_optional_params,
};
use litellm_python_interop::from_py_preserving_errors as from_py;

use super::document::FileDocumentInput;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use super::host::OcrRetained;
use crate::lifecycle::BoundArguments;
use crate::marshal::{BoundRouteInputs, Projection};

/// Positional parameters of `ocr()` that are never projected into
/// `optional_params`.
const BOUND_FIELDS: &[&str] = &["model", "document", "timeout", "input_sources"];

fn project_document(document: &Bound<'_, PyAny>) -> PyResult<Result<FileDocumentInput, litellm_core::ocr::Error>> {
    let kind: String = document.get_item("type")?.extract()?;
    if kind != "file" {
        let value: serde_json::Value = from_py(document)?;
        return Ok(OcrDocument::try_from(value).map(|document| FileDocumentInput {
            input: document.into(),
            reader: None,
        }));
    }
    document.extract().map(Ok)
}

/// Pure core assembly; every failure here is a typed `ocr::Error`.
fn build_request(
    inputs: BoundRouteInputs,
    document: Result<FileDocumentInput, litellm_core::ocr::Error>,
) -> Result<Projection<LiteLLMOcrRequest, OcrRetained>, litellm_core::ocr::Error> {
    let document = document?;
    let BoundRouteInputs {
        model,
        custom_llm_provider,
        api_key,
        api_base,
        extra_headers,
        timeout,
        optional_params,
        input_sources,
        azure_ad_token_provider,
        secret_fields,
    } = inputs;
    let request = LiteLLMOcrRequest::from_inputs(
        model,
        document.input,
        custom_llm_provider.as_deref(),
        optional_params.into(),
        OcrConnectionInputs {
            api_key,
            api_base,
            extra_headers,
            timeout,
            input_sources,
        },
    )?;
    Ok(Projection {
        retained: OcrRetained {
            model: request.model.clone(),
            provider: request.provider_name(),
            azure_ad_token_provider,
            secret_fields,
            reader: document.reader,
            payload: None,
        },
        native: request,
    })
}

pub(super) fn project(
    py: Python<'_>,
    arguments: &BoundArguments<'_>,
) -> PyResult<Projection<LiteLLMOcrRequest, OcrRetained>> {
    let model: String = arguments.extract("model")?;
    let custom_llm_provider: Option<String> = arguments.optional("custom_llm_provider")?;
    let document = project_document(&arguments.required("document")?)?;
    let consumed = consumed_optional_params(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let inputs = BoundRouteInputs::extract(py, arguments, consumed, BOUND_FIELDS)?;
    build_request(inputs, document).map_err(ocr_error_to_pyerr)
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{PyKeyError, PyTypeError};
    use pyo3::types::PyDict;

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
    fn file_documents_remain_raw_and_other_documents_pass_through() {
        Python::initialize();
        Python::attach(|py| {
            let file = py
                .eval(
                    c"{'type': 'file', 'file': b'%PDF-1.4', 'mime_type': 'application/pdf'}",
                    None,
                    None,
                )
                .unwrap();
            assert!(matches!(
                project_document(&file).unwrap().unwrap().input,
                litellm_core::ocr::OcrDocumentInput::Bytes { bytes, mime_type, .. }
                    if bytes == b"%PDF-1.4"[..] && mime_type.as_deref() == Some("application/pdf")
            ));

            let original = py
                .eval(
                    c"{'type': 'document_url', 'document_url': 'https://example.com/a.pdf'}",
                    None,
                    None,
                )
                .unwrap();
            assert!(matches!(
                project_document(&original).unwrap().unwrap().input,
                litellm_core::ocr::OcrDocumentInput::Document(OcrDocument::DocumentUrl { document_url, .. })
                    if document_url == "https://example.com/a.pdf"
            ));
        });
    }

    #[test]
    fn unknown_document_types_reach_existing_downstream_validation() {
        Python::initialize();
        Python::attach(|py| {
            let document = py
                .eval(c"{'type': 'mystery', 'mystery': 'x'}", None, None)
                .unwrap();
            let error = project_document(&document)
                .unwrap()
                .err()
                .unwrap();
            assert!(error.to_string().contains("document"));
        });
    }

    #[test]
    fn document_discriminator_errors_keep_their_existing_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let missing = py.eval(c"{}", None, None).unwrap();
            assert!(
                project_document(&missing)
                    .err()
                    .unwrap()
                    .is_instance_of::<PyKeyError>(py)
            );

            let non_string = py.eval(c"{'type': 1}", None, None).unwrap();
            assert!(
                project_document(&non_string)
                    .err()
                    .unwrap()
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
            let error = project_document(&locals.get_item("document").unwrap().unwrap())
                .err()
                .unwrap();
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
            let projected = project_document(&document).unwrap().unwrap();
            assert!(matches!(
                projected.input,
                litellm_core::ocr::OcrDocumentInput::Bytes { .. }
            ));
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type", "mime_type", "file"]);
        });
    }
}
