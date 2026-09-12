use serde_json::{Map, Value};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::auth::ResolvedCredential;
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest, OcrPreCallRequest};
use litellm_core::ocr::wire::{OcrWireRequest, consumed_optional_param_names, decode_request};
use litellm_core::ocr::{
    NativeOutcome, OcrAdmission, OcrCall, OcrClient, OcrHostOperation, OcrHostResult,
};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use super::callbacks;
use super::errors::to_pyerr as ocr_error_to_pyerr;
use crate::auth::{AZURE_AD_TOKEN_PROVIDER, PythonTokenProvider};
use crate::errors::RustBridgeDeclined;
use crate::lifecycle::{
    NativeCall, NativeCallStep, OperationClass, PythonCallState, PythonRoute, missing_state, now,
    run_call,
};
use crate::marshal::{project_optional_fields, python_timeout_seconds, request_input_sources};

struct PythonOcrHost {
    state: PythonCallState,
    request: Option<Py<PyAny>>,
    pre_call: Option<OcrPreCallRequest>,
    document: Option<Py<PyAny>>,
    api_key: Option<Py<PyAny>>,
    azure_ad_token_provider: Option<PythonTokenProvider>,
    provider: String,
    retained_fields: Option<Py<PyDict>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
}

struct AdmittedOcrCall {
    request: litellm_core::ocr::LiteLLMOcrRequest,
    document: Py<PyAny>,
    api_key: Py<PyAny>,
    azure_ad_token_provider: Option<PythonTokenProvider>,
    provider: String,
}

impl PythonOcrHost {
    fn pre_call(
        &mut self,
        py: Python<'_>,
        request: OcrPreCallRequest,
    ) -> PyResult<OcrPreCallRequest> {
        let kwargs = self.state.kwargs.bind(py);
        let retained_fields = PyDict::new(py);
        for name in request
            .optional_params
            .as_object()
            .ok_or_else(missing_state)?
            .keys()
        {
            if let Some(value) = kwargs.get_item(name)? {
                retained_fields.set_item(name, value)?;
            }
        }
        retained_fields.set_item(
            "document",
            self.document.as_ref().ok_or_else(missing_state)?,
        )?;
        self.retained_fields = Some(retained_fields.unbind());
        self.pre_call = Some(request.clone());
        Ok(request)
    }

    fn acquire_azure_ad_token(&self, py: Python<'_>) -> PyResult<ResolvedCredential> {
        let provider = self
            .azure_ad_token_provider
            .as_ref()
            .ok_or_else(missing_state)?;
        provider.acquire(py)
    }

    fn python_pre_call(
        &mut self,
        py: Python<'_>,
        mut request: OcrDuringCallRequest,
    ) -> PyResult<OcrDuringCallRequest> {
        let pre_call = self.pre_call.as_ref().ok_or_else(missing_state)?;
        if let Some(body) = request.body.as_object_mut() {
            for name in &request.retained_fields {
                body.remove(name);
            }
        }
        let body = to_py(py, &request.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        if let Some(retained) = &self.retained_fields {
            for name in &request.retained_fields {
                if let Some(value) = retained.bind(py).get_item(name)? {
                    body.set_item(name, value)?;
                }
            }
        }
        let headers = PyDict::new(py);
        for (name, value) in &request.headers {
            headers.set_item(name, value)?;
        }
        self.body = Some(body.clone().unbind());
        self.headers = Some(headers.clone().unbind());
        let logger = self.state.logger()?;
        logger.update_ocr(py, &self.state.kwargs, pre_call, &request.url)?;
        logger.pre_ocr(py, &self.api_key, &body, &headers, &request.url)?;
        let headers = headers
            .iter()
            .map(|(name, value)| Ok((name.extract::<String>()?, value.extract::<String>()?)))
            .collect::<PyResult<Vec<_>>>()?;
        request.body = from_py(&body)?;
        request.headers = headers;
        Ok(request)
    }

    fn python_post_call(
        &mut self,
        py: Python<'_>,
        request: OcrPostCallRequest,
    ) -> PyResult<OcrPostCallRequest> {
        self.state
            .logger()?
            .post_ocr(py, &request.original_response, &self.body, &self.headers)?;
        Ok(request)
    }
}

impl NativeCall for OcrCall {
    type Operation = OcrHostOperation;
    type Result = OcrHostResult;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            OcrCall::resume(self, result).await.map(|step| match step {
                litellm_core::ocr::OcrCallStep::Host(operation) => NativeCallStep::Host(operation),
                litellm_core::ocr::OcrCallStep::Complete(_) => NativeCallStep::Complete,
            })
        })
    }

    fn interrupt(
        &mut self,
        failure: litellm_core::call_lifecycle::host::HostFailure,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<NativeCallStep<Self::Operation>, litellm_core::Error>,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            OcrCall::interrupt(self, failure)
                .await
                .map(|step| match step {
                    litellm_core::ocr::OcrCallStep::Host(operation) => {
                        NativeCallStep::Host(operation)
                    }
                    litellm_core::ocr::OcrCallStep::Complete(_) => NativeCallStep::Complete,
                })
        })
    }
}

impl PythonRoute for PythonOcrHost {
    type Call = OcrCall;

    fn state(&self) -> &PythonCallState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut PythonCallState {
        &mut self.state
    }

    fn classify(operation: &OcrHostOperation) -> OperationClass {
        match operation {
            OcrHostOperation::Lifecycle(phase) => OperationClass::Phase(*phase),
            OcrHostOperation::Success { .. } => {
                OperationClass::Phase(litellm_core::call_lifecycle::host::HostPhase::Success)
            }
            OcrHostOperation::Failure { .. } => {
                OperationClass::Phase(litellm_core::call_lifecycle::host::HostPhase::Failure)
            }
            _ => OperationClass::Route,
        }
    }

    fn lifecycle_result() -> OcrHostResult {
        OcrHostResult::Lifecycle(Ok(()))
    }

    fn map_error(error: litellm_core::Error) -> PyErr {
        ocr_error_to_pyerr(error)
    }

    fn invoke(&mut self, py: Python<'_>, operation: OcrHostOperation) -> PyResult<OcrHostResult> {
        Ok(match operation {
            OcrHostOperation::ProjectRequest => {
                let projected = project_request(
                    py,
                    self.request.as_ref().ok_or_else(missing_state)?.bind(py),
                    self.state.kwargs.bind(py),
                )?;
                self.document = Some(projected.document);
                self.api_key = Some(projected.api_key);
                self.azure_ad_token_provider = projected.azure_ad_token_provider;
                self.provider = projected.provider;
                OcrHostResult::Request(Ok((
                    Box::new(projected.request),
                    self.azure_ad_token_provider.is_some(),
                )))
            }
            OcrHostOperation::AcquireAzureAdToken => {
                OcrHostResult::AzureAdToken(Ok(self.acquire_azure_ad_token(py)?))
            }
            OcrHostOperation::PreCall(request) => {
                OcrHostResult::PreCall(Ok(self.pre_call(py, request)?))
            }
            OcrHostOperation::DuringCall(request) => {
                OcrHostResult::DuringCall(Ok(self.python_pre_call(py, request)?))
            }
            OcrHostOperation::PostCall(request) => {
                OcrHostResult::PostCall(Ok(self.python_post_call(py, request)?))
            }
            OcrHostOperation::ConstructResponse(response) => {
                self.state.end = Some(now(py)?);
                self.state.response = Some(callbacks::response(py, response.as_ref())?);
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::MapFailure(error) => {
                if self.state.error.is_none() {
                    self.state.retain_error(py, ocr_error_to_pyerr(error));
                }
                if self.state.end.is_none() {
                    self.state.end = Some(now(py)?);
                }
                let error = self.state.error.as_ref().ok_or_else(missing_state)?;
                let request = self.request.as_ref().ok_or_else(missing_state)?.bind(py);
                let mapped = callbacks::map_failure(py, error, request, &self.provider)?;
                self.state
                    .retain_error(py, PyErr::from_value(mapped.into_bound(py).into_any()));
                OcrHostResult::Lifecycle(Ok(()))
            }
            OcrHostOperation::Lifecycle(_)
            | OcrHostOperation::Success { .. }
            | OcrHostOperation::Failure { .. } => return Err(missing_state()),
        })
    }

    fn cleanup(&mut self) {
        self.request = None;
        self.pre_call = None;
        self.document = None;
        self.api_key = None;
        self.azure_ad_token_provider = None;
        self.retained_fields = None;
        self.body = None;
        self.headers = None;
    }
    fn traverse(&self, visit: &pyo3::gc::PyVisit<'_>) -> Result<(), pyo3::gc::PyTraverseError> {
        visit.call(&self.request)?;
        visit.call(&self.document)?;
        visit.call(&self.api_key)?;
        if let Some(provider) = &self.azure_ad_token_provider {
            provider.traverse(visit)?;
        }
        visit.call(&self.retained_fields)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
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

    fn document(&self) -> PyResult<CapturedDocument<'py>> {
        Ok(CapturedDocument(self.lookup("document")?))
    }

    fn api_key(&self) -> PyResult<CapturedApiKey<'py>> {
        Ok(CapturedApiKey(self.lookup("api_key")?))
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

struct CapturedDocument<'py>(Bound<'py, PyAny>);

impl<'py> CapturedDocument<'py> {
    fn as_bound(&self) -> &Bound<'py, PyAny> {
        &self.0
    }
}

struct CapturedApiKey<'py>(Bound<'py, PyAny>);

impl CapturedApiKey<'_> {
    fn value(&self) -> PyResult<Option<String>> {
        self.0.extract()
    }

    fn into_object(self) -> Py<PyAny> {
        self.0.unbind()
    }
}

enum DocumentKind {
    File,
    Other,
}

impl FromPyObject<'_, '_> for DocumentKind {
    type Error = PyErr;

    fn extract(document: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let kind: String = document.get_item("type")?.extract()?;

        Ok(match kind.as_str() {
            "file" => Self::File,
            _ => Self::Other,
        })
    }
}

fn project_request(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<AdmittedOcrCall> {
    let arguments = OcrArguments { request, kwargs };
    let model = arguments.model()?;
    let custom_llm_provider = arguments.custom_llm_provider()?;
    let document = arguments.document()?;
    let wire_document = extract_document(py, document.as_bound())?;
    let retained_document = retained_document(py, document.as_bound(), &wire_document)?;
    let api_key = arguments.api_key()?;
    let request_kwargs = kwargs;
    let consumed = consumed_optional_param_names(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let optional_params = project_optional_fields(request_kwargs, &consumed)?;
    let input_sources = request_input_sources(
        request_kwargs,
        consumed
            .iter()
            .copied()
            .chain(["api_key", "api_base", "extra_headers"]),
    )?;
    let azure_ad_token_provider = request_kwargs
        .get_item("azure_ad_token_provider")?
        .and_then(|provider| PythonTokenProvider::select(provider, AZURE_AD_TOKEN_PROVIDER));
    let wire = OcrWireRequest {
        model,
        document: wire_document,
        api_key: api_key.value()?,
        api_base: arguments.api_base()?,
        custom_llm_provider,
        extra_headers: arguments.extra_headers()?,
        optional_params,
        input_sources,
        timeout_seconds: arguments.timeout_seconds()?,
    };
    let request = decode_request(wire).map_err(ocr_error_to_pyerr)?;
    let provider = request.provider_name().to_string();
    let request = request.with_host_hooks(Arc::new(BridgeOcrHooks), None);
    Ok(AdmittedOcrCall {
        request,
        document: retained_document,
        api_key: api_key.into_object(),
        azure_ad_token_provider,
        provider,
    })
}

fn extract_document(py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<Value> {
    match document.extract::<DocumentKind>()? {
        DocumentKind::Other => from_py(document),
        DocumentKind::File => {
            let input = document.extract()?;
            let encoded = super::document::file_document(py, input)?;
            serde_json::to_value(encoded)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
        }
    }
}

fn retained_document(
    py: Python<'_>,
    document: &Bound<'_, PyAny>,
    wire_document: &Value,
) -> PyResult<Py<PyAny>> {
    match document.extract::<DocumentKind>()? {
        DocumentKind::File => to_py(py, wire_document),
        DocumentKind::Other => Ok(document.clone().unbind()),
    }
}

fn admitted_call(outcome: NativeOutcome<OcrCall>) -> PyResult<OcrCall> {
    match outcome {
        NativeOutcome::Completed(call) => Ok(call),
        NativeOutcome::Declined(reason) => Err(RustBridgeDeclined::new_err(format!(
            "native OCR admission declined: {reason:?}"
        ))),
    }
}

struct BridgeOcrHooks;

impl litellm_core::ocr::hooks::OcrHooks for BridgeOcrHooks {
    fn has_guardrails(&self) -> bool {
        true
    }
}

#[pyfunction]
fn _ocr_lifecycle(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    if let Ok(gil_enabled) = py.import("sys")?.getattr("_is_gil_enabled")
        && !gil_enabled.call0()?.is_truthy()?
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "native OCR requires the Python GIL",
        ));
    }
    let client = OcrClient::shared().map_err(ocr_error_to_pyerr)?;
    let call = admitted_call(OcrCall::admit(
        client,
        OcrAdmission {
            asynchronous,
            ..OcrAdmission::all()
        },
    ))?;
    let host = PythonOcrHost {
        state: PythonCallState::new(
            py,
            args.unbind(),
            kwargs.copy()?.unbind(),
            asynchronous,
            if asynchronous { "aocr" } else { "ocr" },
        )?,
        request: Some(request.unbind()),
        pre_call: None,
        document: None,
        api_key: None,
        azure_ad_token_provider: None,
        provider: String::new(),
        retained_fields: None,
        body: None,
        headers: None,
    };
    run_call(py, call, host)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_ocr_lifecycle, module)?)
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
            extract_document(py, document.as_bound()).unwrap();
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
                    .into_object()
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
                extract_document(py, &file).unwrap(),
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
            let wire = extract_document(py, &original).unwrap();
            assert_eq!(
                wire,
                serde_json::json!({
                    "type": "document_url",
                    "document_url": "https://example.com/a.pdf",
                })
            );
            assert!(
                retained_document(py, &original, &wire)
                    .unwrap()
                    .bind(py)
                    .is(&original)
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
            let wire_document = extract_document(py, &document).unwrap();
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
                extract_document(py, &missing)
                    .unwrap_err()
                    .is_instance_of::<PyKeyError>(py)
            );

            let non_string = py.eval(c"{'type': 1}", None, None).unwrap();
            assert!(
                extract_document(py, &non_string)
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
                extract_document(py, &locals.get_item("document").unwrap().unwrap()).unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
        });
    }

    #[test]
    fn document_kind_reads_only_type_and_classification_happens_twice() {
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
            assert!(matches!(
                document.extract::<DocumentKind>().unwrap(),
                DocumentKind::File
            ));
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type"]);

            py.run(c"document.reads = []", Some(&locals), Some(&locals))
                .unwrap();
            let wire = extract_document(py, &document).unwrap();
            let retained = retained_document(py, &document, &wire).unwrap();
            assert!(retained.bind(py).is(&document));
            let reads: Vec<String> = document.getattr("reads").unwrap().extract().unwrap();
            assert_eq!(reads, ["type", "file", "type"]);
        });
    }
}
