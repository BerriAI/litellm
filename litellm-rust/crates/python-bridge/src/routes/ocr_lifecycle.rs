use serde_json::{Map, Value};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use litellm_core::auth::{ResolvedCredential, SecretValue};
use litellm_core::ocr::hooks::{OcrDuringCallRequest, OcrPostCallRequest, OcrPreCallRequest};
use litellm_core::ocr::wire::{OcrWireRequest, consumed_optional_param_names, decode_request};
use litellm_core::ocr::{
    NativeOutcome, OcrAdmission, OcrCall, OcrClient, OcrHostOperation, OcrHostResult,
};
use litellm_python_interop::{
    from_py_preserving_errors as from_py, to_py_preserving_errors as to_py,
};

use crate::errors::{RustBridgeDeclined, ocr_error_to_pyerr};
use crate::lifecycle::{PythonCallState, PythonRoute, missing_state, now, run_call};

struct PythonOcrHost {
    state: PythonCallState,
    request: Option<Py<PyAny>>,
    pre_call: Option<OcrPreCallRequest>,
    document: Option<Py<PyAny>>,
    api_key: Option<Py<PyAny>>,
    azure_ad_token_provider: Option<Py<PyAny>>,
    provider: String,
    retained_fields: Option<Py<PyDict>>,
    body: Option<Py<PyDict>>,
    headers: Option<Py<PyDict>>,
}

struct AdmittedOcrCall {
    request: litellm_core::ocr::LiteLLMOcrRequest,
    document: Py<PyAny>,
    api_key: Py<PyAny>,
    azure_ad_token_provider: Option<Py<PyAny>>,
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
        let token: String = py
            .import("litellm.rust_bridge.ocr_lifecycle")?
            .getattr("call_azure_ad_token_provider")?
            .call1((provider,))?
            .extract()?;
        Ok(ResolvedCredential::AccessToken {
            token: SecretValue::new(token),
            expires_on: None,
        })
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
        let logger = self.state.logger(py)?;
        let redact = py
            .import("litellm.rust_bridge.ocr")?
            .getattr("redact_logging_params")?;
        let update = PyDict::new(py);
        update.set_item("kwargs", redact.call1((&self.state.kwargs,))?)?;
        update.set_item("model", &pre_call.model)?;
        update.set_item(
            "optional_params",
            redact.call1((to_py(py, &pre_call.optional_params)?,))?,
        )?;
        let params = PyDict::new(py);
        params.set_item(
            "litellm_call_id",
            self.state.kwargs.bind(py).get_item("litellm_call_id")?,
        )?;
        params.set_item("api_base", &request.url)?;
        update.set_item("litellm_params", params)?;
        update.set_item("custom_llm_provider", &pre_call.custom_llm_provider)?;
        logger.call_method("update_from_kwargs", (), Some(&update))?;
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", &body)?;
        additional.set_item("headers", &headers)?;
        additional.set_item("api_base", &request.url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", &self.api_key)?;
        kwargs.set_item("additional_args", additional)?;
        logger.call_method("pre_call", (), Some(&kwargs))?;
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
        let logger = self.state.logger(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("original_response", to_py(py, &request.original_response)?)?;
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", &self.body)?;
        additional.set_item("headers", &self.headers)?;
        kwargs.set_item("additional_args", additional)?;
        logger.call_method("post_call", (), Some(&kwargs))?;
        Ok(request)
    }
}

impl PythonRoute for PythonOcrHost {
    fn state(&self) -> &PythonCallState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut PythonCallState {
        &mut self.state
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
                self.state.response = Some(
                    py.import("litellm.rust_bridge.ocr")?
                        .getattr("_response")?
                        .call1((to_py(py, response.as_ref())?,))?
                        .unbind(),
                );
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
                let mapped = py
                    .import("litellm.rust_bridge.ocr_lifecycle")?
                    .getattr("map_failure")?
                    .call1((error, request, &self.provider))?;
                self.state.retain_error(py, PyErr::from_value(mapped));
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
        visit.call(&self.azure_ad_token_provider)?;
        visit.call(&self.retained_fields)?;
        visit.call(&self.body)?;
        visit.call(&self.headers)
    }
}

fn project_request(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<AdmittedOcrCall> {
    let argument = |name: &str| {
        kwargs
            .get_item(name)?
            .map(Ok)
            .unwrap_or_else(|| request.getattr(name))
    };
    let model: String = argument("model")?.extract()?;
    let custom_llm_provider: Option<String> = argument("custom_llm_provider")?.extract()?;
    let document = argument("document")?;
    let wire_document = extract_document(py, &document)?;
    let retained_document = retained_document(py, &document, &wire_document)?;
    let api_key = argument("api_key")?;
    let request_kwargs = kwargs;
    let consumed = consumed_optional_param_names(&model, custom_llm_provider.as_deref())
        .map_err(ocr_error_to_pyerr)?;
    let optional_params = extract_optional_params(request_kwargs, &consumed)?;
    let input_sources = extract_input_sources(request_kwargs, &consumed)?;
    let azure_ad_token_provider = request_kwargs
        .get_item("azure_ad_token_provider")?
        .filter(|provider| provider.is_callable() && provider.is_truthy().unwrap_or(false))
        .map(Bound::unbind);
    let wire = OcrWireRequest {
        model,
        document: wire_document,
        api_key: api_key.extract()?,
        api_base: argument("api_base")?.extract()?,
        custom_llm_provider,
        extra_headers: argument("extra_headers")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| from_py(value.bind(py)))
            .transpose()?,
        optional_params,
        input_sources,
        timeout_seconds: argument("timeout")?
            .extract::<Option<Py<PyAny>>>()?
            .map(|value| {
                py.import("litellm.rust_bridge.timeouts")?
                    .getattr("timeout_to_seconds")?
                    .call1((value,))?
                    .extract()
            })
            .transpose()?
            .flatten(),
    };
    let request = decode_request(wire).map_err(ocr_error_to_pyerr)?;
    let provider = request.provider_name().to_string();
    let request = request.with_host_hooks(Arc::new(BridgeOcrHooks), None);
    Ok(AdmittedOcrCall {
        request,
        document: retained_document,
        api_key: api_key.unbind(),
        azure_ad_token_provider,
        provider,
    })
}

fn extract_optional_params(
    kwargs: &Bound<'_, PyDict>,
    consumed: &[&str],
) -> PyResult<Map<String, Value>> {
    let mut optional_params = Map::new();
    for name in consumed {
        if let Some(value) = kwargs.get_item(name)? {
            optional_params.insert((*name).to_string(), from_py(&value)?);
        }
    }
    Ok(optional_params)
}

fn extract_input_sources(
    kwargs: &Bound<'_, PyDict>,
    consumed: &[&str],
) -> PyResult<std::collections::BTreeMap<String, litellm_core::auth::InputSource>> {
    let Some(proxy_request) = kwargs.get_item("proxy_server_request")? else {
        return Ok(Default::default());
    };
    let proxy_request = proxy_request.cast_into::<PyDict>()?;
    let body_fields = proxy_request
        .get_item("body_fields")?
        .or(proxy_request.get_item("body")?);
    let credential_fields = proxy_request.get_item("credential_fields")?;
    let mut sources = std::collections::BTreeMap::new();
    for name in consumed
        .iter()
        .copied()
        .chain(["api_key", "api_base", "extra_headers"])
    {
        let present = body_fields
            .as_ref()
            .is_some_and(|fields| fields.contains(name).unwrap_or(false))
            || credential_fields
                .as_ref()
                .is_some_and(|fields| fields.contains(name).unwrap_or(false));
        if present {
            sources.insert(name.to_string(), litellm_core::auth::InputSource::Request);
        }
    }
    Ok(sources)
}

fn extract_document(py: Python<'_>, document: &Bound<'_, PyAny>) -> PyResult<Value> {
    if document.get_item("type")?.extract::<String>()? != "file" {
        return from_py(document);
    }
    serde_json::to_value(super::ocr_document::file_document(py, document)?)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn retained_document(
    py: Python<'_>,
    document: &Bound<'_, PyAny>,
    wire_document: &Value,
) -> PyResult<Py<PyAny>> {
    if document.get_item("type")?.extract::<String>()? == "file" {
        to_py(py, wire_document)
    } else {
        Ok(document.clone().unbind())
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
    use pyo3::exceptions::PyValueError;

    use super::*;

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
}
