use std::future::Future;

use litellm_core::Error;
use litellm_core::ocr::{
    OcrRequest, PreparedOcrRequest, execute as execute_ocr, ocr as run_ocr,
    prepare as prepare_core_ocr,
};
use litellm_core::routing_utils::provider::get_custom_llm_provider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde_json::Value;

use crate::client::shared_http_client;
use crate::constants::RUST_OCR_PROVIDERS;
use crate::errors::{ocr_error_to_pyerr, ocr_prepare_error_to_pyerr};
use crate::execution;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

#[pyclass]
struct PreparedOcr {
    request: Option<PreparedOcrRequest>,
}

struct OwnedOcrRequest {
    document: Value,
    options: RouteOptions,
    optional_params: serde_json::Map<String, Value>,
    max_document_download_bytes: u64,
}

#[pyfunction]
#[pyo3(signature = (model, optional_params=None, custom_llm_provider=None))]
fn ocr_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
) -> PyResult<Option<String>> {
    let optional_params = object_or_empty("optional_params", optional_params)?;
    Ok(ocr_decline_reason(
        &model,
        &optional_params,
        custom_llm_provider.as_deref(),
    ))
}

fn ocr_decline_reason(
    model: &str,
    optional_params: &serde_json::Map<String, Value>,
    custom_llm_provider: Option<&str>,
) -> Option<String> {
    let provider = get_custom_llm_provider(model, custom_llm_provider)
        .map(|info| info.custom_llm_provider)
        .unwrap_or("mistral");
    if !RUST_OCR_PROVIDERS.contains(&provider) {
        return Some(format!("provider {provider} is not supported"));
    }
    if optional_params.get("req_format").and_then(Value::as_str) == Some("native") {
        return Some("req_format=native is not supported".to_string());
    }
    None
}

fn owned_ocr_request(inputs: OcrInputs) -> PyResult<OwnedOcrRequest> {
    let document = inputs.document;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    Ok(OwnedOcrRequest {
        document,
        options,
        optional_params,
        max_document_download_bytes: inputs.max_document_download_bytes,
    })
}

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let request = owned_ocr_request(inputs)?;
    Ok(async move {
        let OwnedOcrRequest {
            document,
            options,
            optional_params,
            max_document_download_bytes,
        } = request;
        let client = shared_http_client().map_err(Error::Network)?;
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        run_ocr(
            &client,
            OcrRequest {
                model: &model,
                document,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                optional_params,
                timeout,
                max_document_download_bytes,
            },
        )
        .await
    })
}

fn prepare_dispatch(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<PreparedOcr, Error>> + Send + 'static> {
    let request = owned_ocr_request(inputs)?;
    Ok(async move {
        let OwnedOcrRequest {
            document,
            options,
            optional_params,
            max_document_download_bytes,
        } = request;
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        if let Some(reason) =
            ocr_decline_reason(&model, &optional_params, custom_llm_provider.as_deref())
        {
            return Err(Error::InvalidRequest(reason));
        }
        let request = prepare_core_ocr(OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
            max_document_download_bytes,
        })
        .await?;
        Ok(PreparedOcr {
            request: Some(request),
        })
    })
}

#[pyfunction(name = "ocr_prepare")]
#[pyo3(signature = (model, document, max_document_download_bytes, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn ocr_prepare(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] document: Value,
    max_document_download_bytes: u64,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let future = prepare_dispatch(OcrInputs {
        model,
        document,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
        max_document_download_bytes,
    })?;
    let prepared = execution::run_sync_value(py, future, ocr_prepare_error_to_pyerr)?;
    Py::new(py, prepared).map(Py::into_any)
}

#[pyfunction(name = "aocr_prepare")]
#[pyo3(signature = (model, document, max_document_download_bytes, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, optional_params=None, timeout_seconds=None))]
#[allow(clippy::too_many_arguments)]
fn aocr_prepare<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] document: Value,
    max_document_download_bytes: u64,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] extra_headers: Option<Value>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let future = prepare_dispatch(OcrInputs {
        model,
        document,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        optional_params,
        timeout_seconds,
        max_document_download_bytes,
    })?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let prepared = execution::await_value(future, ocr_prepare_error_to_pyerr).await?;
        Python::attach(|py| Py::new(py, prepared).map(Py::into_any))
    })
}

fn take_request(py: Python<'_>, prepared: Py<PreparedOcr>) -> PyResult<PreparedOcrRequest> {
    prepared
        .borrow_mut(py)
        .request
        .take()
        .ok_or_else(|| PyRuntimeError::new_err("prepared OCR request was already executed"))
}

async fn execute_prepared_ocr(request: PreparedOcrRequest) -> Result<Value, Error> {
    let client = shared_http_client().map_err(Error::Network)?;
    execute_ocr(&client, request).await
}

#[pyfunction]
fn ocr_execute(py: Python<'_>, prepared: Py<PreparedOcr>) -> PyResult<Py<PyAny>> {
    let request = take_request(py, prepared)?;
    execution::run_sync(py, execute_prepared_ocr(request), ocr_error_to_pyerr)
}

#[pyfunction]
fn aocr_execute<'py>(py: Python<'py>, prepared: Py<PreparedOcr>) -> PyResult<Bound<'py, PyAny>> {
    let request = take_request(py, prepared)?;
    execution::run_async(py, execute_prepared_ocr(request), ocr_error_to_pyerr)
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        document: serde_json::Value,
        max_document_download_bytes: u64,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
    },
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
    extra = [ocr_decline, ocr_prepare, aocr_prepare, ocr_execute, aocr_execute],
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn decline_gate_owns_provider_and_request_capability() {
        assert_eq!(
            ocr_decline("reducto/parse-v3".to_string(), None, None).expect("gate should evaluate"),
            Some("provider reducto is not supported".to_string())
        );
        assert_eq!(
            ocr_decline(
                "mistral-ocr-latest".to_string(),
                Some(json!({"req_format": "native"})),
                Some("mistral".to_string()),
            )
            .expect("gate should evaluate"),
            Some("req_format=native is not supported".to_string())
        );
        assert_eq!(
            ocr_decline(
                "mistral/mistral-ocr-latest".to_string(),
                Some(json!({})),
                None,
            )
            .expect("gate should evaluate"),
            None
        );
    }

    #[tokio::test]
    async fn native_preparation_rejects_invalid_request_before_dispatch() {
        let future = prepare_dispatch(OcrInputs {
            model: "mistral-ocr-latest".to_string(),
            document: json!("not-an-object"),
            api_key: Some("sk-test".to_string()),
            api_base: None,
            custom_llm_provider: Some("mistral".to_string()),
            extra_headers: None,
            optional_params: Some(json!({})),
            timeout_seconds: None,
            max_document_download_bytes: 50 * 1024 * 1024,
        })
        .expect("request should marshal");

        assert!(matches!(
            future.await,
            Err(Error::InvalidType {
                expected: "object",
                actual: "string"
            })
        ));
    }
}
