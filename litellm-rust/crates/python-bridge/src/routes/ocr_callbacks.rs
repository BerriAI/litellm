use std::sync::Mutex;

use litellm_core::Error;
use litellm_core::auth::{ResolvedCredential, SecretValue, TokenFuture, TokenProvider};
use litellm_core::ocr::hooks::{OcrHookFuture, OcrHooks, OcrRequestDraft};
use litellm_python_interop::{from_py, to_py};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_async_runtimes::TaskLocals;

struct PendingCallback(Option<Py<PyAny>>);

impl Drop for PendingCallback {
    fn drop(&mut self) {
        if let Some(future) = self.0.take() {
            Python::attach(|py| {
                if let Err(error) = future.call_method0(py, "cancel") {
                    error.write_unraisable(py, Some(future.bind(py)));
                }
            });
        }
    }
}

pub(super) struct PythonOcrHooks {
    pub logger: Option<Py<PyAny>>,
    pub token_provider: Option<Py<PyAny>>,
    pub document: Py<PyAny>,
    pub document_snapshot: serde_json::Value,
    pub api_key: Option<String>,
    pub locals: Option<TaskLocals>,
    pub error: Mutex<Option<PyErr>>,
    pub execution_body: Mutex<Option<Py<PyDict>>>,
}

#[pyclass]
pub(super) struct NativeOcrCompletion {
    python_completion: Py<PyAny>,
}

impl NativeOcrCompletion {
    pub(super) fn attach(call_completion: &Py<PyAny>) -> PyResult<()> {
        Python::attach(|py| {
            let python_completion = call_completion.getattr(py, "python_implementation")?;
            let native = Py::new(py, Self { python_completion })?;
            let attached: bool = call_completion
                .call_method1(py, "attach", (native,))?
                .extract(py)?;
            if attached {
                Ok(())
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "OCR completion was already attached",
                ))
            }
        })
    }
}

#[pymethods]
impl NativeOcrCompletion {
    fn success(
        &self,
        py: Python<'_>,
        result: Py<PyAny>,
        start_time: Py<PyAny>,
        end_time: Py<PyAny>,
    ) -> PyResult<()> {
        self.python_completion
            .call_method1(py, "success", (result, start_time, end_time))?;
        Ok(())
    }

    fn failure(
        &self,
        py: Python<'_>,
        exception: Py<PyAny>,
        traceback_exception: String,
        start_time: Py<PyAny>,
        end_time: Py<PyAny>,
    ) -> PyResult<()> {
        self.python_completion.call_method1(
            py,
            "failure",
            (exception, traceback_exception, start_time, end_time),
        )?;
        Ok(())
    }

    fn async_failure(
        &self,
        py: Python<'_>,
        exception: Py<PyAny>,
        traceback_exception: String,
        start_time: Py<PyAny>,
        end_time: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        self.python_completion.call_method1(
            py,
            "async_failure",
            (exception, traceback_exception, start_time, end_time),
        )
    }
}

impl PythonOcrHooks {
    pub async fn invoke(&self, callback: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if let Some(locals) = &self.locals {
            let (mut pending, future) = Python::attach(|py| {
                let coroutine = py
                    .import("litellm.rust_bridge._callbacks")?
                    .getattr("invoke_callback")?
                    .call1((callback,))?;
                let asyncio = py.import("asyncio")?;
                let submitted = locals.context(py).call_method1(
                    "run",
                    (
                        asyncio.getattr("run_coroutine_threadsafe")?,
                        coroutine,
                        locals.event_loop(py),
                    ),
                )?;
                let pending = PendingCallback(Some(submitted.clone().unbind()));
                let kwargs = PyDict::new(py);
                kwargs.set_item("loop", locals.event_loop(py))?;
                let wrapped = asyncio
                    .getattr("wrap_future")?
                    .call((submitted,), Some(&kwargs))?;
                let future = pyo3_async_runtimes::into_future_with_locals(locals, wrapped)?;
                Ok::<_, PyErr>((pending, future))
            })?;
            let result = future.await;
            pending.0.take();
            return result;
        }
        tokio::task::block_in_place(|| litellm_python_interop::invoke_callback(&callback))
    }

    fn retain_error(&self, error: PyErr) {
        *self
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error);
    }

    fn callback(
        &self,
        py: Python<'_>,
        request: &OcrRequestDraft,
    ) -> PyResult<(Py<PyAny>, Py<PyDict>, Py<PyDict>)> {
        let body = to_py(py, &request.body)?
            .into_bound(py)
            .cast_into::<PyDict>()?;
        if request.body.get("document") == Some(&self.document_snapshot) {
            body.set_item("document", &self.document)?;
        }
        let headers = PyDict::new(py);
        for (name, value) in &request.headers {
            headers.set_item(name, value)?;
        }
        let additional = PyDict::new(py);
        additional.set_item("complete_input_dict", &body)?;
        additional.set_item("headers", &headers)?;
        additional.set_item("api_base", &request.url)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("input", "OCR document processing")?;
        kwargs.set_item("api_key", &self.api_key)?;
        kwargs.set_item("model", &request.model)?;
        kwargs.set_item("additional_args", additional)?;
        let callback = py.import("functools")?.getattr("partial")?.call(
            PyTuple::new(
                py,
                [self
                    .logger
                    .as_ref()
                    .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("missing OCR logger"))?
                    .bind(py)
                    .getattr("pre_call")?],
            )?,
            Some(&kwargs),
        )?;
        Ok((callback.unbind(), body.unbind(), headers.unbind()))
    }
}

impl OcrHooks for PythonOcrHooks {
    fn before_send(&self, request: OcrRequestDraft) -> OcrHookFuture<'_, OcrRequestDraft> {
        Box::pin(async move {
            if self.logger.is_none() {
                return Ok(request);
            }
            let result: PyResult<OcrRequestDraft> = async {
                let (callback, body, headers) = Python::attach(|py| self.callback(py, &request))?;
                self.invoke(callback).await?;
                Python::attach(|py| {
                    *self
                        .execution_body
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                        Some(body.clone_ref(py));
                });
                Python::attach(|py| {
                    Ok(OcrRequestDraft {
                        body: from_py(body.bind(py).as_any())?,
                        headers: headers
                            .bind(py)
                            .extract::<std::collections::BTreeMap<String, String>>()?
                            .into_iter()
                            .collect(),
                        ..request
                    })
                })
            }
            .await;
            result.map_err(|error| {
                self.retain_error(error);
                Error::InvalidRequest("OCR pre-call hook failed".into())
            })
        })
    }

    fn post_call<'a>(&'a self, response: &'a str) -> OcrHookFuture<'a, ()> {
        Box::pin(async move {
            let Some(logger) = &self.logger else {
                return Ok(());
            };
            let result: PyResult<()> = async {
                let callback = Python::attach(|py| {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("api_key", &self.api_key)?;
                    kwargs.set_item("original_response", response)?;
                    let additional = PyDict::new(py);
                    let body = self
                        .execution_body
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    additional.set_item(
                        "complete_input_dict",
                        body.as_ref().ok_or_else(|| {
                            pyo3::exceptions::PyRuntimeError::new_err("missing OCR execution body")
                        })?,
                    )?;
                    kwargs.set_item("additional_args", additional)?;
                    Ok::<_, PyErr>(
                        py.import("functools")?
                            .getattr("partial")?
                            .call((logger.bind(py).getattr("post_call")?,), Some(&kwargs))?
                            .unbind(),
                    )
                })?;
                self.invoke(callback).await?;
                Ok(())
            }
            .await;
            result.map_err(|error| {
                self.retain_error(error);
                Error::InvalidRequest("OCR post-call hook failed".into())
            })
        })
    }
}

impl std::fmt::Debug for PythonOcrHooks {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("PythonOcrHooks")
    }
}

impl TokenProvider for PythonOcrHooks {
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            let result: PyResult<String> = async {
                let callback = Python::attach(|py| {
                    self.token_provider
                        .as_ref()
                        .map(|provider| provider.clone_ref(py))
                        .ok_or_else(|| {
                            pyo3::exceptions::PyRuntimeError::new_err("missing token provider")
                        })
                })?;
                let value = self.invoke(callback).await.map_err(|error| {
                    Python::attach(|py| {
                        if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                            || !error.is_instance_of::<pyo3::exceptions::PyException>(py)
                        {
                            return error;
                        }
                        let wrapped = pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to get Azure AD token: {}",
                            error.value(py)
                        ));
                        wrapped.set_cause(py, Some(error));
                        wrapped
                    })
                })?;
                Python::attach(|py| {
                    value.extract::<String>(py).map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err("Azure AD token must be a string")
                    })
                })
            }
            .await;
            match result {
                Ok(token) if token.is_empty() => {
                    Err(litellm_core::AuthError::AzureTokenAcquisition(
                        "Missing Azure AI credentials".into(),
                    ))
                }
                Ok(token) => Ok(ResolvedCredential::AccessToken {
                    token: SecretValue::new(token),
                    expires_on: None,
                }),
                Err(error) => {
                    self.retain_error(error);
                    Err(litellm_core::AuthError::AzureTokenAcquisition(
                        "host token provider failed".into(),
                    ))
                }
            }
        })
    }
}
