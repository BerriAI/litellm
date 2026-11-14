use std::{sync::Arc, time::Duration};

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_python::{RustCallback, payload};
use litellm_callbacks_s3_log::{
    Constants, Host, S3Log, S3LogParts, Settings, python_backoff, system_clock,
};
use litellm_host_python::{from_py, json_from_py};
use litellm_http::ClientVariant;
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyInt, PyString},
};
use serde_json::{Map, Value};

struct PythonHost {
    host: Arc<Py<PyAny>>,
}

impl PythonHost {
    fn ask<T>(
        &self,
        method: &str,
        argument: Option<&str>,
        read: impl for<'py> FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Result<T, String> {
        Python::attach(|py| {
            let answer = match argument {
                Some(argument) => self.host.call_method1(py, method, (argument,)),
                None => self.host.call_method0(py, method),
            };
            answer
                .and_then(|answer| read(answer.bind(py)))
                .map_err(|error| error.to_string())
        })
    }
}

impl Host for PythonHost {
    fn callback_failed(&self) {
        let host = Arc::clone(&self.host);
        let report = move || {
            Python::attach(|py| {
                if let Err(error) = host.call_method0(py, "callback_failed") {
                    error.print(py);
                }
            });
        };
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => drop(runtime.spawn_blocking(report)),
            Err(_) => report(),
        }
    }

    fn batch_upload_failed(&self, queued: usize, failed: usize, total: usize) {
        Python::attach(|py| {
            if let Err(error) =
                self.host
                    .call_method1(py, "batch_upload_failed", (queued, failed, total))
            {
                error.print(py);
            }
        });
    }

    fn cold_storage_is_s3_v2(&self) -> bool {
        Python::attach(|py| {
            self.host
                .call_method0(py, "cold_storage_is_s3_v2")
                .and_then(|answer| answer.extract::<bool>(py))
                .unwrap_or(false)
        })
    }

    fn region_from_environment(&self) -> Result<Option<String>, String> {
        self.ask("region_from_environment", None, |answer| {
            answer.extract::<Option<String>>()
        })
    }

    fn web_identity_token(&self, reference: &str) -> Result<String, String> {
        self.ask("web_identity_token", Some(reference), |answer| {
            answer.extract::<String>()
        })
    }
}

#[pyclass(frozen, name = "NativeS3Log")]
pub struct NativeS3Log {
    sink: Arc<S3Log>,
}

fn client(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    connect_timeout_seconds: f64,
) -> PyResult<litellm_http::Client> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("ssl_verify", params.get_item("s3_verify")?)?;
    let mut config = crate::http::call_config(py, &kwargs, true)?;
    config.connect_timeout = Duration::from_secs_f64(connect_timeout_seconds.max(0.0));
    crate::http::pool()
        .client(&config, ClientVariant::Provider)
        .map_err(crate::http::client_error)
}

fn settings_param(value: &Bound<'_, PyAny>) -> Value {
    if let Ok(integer) = value.cast::<PyInt>()
        && !value.is_instance_of::<PyBool>()
        && integer.extract::<i64>().is_err()
        && integer.extract::<u64>().is_err()
    {
        return value
            .str()
            .map_or(Value::Null, |text| Value::String(text.to_string()));
    }
    json_from_py(value).unwrap_or(Value::Null)
}

fn settings_params(params: &Bound<'_, PyDict>) -> Map<String, Value> {
    params
        .iter()
        .filter_map(|(key, value)| {
            let key = key.cast_into::<PyString>().ok()?;
            Some((key.to_str().ok()?.to_owned(), settings_param(&value)))
        })
        .collect()
}

#[pymethods]
impl NativeS3Log {
    #[new]
    #[pyo3(signature = (params, constants, host))]
    fn new(
        py: Python<'_>,
        params: &Bound<'_, PyDict>,
        constants: &Bound<'_, PyAny>,
        host: Py<PyAny>,
    ) -> PyResult<Self> {
        let resolved = settings_params(params);
        let constants: Constants = from_py(constants)?;
        let host = Arc::new(PythonHost {
            host: Arc::new(host),
        });
        let sink = S3Log::new(S3LogParts {
            settings: Settings::from_params(&resolved, &constants),
            client: client(py, params, constants.upload_attempt_timeout_seconds)?,
            download_client: client(py, params, constants.download_connect_timeout_seconds)?,
            environment: Arc::new(|name: &str| std::env::var(name).ok()),
            clock: system_clock,
            backoff: Arc::new(python_backoff),
            host: host as Arc<dyn Host>,
        });
        Ok(Self { sink })
    }

    fn callback(&self) -> PyResult<RustCallback> {
        Ok(RustCallback::new(
            Arc::clone(&self.sink) as Arc<dyn ErasedCallback>
        )?)
    }

    fn stashed(&self) -> usize {
        self.sink.held_payloads()
    }

    fn partition_granularity(&self) -> &'static str {
        self.sink.partition_granularity().as_str()
    }

    #[getter]
    fn s3_path(&self) -> Option<String> {
        self.sink.settings().path.clone()
    }

    #[getter]
    fn s3_bucket_name(&self) -> Option<String> {
        self.sink.settings().bucket_name.clone()
    }

    #[getter]
    fn strip_base64_files(&self) -> bool {
        self.sink.settings().strip_base64_files
    }

    fn queued_keys(&self) -> Vec<String> {
        self.sink.queued_keys()
    }

    fn submit_audit<'py>(
        &self,
        py: Python<'py>,
        audit_log: Py<PyAny>,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        let stashed = payload::stash(audit_log);
        let event_loop = payload::running_loop(py);
        crate::execution::run_async_value(
            py,
            payload::on_render_loop(event_loop, async move {
                sink.submit_audit(stashed, &id).await;
                Ok(())
            }),
        )
    }

    fn upload_line_items<'py>(
        &self,
        py: Python<'py>,
        key: String,
        body: String,
        download_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            Ok(sink.upload_now(key, body, download_name).await)
        })
    }

    fn download<'py>(&self, py: Python<'py>, key: String) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        crate::execution::run_async_value(py, async move {
            Ok(sink
                .download(&key)
                .await
                .map(|body| String::from_utf8_lossy(&body).into_owned()))
        })
    }

    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let sink = Arc::clone(&self.sink);
        let event_loop = payload::running_loop(py);
        crate::execution::run_async_value(
            py,
            payload::on_render_loop(event_loop, async move {
                sink.flush().await;
                Ok(())
            }),
        )
    }
}
