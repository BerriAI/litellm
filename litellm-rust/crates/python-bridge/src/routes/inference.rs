use litellm_core::RouteError;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host_python::{from_py, lookup, to_py};
use litellm_http::transport::Error as TransportError;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{RouteOptions, optional_timeout, python_timeout_seconds},
};

pub(super) struct InferenceHost {
    pub request: Py<PyAny>,
    module: &'static str,
}

pub(super) struct ProjectedCall {
    pub options: RouteOptions,
    pub input: Value,
    pub params: Map<String, Value>,
}

impl InferenceHost {
    pub fn new(request: Py<PyAny>, module: &'static str) -> Self {
        Self { request, module }
    }

    pub fn project(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        input: &str,
    ) -> PyResult<ProjectedCall> {
        let request = self.request.bind(py);
        let argument = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            if let Some(value) = lookup(arguments, request, name)? {
                return Ok((!value.is_none()).then_some(value));
            }
            let parameter = request
                .getattr("parameters")?
                .call_method1("get", (name,))?;
            if !parameter.is_none() {
                return Ok(Some(parameter));
            }
            let extra = request.getattr("kwargs")?.call_method1("get", (name,))?;
            Ok((!extra.is_none()).then_some(extra))
        };
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let names: Vec<String> = py.import(self.module)?.getattr("PARAMETERS")?.extract()?;
        let params = names
            .iter()
            .filter_map(|name| match argument(name) {
                Ok(Some(value)) => Some(from_py(&value).map(|value| (name.clone(), value))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect::<PyResult<Map<String, Value>>>()?;
        let timeout = argument("timeout")?
            .or(argument("request_timeout")?)
            .map(|value| python_timeout_seconds(py, value.unbind()))
            .transpose()?
            .flatten();
        let model = string("model")?.ok_or_else(|| PyValueError::new_err("model is required"))?;
        let custom_llm_provider = string("custom_llm_provider")?;
        let provider = get_custom_llm_provider(&model, custom_llm_provider.as_deref())
            .map_or("", |resolved| resolved.custom_llm_provider);
        let (default_key, default_base): (Option<String>, Option<String>) = py
            .import(self.module)?
            .getattr("connection_defaults")?
            .call1((provider,))?
            .extract()?;
        Ok(ProjectedCall {
            options: RouteOptions {
                model,
                api_key: string("api_key")?
                    .filter(|key| !key.is_empty())
                    .or(default_key),
                api_base: string("api_base")?
                    .filter(|base| !base.is_empty())
                    .or(string("base_url")?.filter(|base| !base.is_empty()))
                    .or(default_base),
                custom_llm_provider,
                extra_headers: argument("extra_headers")?
                    .map(|value| from_py(&value))
                    .transpose()?,
                timeout: optional_timeout(timeout),
            },
            input: from_py(
                &argument(input)?
                    .ok_or_else(|| PyValueError::new_err(format!("{input} is required")))?,
            )?,
            params,
        })
    }

    pub fn response(&self, py: Python<'_>, response: &impl Serialize) -> PyResult<Py<PyAny>> {
        py.import(self.module)?
            .getattr("response")?
            .call1((to_py(py, response)?,))
            .map(Bound::unbind)
    }

    pub fn error(&self, py: Python<'_>, error: RouteError) -> PyResult<PyErr> {
        if let RouteError::Secret(source) = &error
            && let Some(original) = crate::secrets::python_error(py, source.source_error())
        {
            return Ok(original);
        }
        let native = match error {
            RouteError::Transport(TransportError::Http { status, body }) => {
                let error = RustUpstreamError::new_err((status, body));
                error
                    .value(py)
                    .setattr("headers", Vec::<(String, String)>::new())?;
                error
            }
            other => route_error_to_pyerr(other),
        };
        let mapped = py
            .import(self.module)?
            .getattr("map_failure")?
            .call1((native.value(py), self.request.bind(py)))?;
        Ok(PyErr::from_value(mapped))
    }
}
