use std::collections::BTreeSet;

use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host_python::{from_py, lookup};
use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{RouteOptions, optional_timeout, public_response, python_timeout_seconds},
};

pub(super) struct InferenceHost {
    pub request: Py<PyDict>,
    module: &'static str,
    defaulted: BTreeSet<String>,
}

pub(super) struct ProjectedCall {
    pub options: RouteOptions,
    pub input: Value,
    pub params: Map<String, Value>,
}

impl InferenceHost {
    pub fn new(
        request: Bound<'_, PyDict>,
        module: &'static str,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        Ok(Self {
            defaulted: super::parameters::defaulted_keys(&request, kwargs)?,
            request: request.unbind(),
            module,
        })
    }

    pub fn project(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        input: &str,
    ) -> PyResult<ProjectedCall> {
        let argument = |name: &str| self.argument(py, arguments, name);
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let params = self.parameters(py, arguments, input)?;
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

    pub fn argument<'py>(
        &self,
        py: Python<'py>,
        arguments: &Bound<'py, PyDict>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(lookup(arguments, self.request.bind(py).as_any(), name)?
            .filter(|value| !value.is_none()))
    }

    pub fn parameters(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        input: &str,
    ) -> PyResult<Map<String, Value>> {
        super::parameters::provider_parameters(
            py,
            arguments,
            self.request.bind(py),
            &self.defaulted,
            &[input],
            &["metadata"],
        )
        .map(Into::into)
    }

    pub fn response(&self, py: Python<'_>, response: &impl Serialize) -> PyResult<Py<PyAny>> {
        public_response(py, self.module, response)
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
