use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_cache_response::{CachePolicy, CacheScope, ScopedCache};
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host::{call::HostedMachine, protocol::Protocol};
use litellm_host_python::{PythonBinding, PythonHostCalls, from_py};
use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use litellm_secrets::source::SecretSource;
use pyo3::{
    exceptions::PyValueError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde::Serialize;
use serde_json::{Map, Value};

use super::{NativeCall, RequestView};
use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{
        RouteOptions, optional_timeout, project_optional_fields, public_response,
        python_timeout_seconds,
    },
};

pub(super) struct InferenceHost {
    view: RequestView,
    module: &'static str,
}

pub(super) struct ProjectedCall {
    pub options: RouteOptions,
    pub input: Value,
    pub params: Map<String, Value>,
}

impl InferenceHost {
    pub fn new(view: RequestView, module: &'static str) -> Self {
        Self { view, module }
    }

    pub fn project(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        input: &str,
    ) -> PyResult<ProjectedCall> {
        self.view.resolve(py, arguments)?;
        let argument = |name: &str| self.argument(py, arguments, name);
        let string = |name: &str| -> PyResult<Option<String>> {
            argument(name)?.map(|value| value.extract()).transpose()
        };
        let params = self.parameters(py, arguments)?;
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
        self.view.argument(py, arguments, name)
    }

    pub fn close(&mut self) {
        self.view.close();
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.view.traverse(visit)
    }

    pub fn parameters(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Map<String, Value>> {
        let names: Vec<String> = py.import(self.module)?.getattr("PARAMETERS")?.extract()?;
        project_optional_fields(names.iter().map(String::as_str), |name| {
            self.argument(py, arguments, name)
        })
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
            .call1((native.value(py), self.view.request(py)?))?;
        Ok(PyErr::from_value(mapped))
    }
}

pub(super) trait InferenceRoute: Sized + 'static {
    type Protocol: Protocol<Error = RouteError>;
    const OPERATION: LoggingOperation;
    const SYNC_CALL_TYPE: &'static str;
    const ASYNC_CALL_TYPE: &'static str;

    fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self;
    fn with_cache(self, cache: ScopedCache) -> Self;
    fn machine(
        self,
        call: <Self::Protocol as Protocol>::Request,
        policy: CachePolicy,
    ) -> HostedMachine<Self::Protocol>;
}

pub(super) fn run_inference<R, H>(
    py: Python<'_>,
    call: NativeCall<'_>,
    asynchronous: bool,
    host: H,
) -> PyResult<Py<PyAny>>
where
    R: InferenceRoute,
    H: PythonBinding<Protocol = R::Protocol> + PythonHostCalls<R::Protocol> + 'static,
{
    let call_type = if asynchronous {
        R::ASYNC_CALL_TYPE
    } else {
        R::SYNC_CALL_TYPE
    };
    crate::cache::admit_native(py, &call.kwargs, call_type)?;
    let (arguments, hooks) = super::call_hooks(py, R::OPERATION, &call, asynchronous)?;
    super::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = R::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            let (cache, cache_options) = crate::cache::configured_native(py, arguments, call_type)?;
            let route = match cache {
                Some(cache) => route.with_cache(ScopedCache::new(cache, CacheScope::Shared)),
                None => route,
            };
            Ok(route.machine(request, cache_options.policy))
        },
        host,
        hooks,
        asynchronous,
    )
}
