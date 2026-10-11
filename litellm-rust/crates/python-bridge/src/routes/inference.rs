use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_cache_response::ScopedCache;
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host::{
    call::{OutputOf, hosted_call},
    machine::ChannelInterceptors,
    protocol::Protocol,
};
use litellm_host_python::{
    InvokeError, PythonBinding, PythonHostCalls, PythonLifecycle, PythonTask, from_py,
};
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
    cache::{CacheCall, Cached, PythonCache, Selection},
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{
        RouteOptions, optional_timeout, project_optional_fields, public_response,
        python_timeout_seconds,
    },
};

pub(super) struct InferenceHost {
    view: RequestView,
    module: &'static str,
    route: &'static str,
    call_type: &'static str,
    cache: PythonCache,
}

pub(super) struct ProjectedCall {
    pub options: RouteOptions,
    pub input: Value,
    pub params: Map<String, Value>,
}

impl InferenceHost {
    pub fn new(
        view: RequestView,
        module: &'static str,
        route: &'static str,
        call_type: &'static str,
    ) -> Self {
        Self {
            view,
            module,
            route,
            call_type,
            cache: PythonCache::new(),
        }
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

    pub fn select_cache(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Selection> {
        crate::cache::configure(&mut self.cache, py, arguments, self.call_type)
    }

    pub async fn cache_call<E>(
        &mut self,
        task: &PythonTask,
        op: CacheCall,
    ) -> Result<(), InvokeError<E>> {
        self.cache.call(task, op).await
    }

    /// The caller's resolved request, as the argument stage left it.
    pub fn request<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyDict>> {
        self.view.request(py)
    }

    pub fn close(&mut self) {
        self.view.close();
        self.cache.close();
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.view.traverse(visit)?;
        self.cache.traverse(visit)
    }

    pub fn parameters(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
    ) -> PyResult<Map<String, Value>> {
        let names = super::provider_parameters(py, self.route)?;
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

pub(super) trait InferenceRoute: Sized + Send + 'static {
    type Protocol: litellm_inference::caching::Cachable;
    const OPERATION: LoggingOperation;
    /// The Python module whose `drive` and stream classes wrap this route's executions.
    const LIFECYCLE: PythonLifecycle = crate::lifecycle::binding;

    fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self;
    fn with_cache(self, cache: ScopedCache) -> Self;
    fn execute(
        self,
        call: <Self::Protocol as Protocol>::Request,
        interceptors: ChannelInterceptors<Cached<Self::Protocol>>,
        options: litellm_inference::CallOptions,
    ) -> impl Future<Output = Result<OutputOf<Self::Protocol>, RouteError>> + Send;
}

pub(super) fn run_inference<R, H>(
    py: Python<'_>,
    call: NativeCall<'_>,
    asynchronous: bool,
    host: H,
) -> PyResult<Py<PyAny>>
where
    R: InferenceRoute,
    H: PythonBinding<Protocol = Cached<R::Protocol>>
        + PythonHostCalls<Cached<R::Protocol>>
        + 'static,
{
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
            Ok(hosted_call(
                request,
                None,
                move |(call, selection): (_, Selection), services, interceptors, observers| async move {
                    let (cache, options) = selection.attach(services);
                    let route = match cache {
                        Some(cache) => route.with_cache(cache),
                        None => route,
                    };
                    route
                        .execute(
                            call,
                            interceptors,
                            litellm_inference::CallOptions {
                                cache: Some(options.policy),
                                observers,
                            },
                        )
                        .await
                },
            ))
        },
        host,
        hooks,
        asynchronous,
        R::LIFECYCLE,
    )
}
