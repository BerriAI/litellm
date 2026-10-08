use std::sync::Arc;

use litellm_auth::{AuthServices, InputSource, SecretValue, Sourced};
use litellm_cache_response::{CachePolicy, CacheScope, ScopedCache};
use litellm_callbacks_legacy_python::LoggingOperation;
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host::{call::HostedMachine, protocol::Protocol};
use litellm_host_python::{PythonBinding, PythonHostCalls, from_py, present};
use litellm_http::transport::Error as TransportError;
use litellm_inference::{Connection, RouteError};
use litellm_secrets::source::SecretSource;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::Serialize;
use serde_json::{Map, Value};

use super::NativeCall;
use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{
        optional_timeout, project_optional_fields, public_response, python_timeout_seconds,
        request_input_sources,
    },
};

pub(super) struct InferenceHost {
    pub request: Py<PyDict>,
    module: &'static str,
}

impl InferenceHost {
    pub fn new(request: Py<PyDict>, module: &'static str) -> Self {
        Self { request, module }
    }

    pub fn string(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        name: &str,
    ) -> PyResult<Option<String>> {
        self.argument(py, arguments, name)?
            .map(|value| value.extract())
            .transpose()
    }

    pub fn model(&self, py: Python<'_>, arguments: &Bound<'_, PyDict>) -> PyResult<String> {
        self.string(py, arguments, "model")?
            .ok_or_else(|| PyValueError::new_err("model is required"))
    }

    pub fn required(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        name: &str,
    ) -> PyResult<Value> {
        let value = self
            .argument(py, arguments, name)?
            .ok_or_else(|| PyValueError::new_err(format!("{name} is required")))?;
        from_py(&value)
    }

    pub fn connection(
        &self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        model: &str,
        custom_llm_provider: Option<&str>,
    ) -> PyResult<Connection> {
        let argument = |name: &str| self.argument(py, arguments, name);
        let string = |name: &str| self.string(py, arguments, name);
        let timeout = argument("timeout")?
            .or(argument("request_timeout")?)
            .map(|value| python_timeout_seconds(py, value.unbind()))
            .transpose()?
            .flatten();
        let provider = get_custom_llm_provider(model, custom_llm_provider)
            .map_or("", |resolved| resolved.custom_llm_provider);
        let (default_key, default_base): (Option<String>, Option<String>) = py
            .import(self.module)?
            .getattr("connection_defaults")?
            .call1((provider,))?
            .extract()?;
        let sources = request_input_sources(
            arguments,
            ["api_key", "api_base", "base_url", "extra_headers"].into_iter(),
        )?;
        let source = |name: &str| sources.get(name).copied().unwrap_or_default();
        let sourced = |name: &str| -> PyResult<Option<Sourced<String>>> {
            Ok(string(name)?
                .filter(|value| !value.is_empty())
                .map(|value| Sourced::new(value, source(name))))
        };
        let environment = |value| Sourced::new(value, InputSource::Environment);
        Ok(Connection {
            api_key: sourced("api_key")?
                .or(default_key.map(environment))
                .map(|key| key.map(SecretValue::new)),
            api_base: sourced("api_base")?
                .or(sourced("base_url")?)
                .or(default_base.map(environment)),
            extra_headers: argument("extra_headers")?
                .map(|value| from_py(&value))
                .transpose()?
                .map(|headers| Sourced::new(headers, source("extra_headers"))),
            timeout: optional_timeout(timeout),
        })
    }

    pub fn argument<'py>(
        &self,
        py: Python<'py>,
        arguments: &Bound<'py, PyDict>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        present(arguments, self.request.bind(py), name)
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
            .call1((native.value(py), self.request.bind(py)))?;
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MODULE: &str = "native_connection_test_host";

    #[rstest]
    #[case::caller_values_are_request(
        "{'api_key': 'caller-key', 'api_base': 'https://caller.test', 'proxy_server_request': {'body_fields': ['api_key', 'api_base']}}",
        ("caller-key", InputSource::Request),
        ("https://caller.test", InputSource::Request)
    )]
    #[case::bound_values_are_deployment(
        "{'api_key': 'deployment-key', 'api_base': 'https://deployment.test', 'proxy_server_request': {'body_fields': []}}",
        ("deployment-key", InputSource::Deployment),
        ("https://deployment.test", InputSource::Deployment)
    )]
    #[case::base_url_keeps_its_own_source(
        "{'base_url': 'https://caller.test', 'proxy_server_request': {'body_fields': ['base_url']}}",
        ("default-key", InputSource::Environment),
        ("https://caller.test", InputSource::Request)
    )]
    #[case::module_defaults_are_environment(
        "{'api_key': '', 'api_base': ''}",
        ("default-key", InputSource::Environment),
        ("https://default.test", InputSource::Environment)
    )]
    fn connection_records_where_each_value_came_from(
        #[case] arguments: &str,
        #[case] api_key: (&str, InputSource),
        #[case] api_base: (&str, InputSource),
    ) {
        Python::initialize();
        Python::attach(|py| {
            py.run(
                c"
import sys, types
module = types.ModuleType('native_connection_test_host')
module.connection_defaults = lambda provider: ('default-key', 'https://default.test')
sys.modules['native_connection_test_host'] = module
",
                None,
                None,
            )
            .unwrap();
            let arguments = py
                .eval(&std::ffi::CString::new(arguments).unwrap(), None, None)
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let connection = InferenceHost::new(PyDict::new(py).unbind(), MODULE)
                .connection(py, &arguments, "anthropic/claude", None)
                .unwrap();
            let key = connection.api_key.unwrap();
            let base = connection.api_base.unwrap();
            assert_eq!((key.value().expose(), key.source()), api_key);
            assert_eq!((base.value().as_str(), base.source()), api_base);
        });
    }
}
