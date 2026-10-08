use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_host_python::from_py;
use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::Serialize;
use serde_json::{Map, Value};

use super::parameters::{field, merged_request, provider_parameters};
use crate::{
    errors::{RustUpstreamError, route_error_to_pyerr},
    marshal::{RouteOptions, optional_timeout, public_response, python_timeout_seconds},
};

pub(super) struct InferenceHost {
    pub bound: Py<PyDict>,
    module: &'static str,
}

pub(super) struct ProjectedCall {
    pub options: RouteOptions,
    pub input: Value,
    pub params: Map<String, Value>,
}

impl ProjectedCall {
    pub fn streams(&self) -> bool {
        self.params.get("stream") == Some(&Value::Bool(true))
    }
}

impl InferenceHost {
    pub fn new(bound: Bound<'_, PyDict>, module: &'static str) -> Self {
        Self {
            bound: bound.unbind(),
            module,
        }
    }

    pub fn project(
        &self,
        py: Python<'_>,
        hooked: &Bound<'_, PyDict>,
        input: &str,
    ) -> PyResult<ProjectedCall> {
        let request = merged_request(self.bound.bind(py), hooked)?;
        let defaults = |provider: &str| -> PyResult<ConnectionDefaults> {
            py.import(self.module)?
                .getattr("connection_defaults")?
                .call1((provider,))?
                .extract()
        };
        let timeout_seconds = |value: Bound<'_, PyAny>| python_timeout_seconds(py, value.unbind());
        Ok(ProjectedCall {
            options: route_options(&request, &defaults, &timeout_seconds)?,
            input: from_py(
                &field(&request, input)?
                    .ok_or_else(|| PyValueError::new_err(format!("{input} is required")))?,
            )?,
            params: provider_parameters(py, &request, input)?.into(),
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
            .call1((native.value(py), self.bound.bind(py)))?;
        Ok(PyErr::from_value(mapped))
    }
}

type ConnectionDefaults = (Option<String>, Option<String>);

fn route_options(
    request: &Bound<'_, PyDict>,
    defaults: &impl Fn(&str) -> PyResult<ConnectionDefaults>,
    timeout_seconds: &impl Fn(Bound<'_, PyAny>) -> PyResult<Option<f64>>,
) -> PyResult<RouteOptions> {
    let string = |name: &str| -> PyResult<Option<String>> {
        field(request, name)?
            .map(|value| value.extract())
            .transpose()
    };
    let non_empty = |name: &str| -> PyResult<Option<String>> {
        Ok(string(name)?.filter(|value| !value.is_empty()))
    };
    let model = string("model")?.ok_or_else(|| PyValueError::new_err("model is required"))?;
    let custom_llm_provider = string("custom_llm_provider")?;
    let provider = get_custom_llm_provider(&model, custom_llm_provider.as_deref())
        .map_or("", |resolved| resolved.custom_llm_provider);
    let (default_key, default_base) = defaults(provider)?;
    let timeout = field(request, "timeout")?
        .or(field(request, "request_timeout")?)
        .map(timeout_seconds)
        .transpose()?
        .flatten();
    Ok(RouteOptions {
        api_key: non_empty("api_key")?.or(default_key),
        api_base: non_empty("api_base")?
            .or(non_empty("base_url")?)
            .or(default_base),
        extra_headers: field(request, "extra_headers")?
            .map(|value| from_py(&value))
            .transpose()?,
        timeout: optional_timeout(timeout),
        model,
        custom_llm_provider,
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn python_dict<'py>(py: Python<'py>, literal: &str) -> Bound<'py, PyDict> {
        py.eval(&CString::new(literal).unwrap(), None, None)
            .unwrap()
            .cast_into()
            .unwrap()
    }

    fn options(request: &str) -> PyResult<Value> {
        Python::initialize();
        Python::attach(|py| {
            let defaults = |provider: &str| -> PyResult<ConnectionDefaults> {
                Ok((
                    Some(format!("{provider}-default-key")),
                    Some(format!("https://{provider}.default")),
                ))
            };
            let seconds = |value: Bound<'_, PyAny>| value.extract::<f64>().map(Some);
            let options = route_options(&python_dict(py, request), &defaults, &seconds)?;
            Ok(json!({
                "model": options.model,
                "api_key": options.api_key,
                "api_base": options.api_base,
                "custom_llm_provider": options.custom_llm_provider,
                "extra_headers": options.extra_headers,
                "timeout": options.timeout.map(|timeout| timeout.as_secs_f64()),
            }))
        })
    }

    #[rstest]
    #[case::provider_defaults_fill_missing_connection(
        "{'model': 'anthropic/claude'}",
        json!({"api_key": "anthropic-default-key", "api_base": "https://anthropic.default"})
    )]
    #[case::none_connection_falls_back_to_defaults(
        "{'model': 'anthropic/claude', 'api_key': None, 'api_base': None, 'base_url': None}",
        json!({"api_key": "anthropic-default-key", "api_base": "https://anthropic.default"})
    )]
    #[case::empty_connection_falls_back_to_defaults(
        "{'model': 'anthropic/claude', 'api_key': '', 'api_base': '', 'base_url': ''}",
        json!({"api_key": "anthropic-default-key", "api_base": "https://anthropic.default"})
    )]
    #[case::caller_connection_beats_defaults(
        "{'model': 'anthropic/claude', 'api_key': 'caller-key', 'api_base': 'https://caller'}",
        json!({"api_key": "caller-key", "api_base": "https://caller"})
    )]
    #[case::api_base_beats_base_url(
        "{'model': 'anthropic/claude', 'api_base': 'https://api-base', 'base_url': 'https://base-url'}",
        json!({"api_base": "https://api-base"})
    )]
    #[case::base_url_replaces_a_missing_api_base(
        "{'model': 'anthropic/claude', 'base_url': 'https://base-url'}",
        json!({"api_base": "https://base-url"})
    )]
    #[case::base_url_replaces_an_empty_api_base(
        "{'model': 'anthropic/claude', 'api_base': '', 'base_url': 'https://base-url'}",
        json!({"api_base": "https://base-url"})
    )]
    #[case::explicit_provider_picks_the_defaults(
        "{'model': 'claude', 'custom_llm_provider': 'anthropic'}",
        json!({"custom_llm_provider": "anthropic", "api_key": "anthropic-default-key"})
    )]
    #[case::timeout_beats_request_timeout(
        "{'model': 'anthropic/claude', 'timeout': 5.0, 'request_timeout': 9.0}",
        json!({"timeout": 5.0})
    )]
    #[case::request_timeout_replaces_a_missing_timeout(
        "{'model': 'anthropic/claude', 'timeout': None, 'request_timeout': 9.0}",
        json!({"timeout": 9.0})
    )]
    #[case::zero_timeout_is_absent("{'model': 'anthropic/claude', 'timeout': 0.0}", json!({"timeout": null}))]
    #[case::negative_timeout_is_absent("{'model': 'anthropic/claude', 'timeout': -1.0}", json!({"timeout": null}))]
    #[case::extra_headers_are_kept(
        "{'model': 'anthropic/claude', 'extra_headers': {'x-trace': '1'}}",
        json!({"extra_headers": {"x-trace": "1"}})
    )]
    #[case::model_is_kept_verbatim("{'model': 'anthropic/claude'}", json!({"model": "anthropic/claude"}))]
    fn resolves_route_options(#[case] request: &str, #[case] expected: Value) {
        let options = options(request).unwrap();
        let Value::Object(expected) = expected else {
            unreachable!()
        };
        for (name, value) in expected {
            assert_eq!(options[&name], value, "{name}");
        }
    }

    #[rstest]
    #[case::missing_model("{}")]
    #[case::none_model("{'model': None}")]
    #[case::non_string_model("{'model': 1}")]
    #[case::non_string_api_key("{'model': 'anthropic/claude', 'api_key': 1}")]
    #[case::non_object_extra_headers(
        "{'model': 'anthropic/claude', 'extra_headers': 'x-trace: 1'}"
    )]
    fn invalid_route_options_are_terminal(#[case] request: &str) {
        assert!(options(request).is_err());
    }

    #[rstest]
    #[case::true_value(json!({"stream": true}), true)]
    #[case::false_value(json!({"stream": false}), false)]
    #[case::null(json!({"stream": null}), false)]
    #[case::truthy_string(json!({"stream": "true"}), false)]
    #[case::truthy_number(json!({"stream": 1}), false)]
    #[case::absent(json!({}), false)]
    fn only_a_literal_true_stream_streams(#[case] params: Value, #[case] streams: bool) {
        let Value::Object(params) = params else {
            unreachable!()
        };
        let call = ProjectedCall {
            options: RouteOptions {
                model: "model".into(),
                api_key: None,
                api_base: None,
                custom_llm_provider: None,
                extra_headers: None,
                timeout: None,
            },
            input: Value::Null,
            params,
        };
        assert_eq!(call.streams(), streams);
    }
}
