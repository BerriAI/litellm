use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use litellm_auth::InputSource;
use litellm_core::call_arguments::ArgumentSpec;
use litellm_python_interop::from_py_preserving_errors as from_py;

use crate::auth::{AZURE_AD_TOKEN_PROVIDER, PythonTokenProvider};
use crate::lifecycle::BoundArguments;

pub(crate) struct Projection<Native, Retained> {
    pub native: Native,
    pub retained: Retained,
}

/// Fields every lifecycle route reads from its bound `*args, **kwargs` before
/// asking core to build the typed request. Route-specific inputs (for example
/// the OCR `document`) are read separately by the route.
pub(crate) struct BoundRouteInputs {
    pub(crate) model: String,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) extra_headers: Map<String, Value>,
    pub(crate) timeout: Option<Duration>,
    /// Only the optional params core reports as consumed plus opaque unknown
    /// values; bound/control fields are never serialized.
    pub(crate) optional_params: Map<String, Value>,
    /// Which projected fields were supplied via the proxy request body.
    pub(crate) input_sources: BTreeMap<String, InputSource>,
    pub(crate) azure_ad_token_provider: Option<PythonTokenProvider>,
    /// Consumed params whose values must be redacted from logging.
    pub(crate) secret_fields: Vec<&'static str>,
}

const CONNECTION_FIELDS: [&str; 3] = ["api_key", "api_base", "extra_headers"];

impl BoundRouteInputs {
    /// `consumed` is the route's provider-selected optional param list for the
    /// already-extracted `model`/`custom_llm_provider`; `bound_fields` are the
    /// route's positional parameter names that must not be projected.
    pub(crate) fn extract(
        py: Python<'_>,
        arguments: &BoundArguments<'_>,
        consumed: Vec<ArgumentSpec>,
        bound_fields: &[&str],
    ) -> PyResult<Self> {
        let kwargs = arguments.kwargs();
        let optional_params = project_optional_fields(kwargs, &consumed, bound_fields)?;
        let input_sources = request_input_sources(
            kwargs,
            optional_params
                .keys()
                .map(String::as_str)
                .chain(CONNECTION_FIELDS),
        )?;
        Ok(Self {
            model: arguments.extract("model")?,
            custom_llm_provider: arguments.optional("custom_llm_provider")?,
            api_key: arguments.optional("api_key")?,
            api_base: arguments.optional("api_base")?,
            extra_headers: arguments
                .optional::<Bound<'_, PyAny>>("extra_headers")?
                .map(|value| {
                    from_py(&value).and_then(|value| required_object("extra_headers", value))
                })
                .transpose()?
                .unwrap_or_default(),
            timeout: bound_timeout(py, arguments)?,
            optional_params,
            input_sources,
            azure_ad_token_provider: kwargs.get_item("azure_ad_token_provider")?.and_then(
                |provider| PythonTokenProvider::select(provider, AZURE_AD_TOKEN_PROVIDER),
            ),
            secret_fields: consumed
                .into_iter()
                .filter(|spec| spec.secret)
                .map(|spec| spec.name)
                .collect(),
        })
    }
}

/// Reads `timeout` through `litellm.rust_bridge.timeouts.timeout_to_seconds`
/// so Python `httpx.Timeout` objects keep their existing semantics. Unlike the
/// value-route [`optional_timeout`], non-finite or negative values are
/// rejected rather than silently dropped.
fn bound_timeout(py: Python<'_>, arguments: &BoundArguments<'_>) -> PyResult<Option<Duration>> {
    arguments
        .optional::<Bound<'_, PyAny>>("timeout")?
        .map(|value| python_timeout_seconds(py, value.unbind()))
        .transpose()?
        .flatten()
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err("timeout must be a non-negative finite number"))
        })
        .transpose()
}

pub(crate) struct RouteOptions {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) timeout: Option<Duration>,
}

pub(crate) struct RouteOptionsInputs {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) extra_headers: Option<Value>,
    pub(crate) timeout_seconds: Option<f64>,
}

impl RouteOptions {
    pub(crate) fn from_python(inputs: RouteOptionsInputs) -> PyResult<Self> {
        Ok(Self {
            model: inputs.model,
            api_key: inputs.api_key,
            api_base: inputs.api_base,
            custom_llm_provider: inputs.custom_llm_provider,
            extra_headers: optional_object("extra_headers", inputs.extra_headers)?,
            timeout: optional_timeout(inputs.timeout_seconds),
        })
    }
}

pub(crate) fn required_array(name: &'static str, value: Value) -> PyResult<Vec<Value>> {
    match value {
        Value::Array(values) => Ok(values),
        _ => Err(PyValueError::new_err(format!("{name} must be a list"))),
    }
}

pub(crate) fn required_object(name: &'static str, value: Value) -> PyResult<Map<String, Value>> {
    match value {
        Value::Object(values) => Ok(values),
        _ => Err(PyValueError::new_err(format!("{name} must be a dict"))),
    }
}

pub(crate) fn object_or_empty(
    name: &'static str,
    value: Option<Value>,
) -> PyResult<Map<String, Value>> {
    match value {
        Some(value) => required_object(name, value),
        None => Ok(Map::new()),
    }
}

fn optional_object(
    name: &'static str,
    value: Option<Value>,
) -> PyResult<Option<Map<String, Value>>> {
    value.map(|value| required_object(name, value)).transpose()
}

pub(crate) fn optional_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

pub(crate) fn python_timeout_seconds(py: Python<'_>, timeout: Py<PyAny>) -> PyResult<Option<f64>> {
    py.import("litellm.rust_bridge.timeouts")?
        .getattr("timeout_to_seconds")?
        .call1((timeout,))?
        .extract()
}

pub(crate) fn project_optional_fields(
    kwargs: &Bound<'_, PyDict>,
    fields: &[litellm_core::call_arguments::ArgumentSpec],
    bound_fields: &[&str],
) -> PyResult<Map<String, Value>> {
    kwargs
        .iter()
        .map(|(name, value)| Ok((name.extract::<String>()?, value)))
        .filter_map(|entry: PyResult<_>| match entry {
            Ok((name, value))
                if litellm_core::call_arguments::should_project(&name, fields, bound_fields) =>
            {
                Some(from_py(&value).map(|value| (name, value)))
            }
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

struct RequestFieldSources<'py> {
    body: Option<Bound<'py, PyAny>>,
    credentials: Option<Bound<'py, PyAny>>,
}

impl<'py> RequestFieldSources<'py> {
    fn extract(proxy_request: &Bound<'py, PyAny>) -> PyResult<Self> {
        let proxy_request = proxy_request.cast::<PyDict>()?;

        let body = proxy_request
            .get_item("body_fields")?
            .or(proxy_request.get_item("body")?);

        let credentials = proxy_request.get_item("credential_fields")?;

        Ok(Self { body, credentials })
    }

    fn contains(&self, name: &str) -> bool {
        self.body
            .as_ref()
            .is_some_and(|fields| fields.contains(name).unwrap_or(false))
            || self
                .credentials
                .as_ref()
                .is_some_and(|fields| fields.contains(name).unwrap_or(false))
    }
}

pub(crate) fn request_input_sources<'a>(
    kwargs: &Bound<'_, PyDict>,
    names: impl Iterator<Item = &'a str>,
) -> PyResult<BTreeMap<String, InputSource>> {
    let Some(proxy_request) = kwargs.get_item("proxy_server_request")? else {
        return Ok(BTreeMap::new());
    };

    let sources = RequestFieldSources::extract(&proxy_request)?;

    Ok(names
        .filter(|name| sources.contains(name))
        .map(|name| (name.to_string(), InputSource::Request))
        .collect())
}

pub(crate) fn marshal_headers(headers: Option<Value>) -> PyResult<HashMap<String, String>> {
    let value = match headers {
        Some(headers) => headers,
        None => Value::Object(Map::new()),
    };
    let Value::Object(headers) = value else {
        return Err(PyValueError::new_err("headers must be a dict"));
    };
    headers
        .into_iter()
        .map(|(name, value)| {
            value
                .as_str()
                .map(|value| (name, value.to_string()))
                .ok_or_else(|| PyValueError::new_err("header values must be strings"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyTypeError;
    use serde_json::json;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn sources(
        py: Python<'_>,
        proxy: &Bound<'_, PyAny>,
        names: &[&str],
    ) -> PyResult<BTreeMap<String, InputSource>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("proxy_server_request", proxy)?;
        request_input_sources(&kwargs, names.iter().copied())
    }

    #[test]
    fn required_shapes_preserve_nested_values_and_existing_errors() {
        Python::initialize();
        let nested = json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]);
        assert_eq!(
            Value::Array(required_array("messages", nested.clone()).unwrap()),
            nested
        );

        let body = json!({"model": "claude", "metadata": {"user": "1"}});
        assert_eq!(
            Value::Object(required_object("body", body.clone()).unwrap()),
            body
        );

        assert_eq!(
            required_array("messages", json!({"role": "user"}))
                .unwrap_err()
                .to_string(),
            "ValueError: messages must be a list"
        );
        assert_eq!(
            required_object("body", json!([])).unwrap_err().to_string(),
            "ValueError: body must be a dict"
        );
    }

    #[test]
    fn optional_parameters_treat_missing_as_empty() {
        assert_eq!(
            object_or_empty("optional_params", None).unwrap(),
            Map::new()
        );
        assert_eq!(
            object_or_empty("optional_params", Some(json!({"temperature": 0.2}))).unwrap(),
            required_object("optional_params", json!({"temperature": 0.2})).unwrap()
        );
    }

    #[test]
    fn missing_none_and_empty_proxy_metadata_are_distinct() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            assert!(
                request_input_sources(&kwargs, ["api_key"].into_iter())
                    .unwrap()
                    .is_empty()
            );

            kwargs.set_item("proxy_server_request", py.None()).unwrap();
            assert!(
                request_input_sources(&kwargs, ["api_key"].into_iter())
                    .unwrap_err()
                    .is_instance_of::<PyTypeError>(py)
            );

            kwargs
                .set_item("proxy_server_request", PyDict::new(py))
                .unwrap();
            assert!(
                request_input_sources(&kwargs, ["api_key"].into_iter())
                    .unwrap()
                    .is_empty()
            );
        });
    }

    #[test]
    fn body_fields_win_over_body_and_explicit_none_does_not_fall_back() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
proxy = {'body_fields': ['api_key'], 'body': ['api_base']}
none_fields = {'body_fields': None, 'body': ['api_key']}
body_only = {'body': ['api_base']}
",
            );
            let named = sources(
                py,
                &locals.get_item("proxy").unwrap().unwrap(),
                &["api_key", "api_base"],
            )
            .unwrap();
            assert_eq!(named.get("api_key").copied(), Some(InputSource::Request));
            assert!(!named.contains_key("api_base"));

            assert!(
                sources(
                    py,
                    &locals.get_item("none_fields").unwrap().unwrap(),
                    &["api_key"],
                )
                .unwrap()
                .is_empty()
            );

            let body_only = sources(
                py,
                &locals.get_item("body_only").unwrap().unwrap(),
                &["api_base"],
            )
            .unwrap();
            assert_eq!(
                body_only.get("api_base").copied(),
                Some(InputSource::Request)
            );
        });
    }

    #[test]
    fn body_and_credential_membership_can_mark_request_fields() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
class Raising:
    def __contains__(self, item):
        raise RuntimeError('credential membership')
proxy = {
    'body_fields': ['api_key'],
    'credential_fields': Raising(),
}
credentials_only = {'credential_fields': ['extra_headers']}
erroring = {'body_fields': Raising()}
extra = {'body_fields': ['api_key', 'unused']}
",
            );
            let skipped = sources(
                py,
                &locals.get_item("proxy").unwrap().unwrap(),
                &["api_key"],
            )
            .unwrap();
            assert_eq!(skipped.get("api_key").copied(), Some(InputSource::Request));

            let credentials = sources(
                py,
                &locals.get_item("credentials_only").unwrap().unwrap(),
                &["extra_headers"],
            )
            .unwrap();
            assert_eq!(
                credentials.get("extra_headers").copied(),
                Some(InputSource::Request)
            );

            assert!(
                sources(
                    py,
                    &locals.get_item("erroring").unwrap().unwrap(),
                    &["api_key"],
                )
                .unwrap()
                .is_empty()
            );

            let requested = sources(
                py,
                &locals.get_item("extra").unwrap().unwrap(),
                &["api_key"],
            )
            .unwrap();
            assert_eq!(requested.len(), 1);
            assert_eq!(
                requested.get("api_key").copied(),
                Some(InputSource::Request)
            );
        });
    }
}
