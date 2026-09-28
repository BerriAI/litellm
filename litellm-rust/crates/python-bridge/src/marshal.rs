use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use litellm_auth::InputSource;
use litellm_host_python::{from_py, from_py_argument};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

/// The keyword arguments every value route shares, validated at the Python boundary.
pub(crate) struct RouteOptions {
    pub(crate) model: String,
    pub(crate) api_key: Option<String>,
    pub(crate) api_base: Option<String>,
    pub(crate) custom_llm_provider: Option<String>,
    pub(crate) extra_headers: Option<Map<String, Value>>,
    pub(crate) timeout: Option<Duration>,
}

pub(crate) fn messages_argument(value: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    match from_py_argument(value)? {
        Value::Array(values) => Ok(values),
        _ => Err(PyValueError::new_err("messages must be a list")),
    }
}

pub(crate) fn optional_params_argument(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<Map<String, Value>>> {
    optional_object("optional_params", value)
}

pub(crate) fn extra_headers_argument(
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<Map<String, Value>>> {
    optional_object("extra_headers", value)
}

fn required_object(name: &'static str, value: Value) -> PyResult<Map<String, Value>> {
    match value {
        Value::Object(values) => Ok(values),
        _ => Err(PyValueError::new_err(format!("{name} must be a dict"))),
    }
}

fn optional_object(
    name: &'static str,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<Map<String, Value>>> {
    if value.is_none() {
        return Ok(None);
    }
    required_object(name, from_py_argument(value)?).map(Some)
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
    names: &[&str],
) -> PyResult<Map<String, Value>> {
    names
        .iter()
        .filter_map(|name| match kwargs.get_item(name) {
            Ok(Some(value)) => Some(from_py(&value).map(|value| ((*name).to_string(), value))),
            Ok(None) => None,
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
    use pyo3::exceptions::PyTypeError;
    use serde_json::json;

    use super::*;

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

    #[serde_with::serde_as]
    #[derive(Debug, serde::Deserialize, serde::Serialize, PartialEq)]
    struct Numbers {
        #[serde_as(deserialize_as = "Option<Vec<litellm_core_utils::serde_compat::LaxI64>>")]
        integers: Option<Vec<i64>>,
        #[serde_as(deserialize_as = "Option<litellm_core_utils::serde_compat::FiniteF64>")]
        float: Option<f64>,
    }

    #[test]
    fn numeric_adapters_agree_across_json_and_python_boundaries() {
        Python::initialize();
        Python::attach(|py| {
            for input in [
                json!({}),
                json!({"integers": null, "float": null}),
                json!({"integers": [i64::MIN, i64::MAX, "9007199254740993.0", " +1_000.00 ", true, 3.0], "float": " 1.25 "}),
                json!({"integers": [u64::MAX]}),
                json!({"integers": ["1.0000000000000001"]}),
                json!({"integers": [2.5]}),
                json!({"float": "NaN"}),
                json!({"float": "inf"}),
                json!({"float": "1e999"}),
                json!({"float": true}),
                json!({"float": u64::MAX}),
            ] {
                let expected = serde_json::from_value::<Numbers>(input.clone());
                let python = litellm_host_python::to_py(py, &input).unwrap();
                let actual = from_py::<Numbers>(python.bind(py));
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(actual, expected);
                        let serialized = litellm_host_python::to_py(py, &actual).unwrap();
                        assert_eq!(
                            from_py::<Value>(serialized.bind(py)).unwrap(),
                            serde_json::to_value(expected).unwrap()
                        );
                    }
                    (Err(_), Err(_)) => {}
                    mismatch => panic!("boundary mismatch for {input}: {mismatch:?}"),
                }
            }
            for source in [
                c"{'float': float('nan')}",
                c"{'float': float('inf')}",
                c"{'integers': [float('inf')]}",
                c"{'integers': [2 ** 100]}",
            ] {
                let value = py.eval(source, None, None).unwrap();
                assert!(from_py::<Numbers>(&value).is_err());
            }
        });
    }

    #[test]
    fn argument_converters_keep_nested_values_and_accept_explicit_none() {
        Python::initialize();
        Python::attach(|py| {
            let messages = py
                .eval(
                    c"[{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]",
                    None,
                    None,
                )
                .unwrap();
            assert_eq!(
                Value::Array(messages_argument(&messages).unwrap()),
                json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
            );

            let params = py.eval(c"{'temperature': 0.2}", None, None).unwrap();
            assert_eq!(
                optional_params_argument(&params).unwrap(),
                Some(required_object("optional_params", json!({"temperature": 0.2})).unwrap())
            );
            assert_eq!(
                optional_params_argument(&py.None().into_bound(py)).unwrap(),
                None
            );
            assert_eq!(
                extra_headers_argument(&py.None().into_bound(py)).unwrap(),
                None
            );
        });
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
