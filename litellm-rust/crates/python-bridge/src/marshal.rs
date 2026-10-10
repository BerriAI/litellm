use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use litellm_auth::InputSource;
use litellm_host_python::{from_py, from_py_argument, to_py};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde::{Serialize, de::DeserializeOwned};
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

pub(crate) fn required_field<'py>(
    fields: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    fields
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("{name} is required")))
}

pub(crate) fn optional_field<T: DeserializeOwned>(
    fields: &Bound<'_, PyDict>,
    name: &str,
) -> PyResult<Option<T>> {
    fields
        .get_item(name)?
        .map(|value| from_py_argument(&value))
        .transpose()
        .map(Option::flatten)
}

pub(crate) fn optional_object_field(
    fields: &Bound<'_, PyDict>,
    name: &'static str,
) -> PyResult<Option<Map<String, Value>>> {
    fields
        .get_item(name)?
        .map(|value| optional_object(name, &value))
        .transpose()
        .map(Option::flatten)
}

pub(crate) fn value_route_options(fields: &Bound<'_, PyDict>) -> PyResult<RouteOptions> {
    Ok(RouteOptions {
        model: from_py_argument(&required_field(fields, "model")?)?,
        api_key: optional_field(fields, "api_key")?,
        api_base: optional_field(fields, "api_base")?,
        custom_llm_provider: optional_field(fields, "custom_llm_provider")?,
        extra_headers: optional_object_field(fields, "extra_headers")?,
        timeout: optional_timeout(optional_field(fields, "timeout_seconds")?),
    })
}

/// Builds a route's optional body fields from the caller's Python arguments, in `names` order.
/// `lookup` decides what counts as unset: a name it returns `None` for is left out of the map.
pub(crate) fn project_optional_fields<'a, 'py>(
    names: impl IntoIterator<Item = &'a str>,
    lookup: impl Fn(&str) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<Map<String, Value>> {
    names
        .into_iter()
        .filter_map(|name| match lookup(name) {
            Ok(Some(value)) => Some(from_py(&value).map(|value| (name.to_string(), value))),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

/// Converts a Rust route response to Python and returns `module.response(...)` called on it,
/// so each route's Python factory builds the public LiteLLM response object.
pub(crate) fn public_response(
    py: Python<'_>,
    module: &str,
    response: &(impl Serialize + ?Sized),
) -> PyResult<Py<PyAny>> {
    py.import(module)?
        .getattr("response")?
        .call1((to_py(py, response)?,))
        .map(Bound::unbind)
}

pub(crate) fn provider_metadata(
    py: Python<'_>,
    response: Py<PyAny>,
    headers: &[(String, String)],
) -> PyResult<Py<PyAny>> {
    let forwarded = litellm_http::request::response_headers(
        &litellm_http::response::forwarded_headers(headers),
    );
    py.import("litellm.rust_bridge.response_metadata")?
        .getattr("with_provider_headers")?
        .call1((response, to_py(py, headers)?, to_py(py, &forwarded)?))
        .map(Bound::unbind)
}

pub(crate) fn public_provider_response<T: Serialize>(
    py: Python<'_>,
    module: &str,
    response: litellm_http::response::ProviderResponse<T>,
) -> PyResult<Py<PyAny>> {
    let public = public_response(py, module, &response.body)?;
    provider_metadata(py, public, &response.headers)
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    #[rstest]
    #[case::keep_none(false)]
    #[case::skip_none(true)]
    fn selected_fields_preserve_lookup_order_and_caller_none_policy(#[case] skip_none: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
reads = []
fields = {'first': {'future': [None, True, 2]}, 'second': None, 'unused': object()}
def lookup(name):
    reads.append(name)
    if name == 'first':
        fields['last'] = 'observed after first'
    return fields.get(name)
",
            );
            let lookup = locals.get_item("lookup").unwrap().unwrap();
            let result = project_optional_fields(["first", "second", "last"], |name| {
                let value = lookup.call1((name,))?;
                Ok((!skip_none || !value.is_none()).then_some(value))
            })
            .unwrap();
            let expected = if skip_none {
                json!({"first": {"future": [null, true, 2]}, "last": "observed after first"})
            } else {
                json!({"first": {"future": [null, true, 2]}, "second": null, "last": "observed after first"})
            };
            assert_eq!(Value::Object(result), expected);
            assert_eq!(
                locals
                    .get_item("reads")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["first", "second", "last"]
            );
        });
    }

    #[rstest]
    fn selected_field_failure_stops_lookup_and_keeps_python_provenance() {
        Python::initialize();
        Python::attach(|py| {
            let locals = eval(
                py,
                c"
reads = []
failure = LookupError('selected field failed')
cause = ValueError('cause')
def lookup(name):
    reads.append(name)
    raise failure from cause
",
            );
            let lookup = locals.get_item("lookup").unwrap().unwrap();
            let error =
                project_optional_fields(["first", "later"], |name| lookup.call1((name,)).map(Some))
                    .unwrap_err();
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            assert!(
                error
                    .cause(py)
                    .unwrap()
                    .value(py)
                    .is(locals.get_item("cause").unwrap().unwrap())
            );
            assert!(error.traceback(py).is_some());
            assert_eq!(
                locals
                    .get_item("reads")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<String>>()
                    .unwrap(),
                ["first"]
            );
        });
    }

    #[rstest]
    #[case::lookup_failure(false)]
    #[case::factory_failure(true)]
    fn public_response_resolves_factory_before_serializing_and_keeps_its_errors(
        #[case] factory: bool,
    ) {
        struct Observed<'a>(&'a std::cell::Cell<bool>);

        impl Serialize for Observed<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.set(true);
                json!({"future": [null, true]}).serialize(serializer)
            }
        }

        Python::initialize();
        Python::attach(|py| {
            let module_name = if factory {
                "bridge_response_conversion_test_factory"
            } else {
                "bridge_response_conversion_test_lookup"
            };
            let locals = eval(
                py,
                c"
import types
failure = LookupError('response failed')
cause = ValueError('cause')
received = []
def fail(name):
    raise failure from cause
def response(value):
    received.append(value)
    return fail('response')
module = types.ModuleType('bridge_response_conversion_test')
module.__getattr__ = fail
",
            );
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item(module_name, locals.get_item("module").unwrap().unwrap())
                .unwrap();
            if factory {
                locals
                    .get_item("module")
                    .unwrap()
                    .unwrap()
                    .setattr("response", locals.get_item("response").unwrap().unwrap())
                    .unwrap();
            }
            let serialized = std::cell::Cell::new(false);
            let error = public_response(py, module_name, &Observed(&serialized)).unwrap_err();
            assert_eq!(serialized.get(), factory);
            assert!(
                error
                    .value(py)
                    .is(locals.get_item("failure").unwrap().unwrap())
            );
            assert!(
                error
                    .cause(py)
                    .unwrap()
                    .value(py)
                    .is(locals.get_item("cause").unwrap().unwrap())
            );
            assert!(error.traceback(py).is_some());
            let received: Value = from_py(&locals.get_item("received").unwrap().unwrap()).unwrap();
            assert_eq!(
                received,
                if factory {
                    json!([{"future": [null, true]}])
                } else {
                    json!([])
                }
            );
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .del_item(module_name)
                .unwrap();
        });
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
        #[serde_as(deserialize_as = "Option<Vec<litellm_llms_types::serde_compat::LaxI64>>")]
        integers: Option<Vec<i64>>,
        #[serde_as(deserialize_as = "Option<litellm_llms_types::serde_compat::FiniteF64>")]
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
            let params = py.eval(c"{'temperature': 0.2}", None, None).unwrap();
            assert_eq!(
                optional_object("optional_params", &params).unwrap(),
                Some(required_object("optional_params", json!({"temperature": 0.2})).unwrap())
            );
            assert_eq!(
                optional_object("optional_params", &py.None().into_bound(py)).unwrap(),
                None
            );
            assert_eq!(
                optional_object("extra_headers", &py.None().into_bound(py)).unwrap(),
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
