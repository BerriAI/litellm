//! The request body check for the Python proxy. This file reads the request's Python objects into
//! what `gateway-auth-core` judges, and answers what the check asks about the proxy's state from
//! the Python modules that own it. No rule lives here.

use litellm_gateway_auth_core::body::{
    self, CallerValue, Deferred, Deployment, Deployments, Failure, Host, HostTables, Names, OptIn,
    Params, Policy, Reading, Request, Surface, VERTEX_TARGET_PARAMS,
};
use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyList, PyModule, PyString},
};

const MODULE: &str = "litellm.rust_bridge.host.request_body";

/// How deep a body is read. No consumer of a body reads a parameter from further down.
const MAX_DEPTH: usize = 32;

/// The name tables the check reads, built once from the Python modules that own them.
#[pyclass(
    frozen,
    name = "RequestBodyNames",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct RequestBodyNames(Names);

#[pymethods]
impl RequestBodyNames {
    #[new]
    #[pyo3(signature = (
        *,
        callback_params,
        blocked_callback_params,
        pricing_params,
        federation_params,
        token_exchange_params,
        github_copilot_params,
        upstream_bound_params,
    ))]
    fn new(
        callback_params: Vec<String>,
        blocked_callback_params: Vec<String>,
        pricing_params: Vec<String>,
        federation_params: Vec<String>,
        token_exchange_params: Vec<String>,
        github_copilot_params: Vec<String>,
        upstream_bound_params: Vec<String>,
    ) -> Self {
        Self(Names::new(HostTables {
            callback_params,
            blocked_callback_params,
            pricing_params,
            federation_params,
            token_exchange_params,
            github_copilot_params,
            upstream_bound_params,
        }))
    }

    /// Every parameter a caller may set only with an admin's grant, in the order a rejection
    /// checks them.
    #[getter]
    fn overrides(&self) -> Vec<String> {
        self.0.overrides().to_vec()
    }

    #[getter]
    fn vertex_target(&self) -> Vec<&'static str> {
        VERTEX_TARGET_PARAMS.to_vec()
    }
}

/// The rejection the caller reads, or `None` when the proxy accepts the body.
#[pyfunction]
#[pyo3(signature = (
    body,
    *,
    names,
    general_settings,
    router,
    model,
    route,
    route_model,
    team_id,
    fallback_targets,
    defer_model_checks,
    defer_s3_buckets,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_request_body<'py>(
    body: &Bound<'py, PyAny>,
    names: &Bound<'py, RequestBodyNames>,
    general_settings: &Bound<'py, PyAny>,
    router: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
    route: Option<&str>,
    route_model: Option<&str>,
    team_id: Option<String>,
    fallback_targets: Vec<Bound<'py, PyAny>>,
    defer_model_checks: bool,
    defer_s3_buckets: bool,
) -> PyResult<Option<String>> {
    let module = body.py().import(MODULE)?;
    let surface = Surface::of(route);
    let params = root(body, &names.get().0)?;
    let fallback_targets = fallback_targets
        .iter()
        .map(|target| read(target, 0))
        .collect::<PyResult<Vec<_>>>()?;
    let host = PythonHost {
        router: (!router.is_none()).then(|| PythonRouter {
            module: module.clone(),
            router: router.clone(),
            team_id,
        }),
        module,
        body,
    };
    let model = text(model).unwrap_or_default();
    let request = Request {
        body: &params,
        surface,
        model: &model,
        route_model,
        fallback_targets: &fallback_targets,
        deferred: Deferred {
            model_checks: defer_model_checks,
            s3_buckets: defer_s3_buckets,
        },
    };
    match body::check(&request, &names.get().0, &policy(general_settings)?, &host) {
        Ok(()) => Ok(None),
        Err(Failure::Rejected(rejection)) => Ok(Some(rejection.to_string())),
        Err(Failure::Host(error)) => Err(error),
    }
}

fn root(body: &Bound<'_, PyAny>, names: &Names) -> PyResult<Params> {
    entries(body)?
        .into_iter()
        .filter_map(|(key, entry)| {
            let key = text(&key)?;
            let entry = match names.reading(&key) {
                Reading::Unread => Ok(CallerValue::Other),
                Reading::Shallow => Ok(shallow(&entry)),
                Reading::Deep => read(&entry, 0),
            };
            Some(entry.map(|entry| (key, entry)))
        })
        .collect()
}

/// A mapping's entries. A key that is not a string names no parameter, so it is left out.
fn params(mapping: &Bound<'_, PyAny>, depth: usize) -> PyResult<Params> {
    entries(mapping)?
        .into_iter()
        .filter_map(|(key, entry)| {
            let key = text(&key)?;
            Some(read(&entry, depth).map(|value| (key, value)))
        })
        .collect()
}

fn entries<'py>(
    mapping: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    if let Ok(dict) = mapping.cast::<PyDict>() {
        return Ok(dict.iter().collect());
    }
    mapping
        .call_method0("items")?
        .try_iter()?
        .map(|item| item?.extract())
        .collect()
}

fn text(value: &Bound<'_, PyAny>) -> Option<String> {
    value
        .cast::<PyString>()
        .ok()
        .map(|text| text.to_string_lossy().into_owned())
}

fn shallow(value: &Bound<'_, PyAny>) -> CallerValue {
    if value.is_none() {
        return CallerValue::Null;
    }
    text(value).map_or(CallerValue::Other, CallerValue::Text)
}

fn read(object: &Bound<'_, PyAny>, depth: usize) -> PyResult<CallerValue> {
    let scalar = shallow(object);
    if scalar != CallerValue::Other || depth >= MAX_DEPTH {
        return Ok(scalar);
    }
    if object.is_instance_of::<PyDict>() {
        return params(object, depth + 1).map(CallerValue::Map);
    }
    if let Ok(list) = object.cast::<PyList>() {
        return list
            .iter()
            .map(|item| read(&item, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(CallerValue::List);
    }
    Ok(CallerValue::Other)
}

fn setting<'py>(general: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let value = general.call_method1("get", (key,))?;
    Ok((!value.is_none()).then_some(value))
}

/// A setting that lists names. An entry that is not a string equals no value a caller sends.
fn listed(general: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<String>> {
    let Some(value) = setting(general, key)? else {
        return Ok(Vec::new());
    };
    if value.is_instance_of::<PyString>() {
        return Ok(Vec::new());
    }
    value
        .try_iter()?
        .filter_map(|entry| entry.map(|entry| text(&entry)).transpose())
        .collect()
}

fn policy(general: &Bound<'_, PyAny>) -> PyResult<Policy> {
    Ok(Policy {
        allow_client_side_credentials: setting(general, "allow_client_side_credentials")?
            .is_some_and(|value| value.cast::<PyBool>().is_ok_and(|flag| flag.is_true())),
        s3_buckets: listed(general, "allowed_client_side_s3_buckets")?,
        s3_endpoint_urls: listed(general, "allowed_client_side_s3_endpoint_urls")?,
    })
}

struct PythonHost<'a, 'py> {
    module: Bound<'py, PyModule>,
    body: &'a Bound<'py, PyAny>,
    router: Option<PythonRouter<'py>>,
}

impl<'py> Host for PythonHost<'_, 'py> {
    type Error = PyErr;
    type Deployments = PythonRouter<'py>;

    fn decode(&self, text: &str) -> PyResult<Option<Params>> {
        let decoded = self.module.call_method1("decode_mapping", (text,))?;
        if decoded.is_none() {
            return Ok(None);
        }
        params(&decoded, 0).map(Some)
    }

    fn form_metadata(&self, prefix: &str) -> PyResult<Params> {
        let entries = self
            .module
            .call_method1("form_metadata", (self.body, prefix))?;
        params(&entries, 0)
    }

    fn credential(&self, name: &str) -> PyResult<Params> {
        params(&self.module.call_method1("credential_values", (name,))?, 0)
    }

    fn url_destination_allowed(&self, value: &str) -> PyResult<bool> {
        self.module
            .call_method1("url_destination_allowed", (value,))?
            .extract()
    }

    fn deployments(&self) -> Option<&PythonRouter<'py>> {
        self.router.as_ref()
    }
}

/// The proxy's router, kept as the object the caller passed, since only its owner reads it.
struct PythonRouter<'py> {
    module: Bound<'py, PyModule>,
    router: Bound<'py, PyAny>,
    team_id: Option<String>,
}

type OptInAnswer = Option<(Vec<String>, Vec<String>)>;

impl Deployments for PythonRouter<'_> {
    type Error = PyErr;

    fn group(&self, model: &str) -> PyResult<Vec<Deployment>> {
        let answered: Vec<(String, Option<String>, OptInAnswer)> = self
            .module
            .call_method1(
                "group_deployments",
                (&self.router, model, self.team_id.as_deref()),
            )?
            .extract()?;
        Ok(answered
            .into_iter()
            .map(|(model_name, provider, opt_in)| Deployment {
                model_name,
                provider,
                opt_in: opt_in.map(|(params, api_base_patterns)| OptIn {
                    params,
                    api_base_patterns,
                }),
            })
            .collect())
    }

    fn alias(&self, name: &str) -> PyResult<Option<String>> {
        self.module
            .call_method1("alias", (&self.router, name))?
            .extract()
    }

    fn fallbacks(&self, name: &str) -> PyResult<Vec<String>> {
        self.module
            .call_method1("fallbacks", (&self.router, name))?
            .extract()
    }

    fn api_base_matches(&self, pattern: &str, value: &str) -> PyResult<bool> {
        self.module
            .call_method1("api_base_matches", (pattern, value))?
            .extract()
    }
}
