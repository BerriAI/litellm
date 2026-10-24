use std::{collections::BTreeMap, sync::Arc};

use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_host_python::from_py;
use litellm_inference_tokenize::{TokenizeBackend, TokenizeRequest};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use litellm_token_counter::raw_text::LocalTokenizerConfig;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use serde_json::{Map, Value};

use crate::{
    marshal::{optional_timeout, python_timeout_seconds},
    routes::RequestView,
};

pub(super) const MODULE: &str = "litellm.rust_bridge.host.routes.tokenize";

pub(super) fn project_request(
    py: Python<'_>,
    view: &mut RequestView,
    arguments: &Bound<'_, PyDict>,
) -> PyResult<TokenizeRequest> {
    view.resolve(py, arguments)?;
    let argument = |name: &str| view.argument(py, arguments, name);
    let request = view.request(py)?;
    let string = |name: &str| -> PyResult<Option<String>> {
        argument(name)?.map(|value| value.extract()).transpose()
    };
    let required = |name: &str| -> PyResult<String> {
        string(name)?.ok_or_else(|| PyValueError::new_err(format!("{name} is required")))
    };
    let timeout = argument("timeout")?
        .or(argument("request_timeout")?)
        .map(|value| python_timeout_seconds(py, value.unbind()))
        .transpose()?
        .flatten();
    let optional_params = view.body_params(py, arguments, MODULE)?;
    let deployment_params: Map<String, Value> = from_py(
        &py.import(MODULE)?
            .getattr("deployment_params")?
            .call1((request,))?,
    )?;
    let extra_headers: Map<String, Value> = argument("extra_headers")?
        .map(|value| from_py::<Map<String, Value>>(&value))
        .transpose()?
        .unwrap_or_default();
    let include: TokenizeInclude = argument("include")?
        .map(|value| from_py::<TokenizeInclude>(&value))
        .transpose()?
        .unwrap_or_default();
    let add_special_tokens = argument("add_special_tokens")?
        .map(|value| from_py::<bool>(&value))
        .transpose()?;
    let api_key = string("api_key")?
        .map(|value| {
            source_for(py, request, "api_key")
                .map(|source| Sourced::new(SecretValue::new(value), source))
        })
        .transpose()?;
    let api_base = string("api_base")?
        .or(string("base_url")?)
        .map(|value| source_for(py, request, "api_base").map(|source| Sourced::new(value, source)))
        .transpose()?;
    Ok(TokenizeRequest {
        model: required("model")?,
        prompt: required("prompt")?,
        include,
        add_special_tokens,
        backend: project_backend(py, request, argument("backend")?)?,
        custom_llm_provider: string("custom_llm_provider")?,
        api_key,
        api_base,
        extra_headers: source_map_with_source(
            extra_headers,
            source_for(py, request, "extra_headers")?,
        ),
        optional_params: source_map(py, request, optional_params)?,
        deployment_params: deployment_params
            .into_iter()
            .map(|(name, value)| (name, Sourced::new(value, InputSource::Deployment)))
            .collect(),
        timeout: optional_timeout(timeout),
    })
}

fn source_for(py: Python<'_>, request: &Bound<'_, PyDict>, name: &str) -> PyResult<InputSource> {
    match py
        .import(MODULE)?
        .getattr("source_for")?
        .call1((request, name))?
        .extract::<String>()?
        .as_str()
    {
        "request" => Ok(InputSource::Request),
        "deployment" => Ok(InputSource::Deployment),
        _ => Err(PyValueError::new_err("invalid tokenization field source")),
    }
}

fn source_map(
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
    values: Map<String, Value>,
) -> PyResult<BTreeMap<String, Sourced<Value>>> {
    values
        .into_iter()
        .map(|(name, value)| {
            source_for(py, request, &name).map(|source| (name, Sourced::new(value, source)))
        })
        .collect()
}

fn source_map_with_source(
    values: Map<String, Value>,
    source: InputSource,
) -> BTreeMap<String, Sourced<Value>> {
    values
        .into_iter()
        .map(|(name, value)| (name, Sourced::new(value, source)))
        .collect()
}

fn project_backend(
    py: Python<'_>,
    request: &Bound<'_, PyDict>,
    value: Option<Bound<'_, PyAny>>,
) -> PyResult<TokenizeBackend> {
    let Some(value) = value else {
        return Ok(TokenizeBackend::Provider);
    };
    let backend: Value = from_py(
        &py.import(MODULE)?
            .getattr("normalize_backend")?
            .call1((value,))?,
    )?;
    let fields = backend
        .as_object()
        .ok_or_else(|| PyValueError::new_err("backend must be a tagged object"))?;
    match fields.get("type").and_then(Value::as_str) {
        Some("provider") if fields.len() == 1 => Ok(TokenizeBackend::Provider),
        Some("local") if fields.len() == 2 => {
            let config: LocalTokenizerConfig = serde_json::from_value(
                fields
                    .get("config")
                    .cloned()
                    .ok_or_else(|| PyValueError::new_err("local backend config is required"))?,
            )
            .map_err(|_| PyValueError::new_err("invalid local tokenizer configuration"))?;
            let source = source_for(py, request, "backend")?;
            Ok(TokenizeBackend::Local {
                config: Sourced::new(Arc::new(config), source),
            })
        }
        _ => Err(PyValueError::new_err("invalid tokenization backend")),
    }
}
