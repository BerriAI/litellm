//! Where an `oidc/` api_key a stored credential holds may be sent, the way
//! `CredentialApiKeySecretReference` in `litellm/types/utils.py` binds it: to the origins of the
//! credential's own upstream params, and nowhere else.

use std::collections::BTreeSet;

use serde_yaml_ng::Mapping;

use crate::{Object, Value};

/// `OIDC_SECRET_REFERENCE_PREFIX`.
const OIDC_PREFIX: &str = "oidc/";
/// `API_KEY_UPSTREAM_PARAMS`: every param that points a call at a host.
const UPSTREAM_PARAMS: [&str; 6] = [
    "api_base",
    "base_url",
    "aws_bedrock_runtime_endpoint",
    "s3_endpoint_url",
    "sagemaker_base_url",
    "deployment_url",
];
const MODEL: &str = "model";
const API_KEY: &str = "api_key";

/// Whether filling `credential`'s api_key into `params` would send an `oidc/` secret to a host
/// the credential does not name, which Python refuses at call time (`may_reach`). The hosts a
/// call goes to are the model's `model` and, per upstream param, the model's value when it sets
/// the param and the credential's otherwise. A call that names no host may carry the secret only
/// when the credential names none either.
pub(super) fn withholds_api_key(params: &Mapping, credential: &Object) -> bool {
    let Some(Value::String(api_key)) = credential.get(API_KEY) else {
        return false;
    };
    if !api_key.starts_with(OIDC_PREFIX) {
        return false;
    }
    let allowed = origins(
        UPSTREAM_PARAMS
            .iter()
            .filter_map(|param| text(credential.get(param)))
            .chain(text(credential.get(MODEL)).and_then(model_url)),
    );
    let called = origins(
        UPSTREAM_PARAMS
            .iter()
            .filter_map(|param| match params.get(*param) {
                Some(own) => text(Some(own)),
                None => text(credential.get(param)),
            })
            .chain(text(params.get(MODEL)).and_then(model_url)),
    );
    match called.is_empty() {
        true => !allowed.is_empty(),
        false => !called.is_subset(&allowed),
    }
}

/// A param's text, `None` when it is unset, empty or not a string.
fn text(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn origins<'a>(urls: impl Iterator<Item = &'a str>) -> BTreeSet<String> {
    urls.map(origin).collect()
}

/// `_model_url`: the first `http://` or `https://` URL inside `model`, up to whitespace.
fn model_url(model: &str) -> Option<&str> {
    let lower = model.to_ascii_lowercase();
    let start = ["http://", "https://"]
        .iter()
        .filter_map(|scheme| lower.find(scheme))
        .min()?;
    model[start..].split_whitespace().next()
}

/// `_origin`: `scheme://host:port` with the scheme's default port filled in, or the text
/// lower-cased without trailing slashes when it is not a URL with a host.
fn origin(text: &str) -> String {
    let text = text.trim();
    let fallback = || text.trim_end_matches('/').to_lowercase();
    let Ok(url) = url::Url::parse(text) else {
        return fallback();
    };
    let host = url.host_str().unwrap_or_default().trim_end_matches('.');
    if host.is_empty() {
        return fallback();
    }
    let scheme = url.scheme();
    let port = url.port().or(match scheme {
        "http" | "ws" => Some(80),
        "https" | "wss" => Some(443),
        _ => None,
    });
    match port {
        Some(port) => format!("{scheme}://{host}:{port}"),
        None => format!("{scheme}://{host}"),
    }
}
