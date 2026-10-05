use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{header_value, without_headers};

use crate::{
    Error,
    base_llm::auth::{AuthScheme, Headers, ValidatedEnvironment},
};

pub(super) const SECRET_NAMES: &[&str] = &["BASETEN_API_KEY", "BASETEN_API_BASE"];
const DEFAULT_API_BASE: &str = "https://inference.baseten.co/v1";

pub(super) fn complete_url(
    api_base: Option<&str>,
    endpoint: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let resolved = resolve_non_empty(api_base, env_lookup, &["BASETEN_API_BASE"])
        .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
    let base = resolved.trim_end_matches('/');
    let suffix = format!("/v1/{endpoint}");
    if base.ends_with(&suffix) {
        return base.to_string();
    }
    if base.ends_with("/v1") {
        return format!("{base}/{endpoint}");
    }
    format!("{base}{suffix}")
}

pub(super) fn validate_environment(
    headers: Headers,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    let forwarded = without_headers(headers, &["x-api-key"]);
    if header_value(&forwarded, "authorization").is_some_and(|value| !value.trim().is_empty()) {
        return Ok(ValidatedEnvironment {
            headers: forwarded,
            auth: AuthScheme::Forwarded,
        });
    }
    let key = resolve_non_empty(api_key, env_lookup, &["BASETEN_API_KEY"]).ok_or(
        litellm_auth::Error::MissingApiKey {
            provider: "Baseten",
            environment_variable: "BASETEN_API_KEY",
        },
    )?;
    Ok(ValidatedEnvironment {
        headers: forwarded,
        auth: AuthScheme::Credential {
            placement: CredentialPlacement::Bearer,
            secret: SecretValue::new(key),
        },
    })
}
