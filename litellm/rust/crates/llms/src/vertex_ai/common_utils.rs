use litellm_llms_types::providers::vertex_ai::{GLOBAL_API_BASE, GLOBAL_LOCATION};
use litellm_python_compat::urllib::{ParsedUrl, urlparse, urlsplit};

use crate::Error;

fn is_location_token(location: &str) -> bool {
    !location.is_empty()
        && location
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && location
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && location
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
}

/// Python's `validate_vertex_location`: a location reaches a hostname, so only `global` or a
/// lowercase alphanumeric token with hyphens is allowed.
pub fn validate_vertex_location(location: &str) -> Result<&str, Error> {
    if location == GLOBAL_LOCATION || is_location_token(location) {
        return Ok(location);
    }
    Err(Error::Auth(litellm_auth::Error::InvalidConfiguration(
        litellm_auth::ErrorDetail::InvalidType {
            field: "vertex_location".into(),
            expected: "global or a lowercase alphanumeric token with hyphens",
        },
    )))
}

/// Python's `get_vertex_base_url`: the global control plane, a multi-region geography, or a
/// regional host.
pub fn get_vertex_base_url(location: &str) -> Result<String, Error> {
    let location = validate_vertex_location(location)?;
    if location == GLOBAL_LOCATION {
        return Ok(GLOBAL_API_BASE.to_string());
    }
    if !location.contains('-') {
        return Ok(format!("https://aiplatform.{location}.rep.googleapis.com"));
    }
    Ok(format!("https://{location}-aiplatform.googleapis.com"))
}

/// Python's `_strip_trailing_slashes_keeping_the_scheme`.
fn strip_trailing_slashes_keeping_the_scheme(api_base: &str) -> String {
    let stripped = api_base.trim_end_matches('/');
    let parts = urlsplit(stripped);
    if !parts.scheme.is_empty()
        && parts.netloc.is_empty()
        && parts.path.is_empty()
        && parts.query.is_empty()
        && parts.fragment.is_empty()
    {
        return api_base.to_string();
    }
    if parts.query.is_empty() || !parts.path.ends_with('/') {
        return stripped.to_string();
    }
    let (before_query, query) = stripped.split_once('?').unwrap_or((stripped, ""));
    format!("{}?{query}", before_query.trim_end_matches('/'))
}

/// Python's `join_api_base`.
pub(crate) fn join_api_base(api_base: &str, route: &str) -> String {
    format!(
        "{}/{}",
        strip_trailing_slashes_keeping_the_scheme(api_base),
        route.trim_start_matches('/')
    )
}

/// Python's `graft_default_vertex_path`: the default URL's path, less its leading API version,
/// appended to the `api_base` path. An `api_base` that is itself a version keeps it, so
/// `https://h/v1` becomes `https://h/v1/v1/projects/...` as it does in Python.
pub(crate) fn graft_default_vertex_path(api_base: &str, default_url: &str) -> String {
    let parsed = urlparse(api_base);
    let default_path = urlparse(default_url).path;
    let segments: Vec<&str> = default_path.trim_start_matches('/').split('/').collect();
    let grafted = match segments.split_first() {
        Some((first, rest)) if matches!(*first, "v1" | "v1beta1") => rest,
        _ => &segments[..],
    };
    ParsedUrl {
        path: format!(
            "{}/{}",
            parsed.path.trim_end_matches('/'),
            grafted.join("/")
        ),
        ..parsed
    }
    .geturl()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::regional("us-east5", "https://us-east5-aiplatform.googleapis.com")]
    #[case::geography("eu", "https://aiplatform.eu.rep.googleapis.com")]
    #[case::global("global", "https://aiplatform.googleapis.com")]
    fn base_url_follows_the_location_kind(#[case] location: &str, #[case] expected: &str) {
        assert_eq!(get_vertex_base_url(location).unwrap(), expected);
    }

    #[rstest]
    #[case::host_injection("attacker.example/")]
    #[case::fragment("evil.com#")]
    #[case::uppercase("US-EAST5")]
    #[case::leading_hyphen("-us")]
    #[case::empty("")]
    fn a_location_that_is_not_a_token_is_rejected(#[case] location: &str) {
        assert!(get_vertex_base_url(location).is_err());
    }
}
