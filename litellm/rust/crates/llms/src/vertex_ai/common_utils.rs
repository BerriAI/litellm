use litellm_auth::VertexParams;
use litellm_auth_gcp::google::{GoogleLoginFailure, PythonException};
use litellm_auth_gcp::vertex::vertex_location;
use litellm_llms_types::providers::vertex_ai::{GLOBAL_API_BASE, GLOBAL_LOCATION};
use litellm_model_catalog::Catalog;
use litellm_python_compat::urllib::urlsplit;

use crate::Error;

/// A lowercase token of letters, digits and hyphens that starts with a letter.
fn is_location_token(location: &str) -> bool {
    location
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_lowercase)
        && location
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// Python's `validate_vertex_location`: a location reaches a hostname, so only `global` or a
/// lowercase token of letters, digits and hyphens is allowed, else its `ValueError`.
pub fn validate_vertex_location(location: &str) -> Result<&str, Error> {
    if location == GLOBAL_LOCATION || is_location_token(location) {
        return Ok(location);
    }
    Err(Error::Auth(
        GoogleLoginFailure::new(PythonException::Value, "Invalid vertex_location format").into(),
    ))
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
    let parsed = urlsplit(api_base);
    let default_path = urlsplit(default_url).path;
    let segments: Vec<&str> = default_path.trim_start_matches('/').split('/').collect();
    let grafted = match segments.split_first() {
        Some((first, rest)) if matches!(*first, "v1" | "v1beta1") => rest,
        _ => &segments[..],
    };
    parsed.with(
        &format!(
            "{}/{}",
            parsed.path.trim_end_matches('/'),
            grafted.join("/")
        ),
        parsed.query,
    )
}

/// Python's `VertexBase.get_vertex_region`: a model the model map lists `supported_regions` for
/// is sent to the first of them unless the configured location is one of them.
pub fn vertex_region(location: Option<String>, supported_regions: &[String]) -> String {
    match (supported_regions.first(), location) {
        (None, location) => location
            .filter(|location| !location.is_empty())
            .unwrap_or_else(|| "us-central1".into()),
        (Some(_), Some(location)) if supported_regions.contains(&location) => location,
        (Some(first), _) => first.clone(),
    }
}

/// The params Python's `VertexBase.explicit_vertex_ai_location` reads an explicitly configured
/// location from, in order. Cost tracking prices the regional uplift from the first.
pub const EXPLICIT_LOCATION_PARAMS: [&str; 2] = ["vertex_location", "vertex_ai_location"];

/// The model map's `supported_regions` for a Claude on Vertex AI model, read under the
/// `vertex_ai/` key Python's `VertexBase.get_vertex_region` looks up.
pub fn supported_regions(catalog: &Catalog, model: &str) -> Vec<String> {
    let key = format!(
        "vertex_ai/{}",
        model.strip_prefix("vertex_ai/").unwrap_or(model)
    );
    catalog
        .get(&key)
        .and_then(|fields| fields.get("supported_regions"))
        .and_then(serde_json::Value::as_array)
        .map(|regions| {
            regions
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// The location Python's `resolve_vertex_location_for_cost` prices a Claude on Vertex AI call
/// in, the one the call is sent to: the configured location, or `VERTEXAI_LOCATION` or
/// `VERTEX_LOCATION`, routed by the model's `supported_regions`.
pub fn cost_location(
    params: &VertexParams,
    model: &str,
    catalog: &Catalog,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    vertex_region(
        vertex_location(params, env_lookup),
        &supported_regions(catalog, model),
    )
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
    #[case::leading_digit("1us")]
    #[case::two_trailing_newlines("us-east5\n\n")]
    #[case::newline_inside("us\neast5")]
    #[case::only_a_newline("\n")]
    #[case::one_trailing_newline("us-east5\n")]
    #[case::empty("")]
    fn a_location_that_is_not_a_token_is_rejected(#[case] location: &str) {
        let Err(Error::Auth(error)) = get_vertex_base_url(location) else {
            panic!("{location:?} should be rejected");
        };
        assert_eq!(
            litellm_error::find::<GoogleLoginFailure>(&error),
            Some(&GoogleLoginFailure::new(
                PythonException::Value,
                "Invalid vertex_location format"
            ))
        );
    }

    #[rstest]
    #[case::trailing_hyphen("us-")]
    #[case::digit_after_letter("u1")]
    fn a_location_python_accepts_is_accepted(#[case] location: &str) {
        assert!(get_vertex_base_url(location).is_ok());
    }

    const CATALOG: &str = r#"{
        "vertex_ai/claude-regional": {"supported_regions": ["us-east5", 7, "europe-west1"]},
        "claude-regional": {"supported_regions": ["asia-east1"]},
        "vertex_ai/claude-listed-oddly": {"supported_regions": "us-east5"}
    }"#;

    #[rstest]
    #[case::bare_model("claude-regional", &["us-east5", "europe-west1"])]
    #[case::prefixed_model("vertex_ai/claude-regional", &["us-east5", "europe-west1"])]
    #[case::not_a_list("claude-listed-oddly", &[])]
    #[case::unlisted("claude-elsewhere", &[])]
    fn supported_regions_are_read_under_the_vertex_ai_key(
        #[case] model: &str,
        #[case] expected: &[&str],
    ) {
        let catalog = Catalog::parse(CATALOG.as_bytes()).unwrap().value;
        assert_eq!(supported_regions(&catalog, model), expected);
    }

    #[rstest]
    #[case::no_regions(Some("europe-west4"), &[], "europe-west4")]
    #[case::default_location(None, &[], "us-central1")]
    #[case::empty_location(Some(""), &[], "us-central1")]
    #[case::first_region_without_location(None, &["us-east5", "europe-west1"], "us-east5")]
    #[case::supported_location_kept(Some("europe-west1"), &["us-east5", "europe-west1"], "europe-west1")]
    #[case::unsupported_location_replaced(Some("asia-east1"), &["us-east5"], "us-east5")]
    fn the_region_follows_the_model_s_supported_regions(
        #[case] location: Option<&str>,
        #[case] supported: &[&str],
        #[case] expected: &str,
    ) {
        let supported: Vec<String> = supported.iter().map(|region| region.to_string()).collect();
        assert_eq!(
            vertex_region(location.map(str::to_string), &supported),
            expected
        );
    }
}
