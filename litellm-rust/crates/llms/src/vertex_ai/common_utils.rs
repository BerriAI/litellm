use litellm_llms_types::providers::vertex_ai::{GLOBAL_API_BASE, GLOBAL_LOCATION};

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

/// Python's `_check_custom_proxy` for a Vertex URL: a custom base with no path of its own, or
/// only an API version, takes the default URL's path; any other base is called as given with
/// the method appended.
pub fn graft_custom_vertex_base(api_base: Option<&str>, default_url: &str, method: &str) -> String {
    let Some(custom) = api_base.filter(|base| !base.is_empty()) else {
        return default_url.to_owned();
    };
    let path_of = |url: &str| -> String {
        let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
        let rest = rest.split(['?', '#']).next().unwrap_or(rest);
        rest.find('/')
            .map_or(String::new(), |at| rest[at..].to_owned())
    };
    let default_path = path_of(default_url);
    let base_path = path_of(custom);
    match base_path.trim_end_matches('/') {
        "" => format!("{}{default_path}", custom.trim_end_matches('/')),
        version @ ("/v1" | "/v1beta1") if default_path.contains("/projects/") => {
            let tail = default_path
                .split_once("/projects/")
                .map_or("", |(_, tail)| tail);
            format!(
                "{}{version}/projects/{tail}",
                custom.trim_end_matches('/').trim_end_matches(version)
            )
        }
        _ => format!("{custom}:{method}"),
    }
}

/// `url` with `pair` added to its query, before any fragment.
pub fn with_query(url: &str, pair: &str) -> String {
    let (head, fragment) = url
        .split_once('#')
        .map_or((url, None), |(head, fragment)| (head, Some(fragment)));
    let joined = match head.split_once('?') {
        Some((_, query)) if !query.is_empty() => format!("{head}&{pair}"),
        Some((path, _)) => format!("{path}?{pair}"),
        None => format!("{head}?{pair}"),
    };
    match fragment {
        Some(fragment) => format!("{joined}#{fragment}"),
        None => joined,
    }
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

    const DEFAULT: &str = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent";

    #[rstest]
    #[case::none(None, DEFAULT)]
    #[case::bare_host(
        Some("https://proxy.example/"),
        "https://proxy.example/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
    )]
    #[case::version_path(
        Some("https://proxy.example/v1beta1"),
        "https://proxy.example/v1beta1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
    )]
    #[case::own_path(
        Some("https://gw.example/route/x"),
        "https://gw.example/route/x:generateContent"
    )]
    fn a_custom_base_grafts_or_appends_the_method(
        #[case] api_base: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            graft_custom_vertex_base(api_base, DEFAULT, "generateContent"),
            expected
        );
    }

    #[rstest]
    #[case::no_query("https://h/x:m", "https://h/x:m?alt=sse")]
    #[case::query("https://h/x?k=1", "https://h/x?k=1&alt=sse")]
    #[case::empty_query("https://h/x?", "https://h/x?alt=sse")]
    fn a_query_pair_joins_the_query(#[case] url: &str, #[case] expected: &str) {
        assert_eq!(with_query(url, "alt=sse"), expected);
    }
}
