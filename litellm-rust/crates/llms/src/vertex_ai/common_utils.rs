use crate::Error;

const GLOBAL_LOCATION: &str = "global";
pub const DEFAULT_VERTEX_LOCATION: &str = "us-central1";

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
        return Ok("https://aiplatform.googleapis.com".to_string());
    }
    if !location.contains('-') {
        return Ok(format!("https://aiplatform.{location}.rep.googleapis.com"));
    }
    Ok(format!("https://{location}-aiplatform.googleapis.com"))
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
