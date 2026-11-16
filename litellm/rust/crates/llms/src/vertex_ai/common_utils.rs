use litellm_auth_gcp::vertex_messages::{GoogleLoginFailure, PythonException};
use litellm_llms_types::providers::vertex_ai::{GLOBAL_API_BASE, GLOBAL_LOCATION};

use crate::Error;

/// Python's `re.match(r"^[a-z][a-z0-9-]*$", location)`, without the newline `$` also matches
/// before: a location reaches a hostname.
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
        GoogleLoginFailure {
            exception: PythonException::Value,
            message: "Invalid vertex_location format".into(),
        }
        .into(),
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
    #[case::trailing_newline("us-east5\n")]
    #[case::empty("")]
    fn a_location_that_is_not_a_token_is_rejected(#[case] location: &str) {
        let Err(Error::Auth(error)) = get_vertex_base_url(location) else {
            panic!("{location:?} should be rejected");
        };
        assert_eq!(
            litellm_error::find::<GoogleLoginFailure>(&error),
            Some(&GoogleLoginFailure {
                exception: PythonException::Value,
                message: "Invalid vertex_location format".into(),
            })
        );
    }

    #[rstest]
    #[case::trailing_hyphen("us-")]
    #[case::digit_after_letter("u1")]
    fn a_location_python_accepts_is_accepted(#[case] location: &str) {
        assert!(get_vertex_base_url(location).is_ok());
    }
}
