use reqwest::header::HeaderMap;

use crate::auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, ResolvedCredential, SecretValue,
};
use crate::error::{AuthError, Error};

pub const REDUCTO_API_KEY_ENV: &str = "REDUCTO_API_KEY";

const AUTH_RULES: &[CredentialRule] = &[CredentialRule {
    kind: CredentialPlanKind::Static,
    placement: CredentialPlacement::Bearer,
}];
const AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: AUTH_RULES,
    accepted_existing_headers: &["Authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            env_lookup(REDUCTO_API_KEY_ENV)
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .ok_or(Error::Auth(AuthError::MissingReductoApiKey))
}

pub fn validate_environment(
    headers: HeaderMap,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<HeaderMap, Error> {
    if AUTH_POLICY.has_existing_credential(&headers) {
        return Ok(headers);
    }
    let credential =
        ResolvedCredential::Static(SecretValue::new(resolve_api_key(api_key, env_lookup)?));
    AUTH_POLICY
        .apply(headers, CredentialPlanKind::Static, &credential)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    #[test]
    fn explicit_key_precedes_environment_key() {
        let key = resolve_api_key(Some("passed-key"), &|_| Some("env-key".to_string()))
            .expect("explicit key resolves");

        assert_eq!(key, "passed-key");
    }

    #[test]
    fn blank_explicit_key_uses_environment_key() {
        let env_lookup =
            |name: &str| (name == REDUCTO_API_KEY_ENV).then(|| " env-key ".to_string());

        assert_eq!(resolve_api_key(Some(" "), &env_lookup).unwrap(), "env-key");
    }

    #[test]
    fn static_key_is_applied_as_bearer() {
        let headers =
            validate_environment(HeaderMap::new(), Some("passed-key"), &|_| None).unwrap();

        assert_eq!(
            headers,
            HeaderMap::from_iter([(
                HeaderName::from_static("authorization"),
                HeaderValue::from_static("Bearer passed-key")
            )])
        );
    }

    #[test]
    fn existing_authorization_is_preserved_without_key_lookup() {
        let headers = HeaderMap::from_iter([(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer existing"),
        )]);

        assert_eq!(
            validate_environment(headers.clone(), None, &|_| None).unwrap(),
            headers
        );
    }
}
