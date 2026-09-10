use crate::auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, ResolvedCredential, SecretValue,
};
use crate::error::AuthError;

pub const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

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
) -> Result<String, AuthError> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or(AuthError::MissingApiKey {
            provider: "Mistral",
        })
}

pub fn validate_environment(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, AuthError> {
    if AUTH_POLICY.has_existing_credential(&headers) {
        return Ok(headers);
    }
    let credential =
        ResolvedCredential::Static(SecretValue::new(resolve_api_key(api_key, env_lookup)?));
    AUTH_POLICY.apply(headers, CredentialPlanKind::Static, &credential)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_key_precedes_environment_key() {
        let env_lookup = |name: &str| (name == MISTRAL_API_KEY_ENV).then(|| "sk-env".to_string());

        assert_eq!(
            resolve_api_key(Some("sk-param"), &env_lookup).unwrap(),
            "sk-param"
        );
    }

    #[test]
    fn blank_explicit_key_uses_environment_key() {
        let env_lookup = |name: &str| (name == MISTRAL_API_KEY_ENV).then(|| "sk-env".to_string());

        assert_eq!(resolve_api_key(Some("  "), &env_lookup).unwrap(), "sk-env");
    }

    #[test]
    fn missing_key_is_typed() {
        let error = resolve_api_key(None, &|_| None).expect_err("missing key should error");

        assert_eq!(
            error,
            AuthError::MissingApiKey {
                provider: "Mistral",
            }
        );
    }

    #[test]
    fn static_key_is_applied_as_bearer() {
        let headers = validate_environment(Vec::new(), Some("sk-param"), &|_| None).unwrap();

        assert_eq!(
            headers,
            vec![("Authorization".to_string(), "Bearer sk-param".to_string())]
        );
    }

    #[test]
    fn existing_authorization_is_preserved_without_key_lookup() {
        let headers = vec![("authorization".to_string(), "Bearer existing".to_string())];

        assert_eq!(
            validate_environment(headers.clone(), None, &|_| None).unwrap(),
            headers
        );
    }
}
