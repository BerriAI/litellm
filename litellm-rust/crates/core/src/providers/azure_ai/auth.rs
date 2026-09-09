use std::sync::OnceLock;

use reqwest::header::HeaderMap;

use crate::auth::azure::{AzureAuthInputs, AzureAuthService};
use crate::auth::http::apply_credential;
use crate::auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRef, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, ResolvedCredential, SecretValue,
};
use crate::error::{AuthError, Error};

pub const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
pub const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";
pub const AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
pub const AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

const AZURE_AI_AUTH_RULES: &[CredentialRule] = &[
    CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Bearer,
    },
    CredentialRule {
        kind: CredentialPlanKind::Entra,
        placement: CredentialPlacement::Bearer,
    },
];
const AZURE_AI_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: AZURE_AI_AUTH_RULES,
    accepted_existing_headers: &["Authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: Some(crate::auth::azure::DEFAULT_AZURE_SCOPE),
    audience: None,
};
const DOCUMENT_INTELLIGENCE_AUTH_RULES: &[CredentialRule] = &[
    CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("Ocp-Apim-Subscription-Key"),
    },
    CredentialRule {
        kind: CredentialPlanKind::Entra,
        placement: CredentialPlacement::Bearer,
    },
];
const DOCUMENT_INTELLIGENCE_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: DOCUMENT_INTELLIGENCE_AUTH_RULES,
    accepted_existing_headers: &["Authorization", "Ocp-Apim-Subscription-Key"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: Some(crate::auth::azure::DEFAULT_AZURE_SCOPE),
    audience: None,
};

fn resolve_value(
    explicit: Option<&str>,
    env_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    missing_error: AuthError,
) -> Result<String, Error> {
    explicit
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(env_name).filter(|value| !value.trim().is_empty()))
        .ok_or(Error::Auth(missing_error))
}

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_value(
        api_key,
        AZURE_AI_API_KEY_ENV,
        env_lookup,
        AuthError::MissingApiKey {
            provider: "Azure AI",
        },
    )
}

pub fn resolve_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_value(
        api_base,
        AZURE_AI_API_BASE_ENV,
        env_lookup,
        AuthError::MissingApiBase {
            provider: "Azure AI",
            environment_variable: AZURE_AI_API_BASE_ENV,
        },
    )
}

pub fn resolve_document_intelligence_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_value(
        api_key,
        AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV,
        env_lookup,
        AuthError::MissingApiKey {
            provider: "Azure Document Intelligence",
        },
    )
}

pub fn resolve_document_intelligence_endpoint(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_value(
        api_base,
        AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
        env_lookup,
        AuthError::MissingApiBase {
            provider: "Azure Document Intelligence",
            environment_variable: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
        },
    )
}

pub fn validate_environment(
    headers: HeaderMap,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<HeaderMap, Error> {
    if headers.contains_key("Authorization") {
        return Ok(headers);
    }
    let api_key = resolve_api_key(api_key, env_lookup)?;
    apply_credential(headers, &api_key, CredentialPlacement::Bearer).map_err(Error::from)
}

pub async fn authenticate(
    headers: HeaderMap,
    api_key: Option<&str>,
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<HeaderMap, Error> {
    authenticate_with_policy(
        headers,
        api_key,
        AZURE_AI_API_KEY_ENV,
        auth_inputs,
        env_lookup,
        &AZURE_AI_AUTH_POLICY,
        AuthError::MissingAzureAiCredentials,
    )
    .await
}

pub async fn authenticate_document_intelligence(
    headers: HeaderMap,
    api_key: Option<&str>,
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<HeaderMap, Error> {
    authenticate_with_policy(
        headers,
        api_key,
        AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV,
        auth_inputs,
        env_lookup,
        &DOCUMENT_INTELLIGENCE_AUTH_POLICY,
        AuthError::MissingAzureDocumentIntelligenceCredentials,
    )
    .await
}

async fn authenticate_with_policy(
    headers: HeaderMap,
    api_key: Option<&str>,
    api_key_env: &str,
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    policy: &ProviderAuthPolicy,
    missing_error: AuthError,
) -> Result<HeaderMap, Error> {
    if policy.has_existing_credential(&headers) {
        return Ok(headers);
    }
    for rule in policy.rules {
        let credential = match rule.kind {
            CredentialPlanKind::Static => {
                resolve_static_credential(api_key, api_key_env, env_lookup)
            }
            CredentialPlanKind::Entra => resolve_entra_credential(auth_inputs, env_lookup).await?,
            CredentialPlanKind::Caller => {
                return Err(AuthError::CallerPlanRequiresProviderInputs.into());
            }
        };
        if let Some(credential) = credential {
            return policy
                .apply(headers, rule.kind, &credential)
                .map_err(Error::from);
        }
    }
    Err(missing_error.into())
}

async fn resolve_entra_credential(
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Option<crate::auth::ResolvedCredential>, AuthError> {
    match auth_inputs {
        Some(inputs) => auth_service().resolve(inputs, env_lookup).await,
        None => Ok(None),
    }
}

fn resolve_static_credential(
    explicit: Option<&str>,
    env_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<ResolvedCredential> {
    let reference = explicit
        .filter(|value| !value.is_empty())
        .map(|value| CredentialRef::Explicit(SecretValue::new(value)))
        .unwrap_or_else(|| CredentialRef::Env(env_name.to_string()));
    match reference {
        CredentialRef::Explicit(secret) => Some(ResolvedCredential::Static(secret)),
        CredentialRef::Env(name) => env_lookup(&name)
            .filter(|value| !value.is_empty())
            .map(SecretValue::new)
            .map(ResolvedCredential::Static),
        CredentialRef::File(_)
        | CredentialRef::Request(_)
        | CredentialRef::Host(_)
        | CredentialRef::None => None,
    }
}

fn auth_service() -> &'static AzureAuthService {
    static SERVICE: OnceLock<AzureAuthService> = OnceLock::new();
    SERVICE.get_or_init(AzureAuthService::default)
}

#[cfg(test)]
mod tests {
    use crate::auth::azure::AzureAuthInputs;
    use crate::error::{AuthError, Error};
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    use serde_json::json;

    use super::{
        AZURE_AI_API_KEY_ENV, authenticate, authenticate_document_intelligence, resolve_api_base,
        resolve_api_key, validate_environment,
    };

    fn azure_inputs(params: serde_json::Value) -> AzureAuthInputs {
        AzureAuthInputs::from_optional_params(params.as_object().expect("object"))
            .expect("valid Azure auth inputs")
    }

    fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
        headers.get(name).and_then(|value| value.to_str().ok())
    }

    #[test]
    fn missing_api_key_has_a_typed_error() {
        assert_eq!(
            resolve_api_key(None, &|_| None).expect_err("missing key errors"),
            Error::Auth(AuthError::MissingApiKey {
                provider: "Azure AI",
            })
        );
    }

    #[test]
    fn missing_api_base_has_a_typed_error() {
        assert_eq!(
            resolve_api_base(None, &|_| None).expect_err("missing base errors"),
            Error::Auth(AuthError::MissingApiBase {
                provider: "Azure AI",
                environment_variable: "AZURE_AI_API_BASE",
            })
        );
    }

    #[test]
    fn api_key_is_sent_as_bearer() {
        let headers = validate_environment(HeaderMap::new(), Some("key"), &|_| None)
            .expect("api key authenticates");

        assert_eq!(
            headers,
            HeaderMap::from_iter([(
                HeaderName::from_static("authorization"),
                HeaderValue::from_static("Bearer key")
            )])
        );
    }

    #[test]
    fn caller_bearer_token_wins_over_environment_key() {
        let headers = validate_environment(
            HeaderMap::from_iter([(
                HeaderName::from_static("authorization"),
                HeaderValue::from_static("Bearer caller-token"),
            )]),
            None,
            &|name| (name == AZURE_AI_API_KEY_ENV).then(|| "environment-key".to_string()),
        )
        .expect("caller header authenticates");

        assert_eq!(
            headers,
            HeaderMap::from_iter([(
                HeaderName::from_static("authorization"),
                HeaderValue::from_static("Bearer caller-token")
            )])
        );
    }

    #[tokio::test]
    async fn azure_ai_uses_supplied_entra_token() {
        let inputs = azure_inputs(json!({"azure_ad_token": "entra-token"}));
        let headers = authenticate(HeaderMap::new(), None, Some(&inputs), &|_| None)
            .await
            .expect("Entra token authenticates");

        assert_eq!(
            header_value(&headers, "Authorization"),
            Some("Bearer entra-token")
        );
    }

    #[tokio::test]
    async fn document_intelligence_api_key_uses_subscription_header() {
        let headers =
            authenticate_document_intelligence(HeaderMap::new(), Some("my-key"), None, &|_| None)
                .await
                .expect("API key authenticates");

        assert_eq!(
            header_value(&headers, "Ocp-Apim-Subscription-Key"),
            Some("my-key")
        );
        assert_eq!(header_value(&headers, "Authorization"), None);
    }

    #[tokio::test]
    async fn document_intelligence_falls_back_to_entra_token() {
        let inputs = azure_inputs(json!({"azure_ad_token": "entra-token"}));
        let headers =
            authenticate_document_intelligence(HeaderMap::new(), None, Some(&inputs), &|_| None)
                .await
                .expect("Entra token authenticates");

        assert_eq!(
            header_value(&headers, "Authorization"),
            Some("Bearer entra-token")
        );
        assert_eq!(header_value(&headers, "Ocp-Apim-Subscription-Key"), None);
    }

    #[tokio::test]
    async fn document_intelligence_api_key_precedes_entra_token() {
        let inputs = azure_inputs(json!({"azure_ad_token": "entra-token"}));
        let headers = authenticate_document_intelligence(
            HeaderMap::new(),
            Some("api-key"),
            Some(&inputs),
            &|_| None,
        )
        .await
        .expect("API key authenticates");

        assert_eq!(
            header_value(&headers, "Ocp-Apim-Subscription-Key"),
            Some("api-key")
        );
        assert_eq!(header_value(&headers, "Authorization"), None);
    }
}
