use std::sync::OnceLock;

use crate::error::{AuthError, Error};
use crate::providers::auth::CredentialPlacement;
use crate::providers::auth::azure::{AzureAuthInputs, AzureAuthService};
use crate::providers::auth::http::apply_credential;

pub const AZURE_AI_API_KEY_ENV: &str = "AZURE_AI_API_KEY";
pub const AZURE_AI_API_BASE_ENV: &str = "AZURE_AI_API_BASE";
pub const AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
pub const AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

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
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    if headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("Authorization"))
    {
        return Ok(headers);
    }
    let api_key = resolve_api_key(api_key, env_lookup)?;
    apply_credential(headers, &api_key, CredentialPlacement::Bearer).map_err(Error::from)
}

pub async fn authenticate(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, Error> {
    if crate::http_utils::has_header(&headers, "Authorization") {
        return Ok(headers);
    }
    if let Some(api_key) = resolve_optional_api_key(api_key, env_lookup) {
        return apply_credential(headers, &api_key, CredentialPlacement::Bearer)
            .map_err(Error::from);
    }
    let credential = resolve_entra_credential(auth_inputs, env_lookup)
        .await?
        .ok_or_else(|| {
            AuthError::InvalidConfiguration(
                "Missing Azure AI credentials - set AZURE_AI_API_KEY or configure Entra ID"
                    .to_string(),
            )
        })?;
    apply_credential(
        headers,
        credential.secret().expose(),
        CredentialPlacement::Bearer,
    )
    .map_err(Error::from)
}

pub async fn authenticate_document_intelligence(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, Error> {
    if crate::http_utils::has_header(&headers, "Authorization")
        || crate::http_utils::has_header(&headers, "Ocp-Apim-Subscription-Key")
    {
        return Ok(headers);
    }
    if let Some(api_key) =
        resolve_optional_value(api_key, AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV, env_lookup)
    {
        return apply_credential(
            headers,
            &api_key,
            CredentialPlacement::Header("Ocp-Apim-Subscription-Key"),
        )
        .map_err(Error::from);
    }
    let credential = resolve_entra_credential(auth_inputs, env_lookup)
        .await?
        .ok_or_else(|| {
            AuthError::InvalidConfiguration(
                "Missing Azure Document Intelligence credentials - set AZURE_DOCUMENT_INTELLIGENCE_API_KEY or configure Entra ID"
                    .to_string(),
            )
        })?;
    apply_credential(
        headers,
        credential.secret().expose(),
        CredentialPlacement::Bearer,
    )
    .map_err(Error::from)
}

async fn resolve_entra_credential(
    auth_inputs: Option<&AzureAuthInputs>,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Option<crate::providers::auth::ResolvedCredential>, AuthError> {
    match auth_inputs {
        Some(inputs) => auth_service().resolve(inputs, env_lookup).await,
        None => Ok(None),
    }
}

fn resolve_optional_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_optional_value(api_key, AZURE_AI_API_KEY_ENV, env_lookup)
}

fn resolve_optional_value(
    explicit: Option<&str>,
    env_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    explicit
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(env_name).filter(|value| !value.is_empty()))
}

fn auth_service() -> &'static AzureAuthService {
    static SERVICE: OnceLock<AzureAuthService> = OnceLock::new();
    SERVICE.get_or_init(AzureAuthService::default)
}

#[cfg(test)]
mod tests {
    use crate::error::{AuthError, Error};
    use crate::providers::auth::azure::AzureAuthInputs;
    use serde_json::json;

    use super::{
        AZURE_AI_API_KEY_ENV, authenticate, authenticate_document_intelligence, resolve_api_base,
        resolve_api_key, validate_environment,
    };

    fn azure_inputs(params: serde_json::Value) -> AzureAuthInputs {
        AzureAuthInputs::from_optional_params(params.as_object().expect("object"))
            .expect("valid Azure auth inputs")
    }

    fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
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
        let headers = validate_environment(Vec::new(), Some("key"), &|_| None)
            .expect("api key authenticates");

        assert_eq!(
            headers,
            vec![("Authorization".to_string(), "Bearer key".to_string())]
        );
    }

    #[test]
    fn caller_bearer_token_wins_over_environment_key() {
        let headers = validate_environment(
            vec![(
                "authorization".to_string(),
                "Bearer caller-token".to_string(),
            )],
            None,
            &|name| (name == AZURE_AI_API_KEY_ENV).then(|| "environment-key".to_string()),
        )
        .expect("caller header authenticates");

        assert_eq!(
            headers,
            vec![(
                "authorization".to_string(),
                "Bearer caller-token".to_string()
            )]
        );
    }

    #[tokio::test]
    async fn azure_ai_uses_supplied_entra_token() {
        let inputs = azure_inputs(json!({"azure_ad_token": "entra-token"}));
        let headers = authenticate(Vec::new(), None, Some(&inputs), &|_| None)
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
            authenticate_document_intelligence(Vec::new(), Some("my-key"), None, &|_| None)
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
            authenticate_document_intelligence(Vec::new(), None, Some(&inputs), &|_| None)
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
        let headers =
            authenticate_document_intelligence(Vec::new(), Some("api-key"), Some(&inputs), &|_| {
                None
            })
            .await
            .expect("API key authenticates");

        assert_eq!(
            header_value(&headers, "Ocp-Apim-Subscription-Key"),
            Some("api-key")
        );
        assert_eq!(header_value(&headers, "Authorization"), None);
    }
}
