//! Endpoint, API version and credential resolution for Azure OpenAI, as
//! `litellm/llms/azure/common_utils.py`, `litellm/main.py` and the `AzureOpenAI` client of the
//! OpenAI SDK resolve them.

use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;

use crate::{
    Error,
    base_llm::auth::{AuthScheme, Headers, ValidatedEnvironment},
};

/// `litellm.AZURE_DEFAULT_API_VERSION`.
pub const AZURE_DEFAULT_API_VERSION: &str = "2025-02-01-preview";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";
/// The SDK's own fallback when LiteLLM passes no endpoint.
pub const AZURE_OPENAI_ENDPOINT_ENV: &str = "AZURE_OPENAI_ENDPOINT";
pub const AZURE_API_VERSION_ENV: &str = "AZURE_API_VERSION";
pub const AZURE_OPENAI_API_KEY_ENV: &str = "AZURE_OPENAI_API_KEY";
pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
const API_KEY_HEADER: &str = "api-key";

/// The environment variables a call may fall back to, besides the credential ones
/// `litellm_auth_azure::SECRET_NAMES` lists.
pub const SECRET_NAMES: &[&str] = &[
    AZURE_API_BASE_ENV,
    AZURE_OPENAI_ENDPOINT_ENV,
    AZURE_API_VERSION_ENV,
    AZURE_OPENAI_API_KEY_ENV,
    AZURE_API_KEY_ENV,
];

/// `BaseAzureLLM._is_azure_v1_api_version`: these versions use the `/openai/v1/` surface
/// through the plain OpenAI client.
pub fn is_v1_api_version(api_version: &str) -> bool {
    matches!(api_version, "preview" | "latest" | "v1")
}

/// The deployment's `api_version`, else `AZURE_API_VERSION`, else LiteLLM's default.
pub fn resolve_api_version(
    api_version: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_non_empty(api_version, lookup, &[AZURE_API_VERSION_ENV])
        .unwrap_or_else(|| AZURE_DEFAULT_API_VERSION.into())
}

/// The deployment's `api_base`, else `AZURE_API_BASE`, else the SDK's `AZURE_OPENAI_ENDPOINT`.
pub fn resolve_api_base(
    api_base: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_non_empty(
        api_base,
        lookup,
        &[AZURE_API_BASE_ENV, AZURE_OPENAI_ENDPOINT_ENV],
    )
    .ok_or(Error::Auth(litellm_auth::Error::MissingApiBase {
        provider: "Azure",
        guidance: "Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.openai.azure.com",
    }))
}

/// The deployment's key, else `AZURE_OPENAI_API_KEY`, else `AZURE_API_KEY`, as `main.py`
/// reads them.
pub fn resolve_api_key(
    api_key: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_non_empty(
        api_key,
        lookup,
        &[AZURE_OPENAI_API_KEY_ENV, AZURE_API_KEY_ENV],
    )
}

/// The Chat Completions URL the SDK builds for `deployment`:
///
/// - `v1`, `latest` and `preview` go to `{api_base}/openai/v1/chat/completions` with no
///   `api-version`, as the plain OpenAI client LiteLLM switches to for them does;
/// - an `api_base` already naming `/openai/deployments/...` is the client's `base_url`, so only
///   `/chat/completions` is added;
/// - any other `api_base` is the resource endpoint, under which the SDK puts
///   `/openai/deployments/{deployment}`.
///
/// A trailing slash on `api_base` is dropped, as `azure_endpoint.rstrip('/')` does.
pub fn complete_url(api_base: &str, deployment: &str, api_version: &str) -> Result<String, Error> {
    if api_base.contains("gateway.ai.cloudflare.com") {
        return Err(Error::Unsupported(
            "Azure deployments behind a Cloudflare AI Gateway api_base",
        ));
    }
    let base = api_base.trim_end_matches('/');
    if is_v1_api_version(api_version) {
        return Ok(format!("{base}/openai/v1/chat/completions"));
    }
    let version: String = url::form_urlencoded::byte_serialize(api_version.as_bytes()).collect();
    Ok(match base.contains("/openai/deployments") {
        true => format!("{base}/chat/completions?api-version={version}"),
        false => {
            format!("{base}/openai/deployments/{deployment}/chat/completions?api-version={version}")
        }
    })
}

/// How a call authenticates once its credential is known.
#[derive(Clone, Debug)]
pub enum AzureCredential {
    ApiKey(SecretValue),
    /// An Entra ID access token, given or acquired.
    AdToken(SecretValue),
}

/// A key goes in `api-key` and a token in `Authorization: Bearer`, as `AzureOpenAI` sends
/// them. On the v1 surface LiteLLM hands either to the plain OpenAI client as its `api_key`,
/// which sends it as a bearer.
pub fn environment(
    headers: Headers,
    credential: AzureCredential,
    v1: bool,
) -> ValidatedEnvironment {
    let (placement, secret) = match credential {
        AzureCredential::ApiKey(secret) if !v1 => {
            (CredentialPlacement::Header(API_KEY_HEADER), secret)
        }
        AzureCredential::ApiKey(secret) | AzureCredential::AdToken(secret) => {
            (CredentialPlacement::Bearer, secret)
        }
    };
    ValidatedEnvironment {
        headers: litellm_http::request::with_default_headers(
            headers,
            &[("content-type", "application/json")],
        ),
        auth: AuthScheme::Credential { placement, secret },
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env(name: &str) -> Option<String> {
        match name {
            AZURE_API_BASE_ENV => Some("https://env.openai.azure.com".into()),
            AZURE_API_VERSION_ENV => Some("2024-10-21".into()),
            AZURE_API_KEY_ENV => Some("env-key".into()),
            _ => None,
        }
    }

    fn endpoint_only(name: &str) -> Option<String> {
        (name == AZURE_OPENAI_ENDPOINT_ENV).then(|| "https://sdk.openai.azure.com".into())
    }

    #[rstest]
    #[case::resource(
        "https://res.openai.azure.com",
        "2024-10-21",
        "https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
    )]
    #[case::trailing_slash(
        "https://res.openai.azure.com/",
        "2024-10-21",
        "https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
    )]
    #[case::preview_version(
        "https://res.openai.azure.com",
        "2025-02-01-preview",
        "https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-02-01-preview"
    )]
    #[case::base_url_with_deployment(
        "https://res.openai.azure.com/openai/deployments/other/",
        "2024-10-21",
        "https://res.openai.azure.com/openai/deployments/other/chat/completions?api-version=2024-10-21"
    )]
    #[case::v1(
        "https://res.openai.azure.com/",
        "v1",
        "https://res.openai.azure.com/openai/v1/chat/completions"
    )]
    #[case::latest(
        "https://res.openai.azure.com",
        "latest",
        "https://res.openai.azure.com/openai/v1/chat/completions"
    )]
    #[case::preview(
        "https://res.openai.azure.com",
        "preview",
        "https://res.openai.azure.com/openai/v1/chat/completions"
    )]
    fn the_url_follows_the_sdk(
        #[case] api_base: &str,
        #[case] api_version: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            complete_url(api_base, "gpt-4o", api_version).unwrap(),
            expected
        );
    }

    #[rstest]
    fn a_cloudflare_gateway_base_is_refused() {
        assert!(matches!(
            complete_url(
                "https://gateway.ai.cloudflare.com/v1/acct/gw/azure-openai/res",
                "gpt-4o",
                "2024-10-21"
            ),
            Err(Error::Unsupported(_))
        ));
    }

    #[rstest]
    #[case::deployment(Some("2024-06-01"), env, "2024-06-01")]
    #[case::environment(None, env, "2024-10-21")]
    #[case::empty_falls_back(Some(""), env, "2024-10-21")]
    #[case::default(None, no_env, AZURE_DEFAULT_API_VERSION)]
    fn the_api_version_prefers_the_deployment_then_the_environment(
        #[case] configured: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        assert_eq!(resolve_api_version(configured, &lookup), expected);
    }

    #[rstest]
    #[case::deployment(
        Some("https://dep.openai.azure.com"),
        env,
        "https://dep.openai.azure.com"
    )]
    #[case::environment(None, env, "https://env.openai.azure.com")]
    #[case::sdk_environment(None, endpoint_only, "https://sdk.openai.azure.com")]
    fn the_api_base_prefers_the_deployment_then_the_environment(
        #[case] configured: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        assert_eq!(resolve_api_base(configured, &lookup).unwrap(), expected);
    }

    #[rstest]
    fn a_missing_api_base_says_where_to_set_it() {
        assert!(matches!(
            resolve_api_base(None, &no_env),
            Err(Error::Auth(litellm_auth::Error::MissingApiBase {
                provider: "Azure",
                ..
            }))
        ));
    }

    #[rstest]
    #[case::deployment(Some("dep-key"), env, Some("dep-key"))]
    #[case::environment(None, env, Some("env-key"))]
    #[case::none(None, no_env, None)]
    fn the_key_prefers_the_deployment_then_the_environment(
        #[case] configured: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(resolve_api_key(configured, &lookup).as_deref(), expected);
    }

    #[rstest]
    fn the_sdk_key_variable_outranks_litellms() {
        let lookup = |name: &str| match name {
            AZURE_OPENAI_API_KEY_ENV => Some("sdk-key".to_string()),
            AZURE_API_KEY_ENV => Some("litellm-key".to_string()),
            _ => None,
        };
        assert_eq!(resolve_api_key(None, &lookup).as_deref(), Some("sdk-key"));
    }

    #[rstest]
    #[case::key(AzureCredential::ApiKey(SecretValue::new("k")), false, "api-key", "k")]
    #[case::token(
        AzureCredential::AdToken(SecretValue::new("t")),
        false,
        "authorization",
        "Bearer t"
    )]
    #[case::v1_key(
        AzureCredential::ApiKey(SecretValue::new("k")),
        true,
        "authorization",
        "Bearer k"
    )]
    #[case::v1_token(
        AzureCredential::AdToken(SecretValue::new("t")),
        true,
        "authorization",
        "Bearer t"
    )]
    #[tokio::test]
    async fn the_credential_lands_in_its_header(
        #[case] credential: AzureCredential,
        #[case] v1: bool,
        #[case] header: &str,
        #[case] value: &str,
    ) {
        let authenticated = crate::base_llm::auth::resolve_auth(
            &litellm_auth::AuthServices::default(),
            environment(Vec::new(), credential, v1),
            &no_env,
        )
        .await
        .unwrap();
        assert_eq!(
            authenticated.headers,
            [
                ("content-type".to_string(), "application/json".to_string()),
                (header.to_string(), value.to_string())
            ]
        );
    }
}
