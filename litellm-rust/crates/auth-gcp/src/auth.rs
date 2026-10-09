use std::{future::Future, path::Path, pin::Pin, sync::Arc};

use gcp_auth::{CustomServiceAccount, TokenProvider};
use litellm_auth_types::{CredentialPlacement, Error, ErrorDetail, http::apply_credential};
use moka::future::Cache;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    config::{
        CredentialSource, VertexConfig, credential_source, get_vertex_ai_project, non_empty_env,
    },
    constants::{
        CLOUD_PLATFORM_SCOPE, GOOGLE_OAUTH_TOKEN_ENDPOINT, VERTEX_AI_API_KEY_ENV,
        VERTEXAI_API_KEY_ENV,
    },
};

pub struct VertexEnvironment {
    pub headers: Vec<(String, String)>,
    pub project_id: String,
}

struct VertexAccessToken {
    token: String,
    project_id: String,
}

#[derive(Clone)]
pub struct VertexAuth {
    providers: Cache<CredentialCacheKey, Arc<dyn VertexTokenSource>>,
    loader: Arc<dyn VertexProviderLoader>,
}

impl Default for VertexAuth {
    fn default() -> Self {
        Self::new(Arc::new(GcpProviderLoader))
    }
}

impl VertexAuth {
    pub fn new(loader: Arc<dyn VertexProviderLoader>) -> Self {
        Self {
            providers: Cache::builder().max_capacity(64).build(),
            loader,
        }
    }

    pub async fn access_token(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<String, Error> {
        self.load_provider(config, env_lookup).await?.token().await
    }

    pub async fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<VertexEnvironment, Error> {
        let has_authorization = headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("Authorization"));
        let static_token = api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| non_empty_env(env_lookup, VERTEX_AI_API_KEY_ENV))
            .or_else(|| non_empty_env(env_lookup, VERTEXAI_API_KEY_ENV));
        let project_id = get_vertex_ai_project(config, env_lookup);

        if !has_authorization && static_token.is_none() {
            let access = self.get_access_token(config, env_lookup).await?;
            return Ok(VertexEnvironment {
                headers: apply_credential(headers, &access.token, CredentialPlacement::Bearer)?,
                project_id: project_id.unwrap_or(access.project_id),
            });
        }

        let project_id = match project_id {
            Some(project_id) => project_id,
            None => {
                self.load_provider(config, env_lookup)
                    .await?
                    .project_id()
                    .await?
            }
        };
        let headers = if has_authorization {
            headers
        } else {
            apply_credential(
                headers,
                static_token.as_deref().expect("static token was checked"),
                CredentialPlacement::Bearer,
            )?
        };
        Ok(VertexEnvironment {
            headers,
            project_id,
        })
    }

    async fn get_access_token(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<VertexAccessToken, Error> {
        let provider = self.load_provider(config, env_lookup).await?;
        let (token, project_id) = tokio::try_join!(provider.token(), provider.project_id())?;
        Ok(VertexAccessToken { token, project_id })
    }

    async fn load_provider(
        &self,
        config: &VertexConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Arc<dyn VertexTokenSource>, Error> {
        let source = credential_source(config, env_lookup);
        let key = source.cache_key();
        self.providers
            .try_get_with(key, self.loader.load(source))
            .await
            .map_err(|error| (*error).clone())
    }
}

pub trait VertexTokenSource: Send + Sync {
    fn project_id(&self) -> VertexAuthFuture<'_, String>;
    fn token(&self) -> VertexAuthFuture<'_, String>;
}

pub trait VertexProviderLoader: Send + Sync {
    fn load(&self, source: CredentialSource) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>>;
}

pub type VertexAuthFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

struct GcpTokenSource(Arc<dyn TokenProvider>);

impl VertexTokenSource for GcpTokenSource {
    fn project_id(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async move {
            self.0
                .project_id()
                .await
                .map(|project| project.to_string())
                .map_err(auth_acquisition_error)
        })
    }

    fn token(&self) -> VertexAuthFuture<'_, String> {
        Box::pin(async move {
            self.0
                .token(&[CLOUD_PLATFORM_SCOPE])
                .await
                .map(|token| token.as_str().to_string())
                .map_err(auth_acquisition_error)
        })
    }
}

struct GcpProviderLoader;

impl VertexProviderLoader for GcpProviderLoader {
    fn load(&self, source: CredentialSource) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>> {
        Box::pin(async move {
            let provider: Arc<dyn TokenProvider> = match source {
                CredentialSource::Inline(configured) => Arc::new(
                    CustomServiceAccount::from_json(validate_request_credentials(
                        configured.expose(),
                    )?)
                    .map_err(auth_acquisition_error)?,
                ),
                CredentialSource::Trusted(configured) => {
                    let configured = configured.expose();
                    let service_account = if Path::new(configured).is_file() {
                        CustomServiceAccount::from_file(configured)
                    } else {
                        CustomServiceAccount::from_json(configured)
                    }
                    .map_err(auth_acquisition_error)?;
                    Arc::new(service_account)
                }
                CredentialSource::ApplicationCredentials(path) => {
                    Arc::new(CustomServiceAccount::from_file(path).map_err(auth_acquisition_error)?)
                }
                CredentialSource::Adc => {
                    gcp_auth::provider().await.map_err(auth_acquisition_error)?
                }
            };
            Ok(Arc::new(GcpTokenSource(provider)) as Arc<dyn VertexTokenSource>)
        })
    }
}

fn validate_request_credentials(configured: &str) -> Result<&str, Error> {
    let token_uri = serde_json::from_str::<Value>(configured)
        .ok()
        .and_then(|credentials| {
            credentials
                .get("token_uri")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    if token_uri.as_deref() != Some(GOOGLE_OAUTH_TOKEN_ENDPOINT) {
        return Err(Error::InvalidConfiguration(ErrorDetail::InvalidType {
            field: "request-controlled vertex_credentials token_uri".into(),
            expected: "the canonical Google OAuth token endpoint",
        }));
    }
    Ok(configured)
}

impl CredentialSource {
    fn cache_key(&self) -> CredentialCacheKey {
        match self {
            Self::Inline(configured) => {
                CredentialCacheKey::Inline(Sha256::digest(configured.expose()).into())
            }
            Self::Trusted(configured) => {
                CredentialCacheKey::Trusted(Sha256::digest(configured.expose()).into())
            }
            Self::ApplicationCredentials(path) => {
                CredentialCacheKey::ApplicationCredentials(path.clone())
            }
            Self::Adc => CredentialCacheKey::Adc,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum CredentialCacheKey {
    Inline([u8; 32]),
    Trusted([u8; 32]),
    ApplicationCredentials(String),
    Adc,
}

fn auth_acquisition_error(error: gcp_auth::Error) -> Error {
    Error::CredentialAcquisition(litellm_auth_types::ErrorDetail::failed(
        "Vertex AI credentials",
        error,
    ))
}

#[cfg(test)]
mod tests {
    use litellm_auth_types::SecretValue;

    use super::*;

    #[rstest::rstest]
    #[case::canonical_endpoint(r#"{"token_uri":"https://oauth2.googleapis.com/token"}"#, true)]
    #[case::noncanonical_endpoint(r#"{"token_uri":"http://127.0.0.1/token"}"#, false)]
    #[case::missing_endpoint("{}", false)]
    fn request_credentials_require_canonical_token_endpoint(
        #[case] credentials: &str,
        #[case] accepted: bool,
    ) {
        assert_eq!(validate_request_credentials(credentials).is_ok(), accepted);
    }

    #[rstest::rstest]
    fn inline_and_trusted_credentials_never_share_a_cache_entry() {
        assert_ne!(
            CredentialSource::Inline(SecretValue::new("same-value")).cache_key(),
            CredentialSource::Trusted(SecretValue::new("same-value")).cache_key()
        );
    }
}
