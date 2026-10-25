use std::sync::Arc;

use google_cloud_auth::credentials::{CacheableResource, CredentialsProvider, EntityTag};
use google_cloud_auth::errors::CredentialsError;
use http::{Extensions, HeaderMap, HeaderName, HeaderValue};
use litellm_auth_types::Error;

use crate::{VertexAuth, VertexConfig};

type EnvironmentLookup = dyn Fn(&str) -> Option<String> + Send + Sync;

pub struct GoogleCredentials {
    auth: VertexAuth,
    config: VertexConfig,
    environment: Arc<EnvironmentLookup>,
}

impl GoogleCredentials {
    pub fn new(config: VertexConfig, environment: Arc<EnvironmentLookup>) -> Self {
        Self {
            auth: VertexAuth::default(),
            config,
            environment,
        }
    }

    pub async fn request_headers(&self) -> Result<HeaderMap, Error> {
        let response = self
            .auth
            .validate_environment(Vec::new(), None, &self.config, &|name| {
                (self.environment)(name)
            })
            .await?;
        response
            .headers
            .into_iter()
            .map(|(key, value)| {
                let name =
                    HeaderName::from_bytes(key.as_bytes()).map_err(|_| Error::InvalidHeader)?;
                let value = HeaderValue::from_str(&value).map_err(|_| Error::InvalidHeader)?;
                Ok((name, value))
            })
            .collect()
    }
}

impl CredentialsProvider for GoogleCredentials {
    async fn headers(
        &self,
        _: Extensions,
    ) -> Result<CacheableResource<HeaderMap>, CredentialsError> {
        self.request_headers()
            .await
            .map(|data| CacheableResource::New {
                entity_tag: EntityTag::new(),
                data,
            })
            .map_err(|_| CredentialsError::from_msg(false, "Google authentication failed"))
    }

    async fn universe_domain(&self) -> Option<String> {
        None
    }
}

impl std::fmt::Debug for GoogleCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleCredentials").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sdk_and_http_credentials_share_token_resolution_and_redaction() {
        let credentials = GoogleCredentials::new(
            VertexConfig::new(None, Some("project".into()), None),
            Arc::new(|name| (name == "VERTEX_AI_API_KEY").then(|| "private-token".into())),
        );
        let direct = credentials.request_headers().await.unwrap();
        let CacheableResource::New { data, .. } =
            credentials.headers(Extensions::new()).await.unwrap()
        else {
            panic!("first request did not return headers");
        };
        assert_eq!(direct, data);
        assert_eq!(data[http::header::AUTHORIZATION], "Bearer private-token");
        assert!(!format!("{credentials:?}").contains("private-token"));
    }

    #[tokio::test]
    async fn invalid_token_headers_return_a_redacted_sdk_error() {
        let credentials = GoogleCredentials::new(
            VertexConfig::new(None, Some("project".into()), None),
            Arc::new(|name| (name == "VERTEX_AI_API_KEY").then(|| "private\nvalue".into())),
        );
        assert_eq!(
            credentials.request_headers().await.unwrap_err(),
            Error::InvalidHeader
        );
        let error = credentials.headers(Extensions::new()).await.unwrap_err();
        assert!(!format!("{error:?}").contains("private"));
    }
}
