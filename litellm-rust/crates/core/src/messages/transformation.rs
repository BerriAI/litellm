use super::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};
use crate::Error;

use crate::auth::CredentialPlacement;

pub trait AnthropicMessagesProviderConfig: Sync {
    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn credential_placement(&self) -> CredentialPlacement {
        CredentialPlacement::Header("x-api-key")
    }

    fn accepts_bearer_auth(&self) -> bool {
        false
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> Result<AnthropicMessagesRequest, Error> {
        Ok(request)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_response(
        &self,
        _model: &str,
        response: AnthropicMessagesResponse,
    ) -> Result<AnthropicMessagesResponse, Error> {
        Ok(response)
    }
}
