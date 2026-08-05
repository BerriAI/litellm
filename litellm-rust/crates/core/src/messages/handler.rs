#[cfg(feature = "bedrock-auth")]
use std::collections::BTreeMap;
#[cfg(feature = "bedrock-auth")]
use std::time::SystemTime;

use crate::error::{CoreError, CoreResult};
use crate::logging::http::{JsonRequest, execute_json, execute_stream};
#[cfg(feature = "bedrock-auth")]
use crate::providers::bedrock::aws_base::{AwsAuthConfig, resolve_credentials, sign_bedrock_post};

use super::client::http_client;
use super::common_utils::has_bearer_auth;
use super::types::{AnthropicMessagesResponse, ProviderMessagesRequest};

async fn upstream_headers(request: &ProviderMessagesRequest) -> CoreResult<Vec<(String, String)>> {
    if request.provider != "bedrock" || has_bearer_auth(&request.upstream_headers) {
        return Ok(request.upstream_headers.clone());
    }

    #[cfg(feature = "bedrock-auth")]
    {
        let body = serde_json::to_vec(&request.body)
            .map_err(|error| CoreError::InvalidRequest(format!("invalid messages request body: {error}")))?;
        let env_lookup = |key: &str| std::env::var(key).ok();
        let region = request
            .config
            .signing_region(None, &env_lookup)
            .ok_or_else(|| CoreError::Auth("Bedrock signing region is unavailable".to_string()))?;
        let headers = BTreeMap::from_iter(request.upstream_headers.iter().cloned());
        let credentials = resolve_credentials(AwsAuthConfig::default(), &env_lookup).await?;
        let signed_headers = sign_bedrock_post(
            &request.url,
            &body,
            &headers,
            &region,
            &credentials,
            SystemTime::now(),
        )?;
        Ok(headers.into_iter().chain(signed_headers).collect())
    }

    #[cfg(not(feature = "bedrock-auth"))]
    Err(CoreError::Auth(
        "Bedrock SigV4 authentication requires the `bedrock-auth` feature".to_string(),
    ))
}

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<AnthropicMessagesResponse> {
    let headers = upstream_headers(&request).await?;
    let response = execute_json::<AnthropicMessagesResponse>(
        http_client(),
        JsonRequest {
            logger: request.logger,
            model: request.model.clone(),
            stream: false,
            url: request.url,
            headers,
            body: request.body,
            timeout: request.timeout,
        },
    )
    .await?;
    request.config.transform_response(&request.model, response)
}

pub(super) async fn execute_messages_provider_stream(
    request: ProviderMessagesRequest,
) -> CoreResult<super::types::MessagesStreamResponse> {
    let headers = upstream_headers(&request).await?;
    let provider = request.provider;
    let logger = request.logger.clone();
    let response = execute_stream(
        http_client(),
        JsonRequest {
            logger: logger.clone(),
            model: request.model,
            stream: true,
            url: request.url,
            headers,
            body: request.body,
            timeout: request.timeout,
        },
    )
    .await?;
    Ok(super::types::MessagesStreamResponse {
        provider,
        response,
        logger,
    })
}
