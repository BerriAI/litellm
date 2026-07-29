use crate::constants::ANTHROPIC_MESSAGES_PROVIDER;
use crate::error::{CoreError, CoreResult};
use crate::logging::http::{JsonRequest, execute_json, execute_stream};

use super::client::http_client;
use super::types::{AnthropicMessagesResponse, ProviderMessagesRequest};

pub(super) async fn execute_messages_provider_call(
    request: ProviderMessagesRequest,
) -> CoreResult<AnthropicMessagesResponse> {
    let response = execute_json::<AnthropicMessagesResponse>(
        http_client(),
        JsonRequest {
            logger: request.logger,
            model: request.model.clone(),
            stream: false,
            url: request.url,
            headers: request.upstream_headers,
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
    if request.provider != ANTHROPIC_MESSAGES_PROVIDER {
        return Err(CoreError::InvalidRequest(
            "streaming messages is not supported for this provider".to_string(),
        ));
    }

    let logger = request.logger.clone();
    let response = execute_stream(
        http_client(),
        JsonRequest {
            logger: logger.clone(),
            model: request.model,
            stream: true,
            url: request.url,
            headers: request.upstream_headers,
            body: request.body,
            timeout: request.timeout,
        },
    )
    .await?;
    Ok(super::types::MessagesStreamResponse { response, logger })
}
