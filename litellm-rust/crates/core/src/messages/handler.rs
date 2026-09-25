use std::time::Duration;

use litellm_http::{request::http_request, transport::Error as TransportError};
use litellm_llms::base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig;
use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use serde_json::Value;

use super::{Error, client::http_client, common_utils::truncate_error_body};

pub(super) fn network(error: reqwest::Error) -> Error {
    Error::Transport(TransportError::Network(error.to_string()))
}

pub(super) async fn send(
    url: &str,
    headers: &[(String, String)],
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Response, Error> {
    let encoded = serde_json::to_vec(body)
        .map_err(|err| Error::InvalidRequest(format!("failed to encode messages body: {err}")))?;
    let builder = headers.iter().fold(
        http_client().post(url).body(encoded),
        |builder, (key, value)| builder.header(key, value),
    );
    let builder = match timeout {
        Some(duration) => builder.timeout(duration),
        None => builder,
    };
    http_request(builder).await.map_err(network)
}

pub(super) async fn provider_error(response: reqwest::Response) -> Error {
    let status = response.status().as_u16();
    match response.text().await {
        Ok(text) => Error::Transport(TransportError::Http {
            status,
            body: truncate_error_body(&text),
        }),
        Err(error) => network(error),
    }
}

pub(super) fn decode_response(
    config: &dyn BaseAnthropicMessagesConfig,
    model: &str,
    text: &str,
) -> Result<AnthropicMessagesResponse, Error> {
    let response = serde_json::from_str(text)
        .map_err(|err| Error::InvalidResponse(format!("invalid messages response JSON: {err}")))?;
    config
        .transform_anthropic_messages_response(model, response)
        .map_err(Error::from)
}
