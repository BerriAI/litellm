use crate::error::{CoreError, CoreResult};

use super::client::http_client;
use super::types::{EmbeddingsResponse, ProviderEmbeddingsRequest};

pub(super) async fn execute_embeddings_provider_call(
    request: ProviderEmbeddingsRequest,
) -> CoreResult<EmbeddingsResponse> {
    let mut request_builder = http_client().post(&request.url).json(&request.body);
    for (key, value) in &request.upstream_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = request_builder
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let transformed = request.config.transform_response(
        &request.model,
        serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid embeddings response JSON: {err}"))
        })?,
    )?;

    serde_json::from_value(transformed)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid embeddings response: {err}")))
}

fn truncate_error_body(body: &str) -> String {
    const MAX_LEN: usize = 500;
    if body.len() <= MAX_LEN {
        body.to_string()
    } else {
        format!("{}...", &body[..MAX_LEN])
    }
}
