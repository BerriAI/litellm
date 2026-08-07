use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::CoreResult;
use crate::error::CoreError;

use super::{CallLogger, ResponseBody};

pub struct JsonRequest {
    pub logger: Arc<CallLogger>,
    pub model: String,
    pub stream: bool,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
    pub timeout: Option<Duration>,
}

pub async fn execute_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    request: JsonRequest,
) -> CoreResult<T> {
    let body_bytes = serde_json::to_vec(&request.body)
        .map_err(|error| CoreError::InvalidRequest(error.to_string()))?;
    request
        .logger
        .request_about_to_be_sent(super::RequestEventInput {
            model: request.model,
            stream: request.stream,
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: request.body,
        });
    let builder = request.headers.iter().fold(
        client.post(&request.url).body(body_bytes),
        |builder, (name, value)| builder.header(name, value),
    );
    let builder = match request.timeout {
        Some(timeout) => builder.timeout(timeout),
        None => builder,
    };
    let response = builder.send().await.map_err(|error| {
        request
            .logger
            .failure(None, "network_error", error.to_string(), None);
        CoreError::Network(error.to_string())
    })?;
    let status = response.status();
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect::<Vec<_>>();
    let media_type = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.clone());
    let text = response.text().await.map_err(|error| {
        request.logger.failure(
            Some(status.as_u16()),
            "network_error",
            error.to_string(),
            None,
        );
        CoreError::Network(error.to_string())
    })?;
    if !status.is_success() {
        let body = serde_json::from_str(&text)
            .map(ResponseBody::Json)
            .unwrap_or(ResponseBody::Binary {
                media_type: media_type.clone(),
                bytes: text.len(),
            });
        request.logger.failure(
            Some(status.as_u16()),
            "http_error",
            format!("provider returned HTTP {}", status.as_u16()),
            Some(body),
        );
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: crate::utils::truncate_error_body(&text),
        });
    }
    let value = serde_json::from_str::<Value>(&text).map_err(|error| {
        request.logger.failure(
            Some(status.as_u16()),
            "invalid_json",
            error.to_string(),
            Some(ResponseBody::Binary {
                media_type: media_type.clone(),
                bytes: text.len(),
            }),
        );
        CoreError::InvalidResponse(format!("invalid provider response JSON: {error}"))
    })?;
    let typed = T::deserialize(&value);
    request
        .logger
        .response_received(status.as_u16(), headers, ResponseBody::Json(value));
    let typed = typed.map_err(|error| {
        request.logger.failure(
            Some(status.as_u16()),
            "invalid_json",
            error.to_string(),
            None,
        );
        CoreError::InvalidResponse(format!("invalid provider response: {error}"))
    })?;
    Ok(typed)
}

pub async fn execute_stream(
    client: &reqwest::Client,
    request: JsonRequest,
) -> CoreResult<reqwest::Response> {
    let body_bytes = serde_json::to_vec(&request.body)
        .map_err(|error| CoreError::InvalidRequest(error.to_string()))?;
    request
        .logger
        .request_about_to_be_sent(super::RequestEventInput {
            model: request.model,
            stream: true,
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: request.body,
        });
    let builder = request.headers.iter().fold(
        client.post(&request.url).body(body_bytes),
        |builder, (name, value)| builder.header(name, value),
    );
    let builder = match request.timeout {
        Some(timeout) => builder.timeout(timeout),
        None => builder,
    };
    let response = builder.send().await.map_err(|error| {
        request
            .logger
            .failure(None, "network_error", error.to_string(), None);
        CoreError::Network(error.to_string())
    })?;
    let status = response.status();
    if !status.is_success() {
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect::<Vec<_>>();
        let text = response.text().await.map_err(|error| {
            request.logger.failure(
                Some(status.as_u16()),
                "network_error",
                error.to_string(),
                None,
            );
            CoreError::Network(error.to_string())
        })?;
        let body = serde_json::from_str(&text)
            .map(ResponseBody::Json)
            .unwrap_or(ResponseBody::Binary {
                media_type: headers
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
                    .map(|(_, value)| value.clone()),
                bytes: text.len(),
            });
        request.logger.failure(
            Some(status.as_u16()),
            "http_error",
            format!("provider returned HTTP {}", status.as_u16()),
            Some(body),
        );
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: crate::utils::truncate_error_body(&text),
        });
    }
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    request.logger.stream_started(status.as_u16(), content_type);
    Ok(response)
}
