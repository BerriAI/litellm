//! End-to-end image-edit orchestration.
//!
//! Owns vLLM image edit calls while the Python side stays a thin bridge:
//! resolve URL/auth, build multipart via pure transforms, POST it, and return
//! the OpenAI-compatible response JSON.

use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Duration;

use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::image_edit::transformation::ImageEditProviderConfig;
use litellm_core::image_edit::types::{ImageEditInputFile, ImageEditRequestFormat};
use litellm_core::{CoreResult, LlmProvider};
use serde_json::{Map, Value};

use crate::vllm::image_edit::transformation::VLLM_IMAGE_EDIT_CONFIG;

const IMAGE_EDIT_TIMEOUT_SECS: u64 = 600;
const ERROR_BODY_MAX_CHARS: usize = 256;

fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(IMAGE_EDIT_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest client")
    })
}

fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

fn image_edit_config_for(provider: LlmProvider) -> Option<&'static dyn ImageEditProviderConfig> {
    match provider {
        LlmProvider::Vllm => Some(&VLLM_IMAGE_EDIT_CONFIG),
        _ => None,
    }
}

fn string_headers(extra_headers: Option<Map<String, Value>>) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "image_edit extra_headers.{key} must be a string, got {}",
                        litellm_core::error::json_type_name(&value)
                    ))
                })
        })
        .collect()
}

fn has_header(headers: &[(String, String)], expected: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(expected))
}

fn multipart_form(
    data: Map<String, Value>,
    files: Vec<litellm_core::image_edit::types::ImageEditMultipartPart>,
) -> CoreResult<reqwest::multipart::Form> {
    let mut form = reqwest::multipart::Form::new();
    for (key, value) in data {
        let text = match value {
            Value::String(value) => value,
            Value::Number(value) => value.to_string(),
            Value::Bool(value) => value.to_string(),
            other => other.to_string(),
        };
        form = form.text(key, text);
    }

    for file in files {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(file.data_base64.as_bytes())
            .map_err(|err| {
                CoreError::InvalidRequest(format!(
                    "invalid base64 for multipart field {}: {err}",
                    file.field_name
                ))
            })?;
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name(file.filename)
            .mime_str(&file.content_type)
            .map_err(|err| {
                CoreError::InvalidRequest(format!(
                    "invalid content type for multipart field {}: {err}",
                    file.field_name
                ))
            })?;
        form = form.part(file.field_name, part);
    }
    Ok(form)
}

pub struct ImageEditRequest<'a> {
    pub model: &'a str,
    pub images: Vec<ImageEditInputFile>,
    pub mask: Option<ImageEditInputFile>,
    pub prompt: Option<&'a str>,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: &'a str,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: Map<String, Value>,
    pub timeout: Option<Duration>,
}

pub async fn image_edit(request: ImageEditRequest<'_>) -> CoreResult<Value> {
    let provider = LlmProvider::from_str(request.custom_llm_provider)?;
    let config = image_edit_config_for(provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;

    let api_key = config.resolve_api_key(request.api_key, &|key| std::env::var(key).ok());
    let url = config.complete_url(request.api_base, &|key| std::env::var(key).ok())?;
    let filtered_params = config.map_image_edit_params(&request.optional_params);
    let transformed = config.transform_image_edit_request(
        request.model,
        request.images,
        request.mask,
        request.prompt,
        filtered_params,
    )?;

    let headers = string_headers(request.extra_headers)?;
    let mut request_builder = http_client().post(&url);
    if let Some(api_key) = api_key {
        if !has_header(&headers, "x-api-key") && !has_header(&headers, "authorization") {
            request_builder = request_builder.header("x-api-key", api_key);
        }
    }
    for (key, value) in headers {
        request_builder = request_builder.header(&key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    request_builder = match transformed.format {
        ImageEditRequestFormat::Multipart => {
            request_builder.multipart(multipart_form(transformed.data, transformed.files)?)
        }
        ImageEditRequestFormat::Json => request_builder.json(&Value::Object(transformed.data)),
    };

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

    let response_json: Value = serde_json::from_str(&text).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid image edit response JSON: {err}"))
    })?;

    Ok(config
        .transform_image_edit_response(request.model, response_json)?
        .into_json())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn image_edit_registry_supports_only_vllm() {
        assert!(image_edit_config_for(LlmProvider::Vllm).is_some());
        assert!(image_edit_config_for(LlmProvider::Openai).is_none());
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({"x-retry-count": 3}).as_object().unwrap().clone();
        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            CoreError::InvalidRequest(
                "image_edit extra_headers.x-retry-count must be a string, got number".to_string()
            )
        );
    }

    #[test]
    fn multipart_form_rejects_invalid_base64() {
        let file = litellm_core::image_edit::types::ImageEditMultipartPart {
            field_name: "image[]".to_string(),
            filename: "image.png".to_string(),
            content_type: "image/png".to_string(),
            data_base64: "not base64".to_string(),
        };

        let err = multipart_form(Map::new(), vec![file]).expect_err("invalid base64 rejected");
        assert!(err.to_string().contains("invalid base64"));
    }

    #[tokio::test]
    async fn image_edit_posts_vllm_multipart_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let addr = listener.local_addr().expect("listener has local addr");

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts one request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let n = socket.read(&mut buffer).await.expect("reads request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..n]);
                if request.windows(5).any(|window| window == b"image") {
                    break;
                }
            }

            let response_body = r#"{"created":1,"data":[{"b64_json":"ZmFrZQ=="}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("writes response");
            String::from_utf8_lossy(&request).to_string()
        });

        let response = image_edit(ImageEditRequest {
            model: "qwen-image-edit",
            images: vec![ImageEditInputFile {
                filename: "image.png".to_string(),
                content_type: "image/png".to_string(),
                data_base64: "aW1hZ2U=".to_string(),
            }],
            mask: Some(ImageEditInputFile {
                filename: "mask.png".to_string(),
                content_type: "image/png".to_string(),
                data_base64: "bWFzaw==".to_string(),
            }),
            prompt: Some("make it brighter"),
            api_key: Some("sk-test"),
            api_base: Some(&format!("http://{addr}")),
            custom_llm_provider: "vllm",
            extra_headers: None,
            optional_params: json!({"quality": "high"}).as_object().unwrap().clone(),
            timeout: Some(Duration::from_secs(5)),
        })
        .await
        .expect("image edit request succeeds");

        assert_eq!(response["data"][0]["b64_json"], "ZmFrZQ==");

        let request = server.await.expect("server task completes");
        assert!(request.starts_with("POST /v1/images/edits "), "{request}");
        assert!(request.contains("x-api-key: sk-test"), "{request}");
        assert!(request.contains("name=\"model\""), "{request}");
        assert!(request.contains("qwen-image-edit"), "{request}");
        assert!(request.contains("name=\"prompt\""), "{request}");
        assert!(request.contains("make it brighter"), "{request}");
        assert!(request.contains("name=\"quality\""), "{request}");
        assert!(request.contains("high"), "{request}");
        assert!(request.contains("name=\"image[]\""), "{request}");
        assert!(request.contains("filename=\"image.png\""), "{request}");
        assert!(request.contains("name=\"mask\""), "{request}");
        assert!(request.contains("filename=\"mask.png\""), "{request}");
    }
}
