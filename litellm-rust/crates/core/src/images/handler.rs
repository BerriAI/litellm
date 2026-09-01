use crate::error::{CoreError, CoreResult};

use super::client::http_client;
use super::types::{
    ImagesEditResponse, ImagesGenerationResponse, ProviderImagesEditRequest,
    ProviderImagesGenerationRequest,
};

pub(super) async fn execute_images_generation_provider_call(
    request: ProviderImagesGenerationRequest,
) -> CoreResult<ImagesGenerationResponse> {
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
            CoreError::InvalidResponse(format!("invalid images generation response JSON: {err}"))
        })?,
    )?;

    serde_json::from_value(transformed).map_err(|err| {
        CoreError::InvalidResponse(format!("invalid images generation response: {err}"))
    })
}

pub(super) async fn execute_images_edit_provider_call(
    request: ProviderImagesEditRequest,
) -> CoreResult<ImagesEditResponse> {
    // For edit requests, we need to use multipart/form-data
    let mut form = reqwest::multipart::Form::new().text(
        "prompt",
        request
            .body
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    );

    if let Some(model) = request.body.get("model").and_then(|v| v.as_str()) {
        form = form.text("model", model.to_string());
    }

    if let Some(n) = request.body.get("n").and_then(|v| v.as_u64()) {
        form = form.text("n", n.to_string());
    }

    if let Some(size) = request.body.get("size").and_then(|v| v.as_str()) {
        form = form.text("size", size.to_string());
    }

    if let Some(response_format) = request.body.get("response_format").and_then(|v| v.as_str()) {
        form = form.text("response_format", response_format.to_string());
    }

    if let Some(user) = request.body.get("user").and_then(|v| v.as_str()) {
        form = form.text("user", user.to_string());
    }

    // Add image file
    let image_part = reqwest::multipart::Part::bytes(request.image)
        .file_name("image.png")
        .mime_str("image/png")
        .map_err(|err| CoreError::InvalidRequest(format!("failed to create image part: {err}")))?;
    form = form.part("image", image_part);

    // Add mask file if present
    if let Some(mask) = request.mask {
        let mask_part = reqwest::multipart::Part::bytes(mask)
            .file_name("mask.png")
            .mime_str("image/png")
            .map_err(|err| {
                CoreError::InvalidRequest(format!("failed to create mask part: {err}"))
            })?;
        form = form.part("mask", mask_part);
    }

    let mut request_builder = http_client().post(&request.url).multipart(form);
    for (key, value) in &request.upstream_headers {
        // Skip content-type header for multipart requests - reqwest will set it automatically
        if key.to_lowercase() != "content-type" {
            request_builder = request_builder.header(key, value);
        }
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
            CoreError::InvalidResponse(format!("invalid images edit response JSON: {err}"))
        })?,
    )?;

    serde_json::from_value(transformed)
        .map_err(|err| CoreError::InvalidResponse(format!("invalid images edit response: {err}")))
}

fn truncate_error_body(body: &str) -> String {
    const MAX_LEN: usize = 500;
    if body.len() <= MAX_LEN {
        body.to_string()
    } else {
        format!("{}...", &body[..MAX_LEN])
    }
}
