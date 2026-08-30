//! Validation middleware for request validation.
//!
//! Validates incoming requests before they reach the handlers, including
//! request size, message format, model names, and input sanitization.

use crate::validation::{InputValidator, ValidationConfig, ValidationError};
use axum::{body::Body, extract::Request, http::StatusCode, middleware::Next, response::Response};
use serde_json::Value;

/// Validation middleware that validates incoming requests.
pub async fn validation_middleware(
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Get content length
    let content_length = request
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    // Create validator with default config
    // In production, this would be loaded from config
    let validator = InputValidator::new(ValidationConfig::default());

    // Check request size
    if content_length > validator.config.max_request_size {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }

    // For POST/PUT/PATCH requests, validate the body
    if matches!(
        request.method(),
        &axum::http::Method::POST | &axum::http::Method::PUT | &axum::http::Method::PATCH
    ) {
        // We can't easily read the body here without consuming it
        // In a real implementation, we'd use a body extractor or buffer the body
        // For now, we'll skip body validation in the middleware
        // and rely on handler-level validation
    }

    // Continue to the next middleware/handler
    Ok(next.run(request).await)
}

/// Validate a JSON request body.
pub fn validate_json_body(
    body: &Value,
    config: &ValidationConfig,
) -> Result<(), Vec<ValidationError>> {
    let validator = InputValidator::new(config.clone());

    // Validate model name
    if let Some(model) = body.get("model") {
        if let Some(model_str) = model.as_str() {
            if model_str.len() > config.max_model_name_length {
                return Err(vec![ValidationError {
                    field: "model".to_string(),
                    message: format!(
                        "Model name length {} exceeds maximum {}",
                        model_str.len(),
                        config.max_model_name_length
                    ),
                    code: crate::validation::ValidationErrorCode::ModelNameTooLong,
                }]);
            }

            if !validator.is_valid_model_name(model_str) {
                return Err(vec![ValidationError {
                    field: "model".to_string(),
                    message: format!("Invalid model name format: {}", model_str),
                    code: crate::validation::ValidationErrorCode::InvalidModelName,
                }]);
            }
        }
    }

    // Validate messages
    if let Some(messages) = body.get("messages") {
        if let Some(messages_arr) = messages.as_array() {
            if messages_arr.len() > config.max_messages {
                return Err(vec![ValidationError {
                    field: "messages".to_string(),
                    message: format!(
                        "Number of messages {} exceeds maximum {}",
                        messages_arr.len(),
                        config.max_messages
                    ),
                    code: crate::validation::ValidationErrorCode::TooManyMessages,
                }]);
            }

            for (i, msg) in messages_arr.iter().enumerate() {
                if let Some(content) = msg.get("content") {
                    if let Some(content_str) = content.as_str() {
                        if content_str.len() > config.max_message_length {
                            return Err(vec![ValidationError {
                                field: format!("messages[{}].content", i),
                                message: format!(
                                    "Message content length {} exceeds maximum {}",
                                    content_str.len(),
                                    config.max_message_length
                                ),
                                code: crate::validation::ValidationErrorCode::MessageTooLong,
                            }]);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_json_body_valid() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        let config = ValidationConfig::default();
        assert!(validate_json_body(&body, &config).is_ok());
    }

    #[test]
    fn test_validate_json_body_invalid_model() {
        let body = json!({
            "model": "gpt-4<script>",
            "messages": []
        });

        let config = ValidationConfig::default();
        let result = validate_json_body(&body, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_json_body_too_many_messages() {
        let body = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "1"},
                {"role": "user", "content": "2"},
                {"role": "user", "content": "3"}
            ]
        });

        let config = ValidationConfig {
            max_messages: 2,
            ..Default::default()
        };
        let result = validate_json_body(&body, &config);
        assert!(result.is_err());
    }
}
