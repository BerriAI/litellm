//! Comprehensive input validation and sanitization.
//!
//! Validates request sizes, message formats, model names, and sanitizes
//! inputs to prevent injection attacks.

use serde_json::Value;

/// Validation configuration.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum request size in bytes.
    pub max_request_size: usize,
    /// Maximum message content length.
    pub max_message_length: usize,
    /// Maximum number of messages in a request.
    pub max_messages: usize,
    /// Allowed model name patterns.
    pub allowed_model_patterns: Vec<String>,
    /// Maximum model name length.
    pub max_model_name_length: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_request_size: 10 * 1024 * 1024, // 10 MB
            max_message_length: 100_000,
            max_messages: 1000,
            allowed_model_patterns: vec![
                r"^[a-zA-Z0-9\-_/\.]+$".to_string(), // Alphanumeric, hyphens, underscores, slashes, dots
            ],
            max_model_name_length: 256,
        }
    }
}

/// Validation error.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub code: ValidationErrorCode,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({})", self.field, self.message, self.code)
    }
}

/// Validation error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    RequestTooLarge,
    MessageTooLong,
    TooManyMessages,
    InvalidModelName,
    ModelNameTooLong,
    InvalidJson,
    MissingRequiredField,
    InvalidFieldType,
    InvalidValue,
}

impl std::fmt::Display for ValidationErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationErrorCode::RequestTooLarge => write!(f, "REQUEST_TOO_LARGE"),
            ValidationErrorCode::MessageTooLong => write!(f, "MESSAGE_TOO_LONG"),
            ValidationErrorCode::TooManyMessages => write!(f, "TOO_MANY_MESSAGES"),
            ValidationErrorCode::InvalidModelName => write!(f, "INVALID_MODEL_NAME"),
            ValidationErrorCode::ModelNameTooLong => write!(f, "MODEL_NAME_TOO_LONG"),
            ValidationErrorCode::InvalidJson => write!(f, "INVALID_JSON"),
            ValidationErrorCode::MissingRequiredField => write!(f, "MISSING_REQUIRED_FIELD"),
            ValidationErrorCode::InvalidFieldType => write!(f, "INVALID_FIELD_TYPE"),
            ValidationErrorCode::InvalidValue => write!(f, "INVALID_VALUE"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Input validator.
pub struct InputValidator {
    /// Validation configuration.
    pub config: ValidationConfig,
}

impl InputValidator {
    /// Create a new input validator.
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Check if a model name is valid.
    pub fn is_valid_model_name(&self, model_name: &str) -> bool {
        for pattern in &self.config.allowed_model_patterns {
            if let Ok(regex) = regex::Regex::new(pattern)
                && regex.is_match(model_name)
            {
                return true;
            }
        }
        false
    }

    /// Validate a request.
    pub fn validate_request(
        &self,
        request: &Value,
        size: usize,
    ) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check request size
        if size > self.config.max_request_size {
            errors.push(ValidationError {
                field: "request".to_string(),
                message: format!(
                    "Request size {} exceeds maximum {}",
                    size, self.config.max_request_size
                ),
                code: ValidationErrorCode::RequestTooLarge,
            });
        }

        // Validate model name
        if let Some(model) = request.get("model") {
            if let Some(model_str) = model.as_str() {
                if model_str.len() > self.config.max_model_name_length {
                    errors.push(ValidationError {
                        field: "model".to_string(),
                        message: format!(
                            "Model name length {} exceeds maximum {}",
                            model_str.len(),
                            self.config.max_model_name_length
                        ),
                        code: ValidationErrorCode::ModelNameTooLong,
                    });
                } else if !self.is_valid_model_name(model_str) {
                    errors.push(ValidationError {
                        field: "model".to_string(),
                        message: format!("Invalid model name format: {}", model_str),
                        code: ValidationErrorCode::InvalidModelName,
                    });
                }
            } else {
                errors.push(ValidationError {
                    field: "model".to_string(),
                    message: "Model must be a string".to_string(),
                    code: ValidationErrorCode::InvalidFieldType,
                });
            }
        } else {
            errors.push(ValidationError {
                field: "model".to_string(),
                message: "Model is required".to_string(),
                code: ValidationErrorCode::MissingRequiredField,
            });
        }

        // Validate messages
        if let Some(messages) = request.get("messages") {
            if let Some(messages_arr) = messages.as_array() {
                if messages_arr.len() > self.config.max_messages {
                    errors.push(ValidationError {
                        field: "messages".to_string(),
                        message: format!(
                            "Number of messages {} exceeds maximum {}",
                            messages_arr.len(),
                            self.config.max_messages
                        ),
                        code: ValidationErrorCode::TooManyMessages,
                    });
                } else {
                    for (i, msg) in messages_arr.iter().enumerate() {
                        if let Some(content) = msg.get("content")
                            && let Some(content_str) = content.as_str()
                            && content_str.len() > self.config.max_message_length
                        {
                            errors.push(ValidationError {
                                field: format!("messages[{}].content", i),
                                message: format!(
                                    "Message content length {} exceeds maximum {}",
                                    content_str.len(),
                                    self.config.max_message_length
                                ),
                                code: ValidationErrorCode::MessageTooLong,
                            });
                        }
                    }
                }
            } else {
                errors.push(ValidationError {
                    field: "messages".to_string(),
                    message: "Messages must be an array".to_string(),
                    code: ValidationErrorCode::InvalidFieldType,
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Sanitize a string input by removing potentially dangerous characters.
    pub fn sanitize_string(input: &str) -> String {
        // Remove null bytes and control characters
        input
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
            .collect()
    }

    /// Sanitize a JSON value by sanitizing all string values.
    pub fn sanitize_json(value: &Value) -> Value {
        match value {
            Value::String(s) => Value::String(Self::sanitize_string(s)),
            Value::Array(arr) => Value::Array(arr.iter().map(Self::sanitize_json).collect()),
            Value::Object(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in obj {
                    map.insert(k.clone(), Self::sanitize_json(v));
                }
                Value::Object(map)
            }
            _ => value.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_request_valid() {
        let validator = InputValidator::new(ValidationConfig::default());
        let request = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });

        assert!(validator.validate_request(&request, 100).is_ok());
    }

    #[test]
    fn test_validate_request_too_large() {
        let validator = InputValidator::new(ValidationConfig::default());
        let request = json!({
            "model": "gpt-4",
            "messages": []
        });

        let result = validator.validate_request(&request, 20 * 1024 * 1024);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].code, ValidationErrorCode::RequestTooLarge);
    }

    #[test]
    fn test_validate_request_invalid_model_name() {
        let validator = InputValidator::new(ValidationConfig::default());
        let request = json!({
            "model": "gpt-4<script>",
            "messages": []
        });

        let result = validator.validate_request(&request, 100);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].code, ValidationErrorCode::InvalidModelName);
    }

    #[test]
    fn test_validate_request_too_many_messages() {
        let config = ValidationConfig {
            max_messages: 2,
            ..Default::default()
        };
        let validator = InputValidator::new(config);
        let request = json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "1"},
                {"role": "user", "content": "2"},
                {"role": "user", "content": "3"}
            ]
        });

        let result = validator.validate_request(&request, 100);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].code, ValidationErrorCode::TooManyMessages);
    }

    #[test]
    fn test_sanitize_string() {
        let input = "Hello\x00World\x01Test";
        let sanitized = InputValidator::sanitize_string(input);
        assert_eq!(sanitized, "HelloWorldTest");
    }

    #[test]
    fn test_sanitize_json() {
        let input = json!({
            "name": "Test\x00User",
            "messages": [
                {"content": "Hello\x01World"}
            ]
        });

        let sanitized = InputValidator::sanitize_json(&input);
        assert_eq!(sanitized["name"], "TestUser");
        assert_eq!(sanitized["messages"][0]["content"], "HelloWorld");
    }

    #[test]
    fn test_validate_model_name_patterns() {
        let validator = InputValidator::new(ValidationConfig::default());

        assert!(validator.is_valid_model_name("gpt-4"));
        assert!(validator.is_valid_model_name("openai/gpt-4"));
        assert!(validator.is_valid_model_name("claude-3-opus"));
        assert!(validator.is_valid_model_name("model_v2.1"));

        assert!(!validator.is_valid_model_name("model<script>"));
        assert!(!validator.is_valid_model_name("model; DROP TABLE"));
    }
}
