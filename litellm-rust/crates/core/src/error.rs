use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Auth(String),
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("OCR request timed out")]
    Timeout,
    #[error("upstream network error: {0}")]
    Network(String),
    #[error("routing error: {0}")]
    Routing(String),
}

impl CoreError {
    pub fn unexpected_response_type(value: &serde_json::Value) -> Self {
        CoreError::InvalidResponse(format!(
            "expected object OCR response, got {}",
            json_type_name(value)
        ))
    }

    pub fn missing_response_field(field: &'static str) -> Self {
        CoreError::InvalidResponse(format!("OCR response missing required field: {field}"))
    }

    pub fn public_status_code(&self) -> Option<u16> {
        match self {
            CoreError::Http { status, .. } => Some(*status),
            CoreError::Auth(_) => Some(401),
            CoreError::InvalidType { .. }
            | CoreError::MissingField(_)
            | CoreError::InvalidProvider(_)
            | CoreError::InvalidRequest(_) => Some(400),
            CoreError::Timeout => Some(408),
            CoreError::InvalidResponse(_) | CoreError::Routing(_) => Some(500),
            CoreError::Network(_) => None,
        }
    }

    pub fn public_message(&self) -> String {
        match self {
            CoreError::Http { status, .. } => format!("OCR request failed with status {status}"),
            CoreError::Network(_) => "OCR request could not reach the provider".to_string(),
            CoreError::InvalidResponse(_) => {
                "OCR provider returned an invalid response".to_string()
            }
            CoreError::Routing(_) => "OCR request could not be routed".to_string(),
            CoreError::Auth(_) => "OCR request failed provider authentication".to_string(),
            CoreError::InvalidRequest(_) | CoreError::InvalidProvider(_) => {
                "Invalid OCR request".to_string()
            }
            CoreError::Timeout | CoreError::InvalidType { .. } | CoreError::MissingField(_) => {
                self.to_string()
            }
        }
    }
}

pub fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_status_code_preserves_public_contracts() {
        assert_eq!(
            CoreError::Http {
                status: 404,
                body: "not found".to_string()
            }
            .public_status_code(),
            Some(404)
        );
        assert_eq!(
            CoreError::Auth("bad key".to_string()).public_status_code(),
            Some(401)
        );
        assert_eq!(CoreError::Timeout.public_status_code(), Some(408));
        assert_eq!(
            CoreError::InvalidRequest("bad".to_string()).public_status_code(),
            Some(400)
        );
        assert_eq!(
            CoreError::MissingField("document.type").public_status_code(),
            Some(400)
        );
        assert_eq!(
            CoreError::InvalidType {
                expected: "object",
                actual: "string"
            }
            .public_status_code(),
            Some(400)
        );
        assert_eq!(
            CoreError::InvalidResponse("empty".to_string()).public_status_code(),
            Some(500)
        );
        assert_eq!(
            CoreError::Routing("no deployment".to_string()).public_status_code(),
            Some(500)
        );
        assert_eq!(
            CoreError::Network("dns".to_string()).public_status_code(),
            None
        );
        assert_eq!(
            CoreError::InvalidProvider("mistral".to_string()).public_status_code(),
            Some(400)
        );
    }

    #[test]
    fn timeout_message_is_data_minimized() {
        assert_eq!(CoreError::Timeout.to_string(), "OCR request timed out");
    }

    #[test]
    fn public_message_hides_upstream_body() {
        let err = CoreError::Http {
            status: 500,
            body: "signed-url=https://secret.example/token=abc123 leaked".to_string(),
        };
        let message = err.public_message();
        assert_eq!(message, "OCR request failed with status 500");
        assert!(!message.contains("secret"));
        assert!(!message.contains("abc123"));
    }

    #[test]
    fn public_message_hides_network_and_response_detail() {
        let network = CoreError::Network(
            "error sending request for url (https://signed.example/token=xyz)".to_string(),
        );
        assert_eq!(
            network.public_message(),
            "OCR request could not reach the provider"
        );
        assert!(!network.public_message().contains("token=xyz"));

        let invalid = CoreError::InvalidResponse("expected value at line 1 column 2".to_string());
        assert_eq!(
            invalid.public_message(),
            "OCR provider returned an invalid response"
        );
    }

    #[test]
    fn public_message_hides_auth_detail() {
        let err = CoreError::Auth(
            "google auth: failed to load service account key /secrets/sa.json token=ya29.abc123"
                .to_string(),
        );
        let message = err.public_message();
        assert_eq!(message, "OCR request failed provider authentication");
        assert!(!message.contains("sa.json"));
        assert!(!message.contains("ya29"));
        assert!(!message.contains("token"));
    }

    #[test]
    fn public_message_hides_invalid_request_detail() {
        let err = CoreError::InvalidRequest(
            "document_url=https://signed.example/doc.pdf?token=SECRET123 \
             base64=QUJDREVG header=x-api-key page=42"
                .to_string(),
        );
        let message = err.public_message();
        assert_eq!(message, "Invalid OCR request");
        assert!(!message.contains("SECRET123"));
        assert!(!message.contains("token"));
        assert!(!message.contains("base64"));
        assert!(!message.contains("QUJDREVG"));
        assert!(!message.contains("x-api-key"));
        assert!(!message.contains("signed.example"));
        assert_eq!(err.public_status_code(), Some(400));
    }

    #[test]
    fn public_message_keeps_only_static_detail() {
        assert_eq!(
            CoreError::MissingField("document.type").public_message(),
            "missing required field: document.type"
        );
        assert_eq!(
            CoreError::InvalidType {
                expected: "object",
                actual: "string"
            }
            .public_message(),
            "expected object, got string"
        );
        assert_eq!(
            CoreError::Routing("model_list parse failed".to_string()).public_message(),
            "OCR request could not be routed"
        );
    }

    #[test]
    fn malformed_response_maps_to_sanitized_500() {
        let wrong_type = CoreError::unexpected_response_type(&serde_json::json!("boom"));
        assert!(matches!(wrong_type, CoreError::InvalidResponse(_)));
        assert_eq!(wrong_type.public_status_code(), Some(500));
        assert_eq!(
            wrong_type.public_message(),
            "OCR provider returned an invalid response"
        );

        let missing = CoreError::missing_response_field("status");
        assert!(matches!(missing, CoreError::InvalidResponse(_)));
        assert_eq!(missing.public_status_code(), Some(500));
        assert_eq!(
            missing.public_message(),
            "OCR provider returned an invalid response"
        );
    }

    #[test]
    fn public_message_hides_invalid_provider_detail() {
        let err = CoreError::InvalidProvider(
            "no OCR provider 'internal-secret-router' registered for model 'gpt-8'".to_string(),
        );
        assert_eq!(err.public_message(), "Invalid OCR request");
        assert!(!err.public_message().contains("internal-secret-router"));
        assert_eq!(err.public_status_code(), Some(400));
    }
}
