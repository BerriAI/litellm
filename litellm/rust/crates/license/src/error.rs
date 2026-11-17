use crate::Feature;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("license public key is not a PEM encoded RSA public key")]
    PublicKey(#[source] rsa::pkcs8::spki::Error),
    #[error("license is not base64")]
    Encoding(#[source] base64::DecodeError),
    #[error("license has no signature")]
    Unsigned,
    #[error("license signature does not match its payload")]
    Signature(#[source] rsa::Error),
    #[error("license payload is not a JSON object")]
    Payload(#[source] serde_json::Error),
    #[error("license expiration_date is not a YYYY-MM-DD string")]
    ExpirationDate(#[source] Option<jiff::Error>),
    #[error("license expired")]
    Expired,
    #[error("license server request failed")]
    Server(#[source] litellm_http::Error),
    #[error("license server replied with status {0}")]
    Status(u16),
    #[error("license server reply has no boolean verify field")]
    Reply(#[source] Option<serde_json::Error>),
    #[error("A LiteLLM license with the '{0}' feature lifts the limit.")]
    MissingFeature(Feature),
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn missing_feature_names_the_license_spelling() {
        assert_eq!(
            Error::MissingFeature(Feature::AutoRouter).to_string(),
            "A LiteLLM license with the 'auto_router' feature lifts the limit."
        );
    }
}
