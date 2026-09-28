#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
    #[error("secret manager operation timed out")]
    Timeout,
    #[error("{0} environment variable is missing")]
    MissingEnvironment(&'static str),
    #[error("AZURE_KEY_VAULT_URI is not a valid https vault URL")]
    VaultUri,
    #[error("Azure Key Vault credentials are not configured")]
    MissingCredentials,
    #[error(transparent)]
    Auth(
        #[from]
        #[redact]
        litellm_auth_types::Error,
    ),
    #[error("Azure Key Vault request failed")]
    Http(
        #[source]
        #[redact]
        reqwest::Error,
    ),
    #[error("Azure Key Vault returned HTTP {0}")]
    Status(u16),
    #[error("Azure Key Vault response is missing the secret value")]
    MissingValue,
}
