use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum AuthError {
    #[error("invalid authentication configuration: credential cannot be empty")]
    EmptyCredential,
    #[error("invalid authentication configuration: credential is not a valid HTTP header value")]
    InvalidCredentialHeaderValue,
    #[error("invalid authentication configuration: invalid credential header name: {name}")]
    InvalidCredentialHeaderName { name: &'static str },
    #[error("invalid authentication configuration: credential header {name} already exists")]
    CredentialHeaderAlreadyExists { name: &'static str },
    #[error("invalid authentication configuration: credential header already exists")]
    ExistingCredentialHeader,
    #[error(
        "invalid authentication configuration: credential plan is not allowed by the provider auth policy"
    )]
    CredentialPlanNotAllowed,
    #[error(
        "invalid authentication configuration: caller credential plan requires provider-specific inputs"
    )]
    CallerPlanRequiresProviderInputs,
    #[error("credential caller failed: credential caller returned an empty credential")]
    EmptyCallerCredential,
    #[error("credential caller failed: Azure AD token provider returned an empty token")]
    EmptyAzureAdToken,
    #[error("credential acquisition failed: Azure OIDC reference did not resolve to a value")]
    UnresolvedOidcReference,
    #[error("invalid authentication configuration: invalid Azure credential selector")]
    InvalidAzureCredentialSelector,
    #[error(
        "invalid authentication configuration: ClientSecretCredential requires tenant_id, client_id, and client_secret"
    )]
    MissingClientSecretFields,
    #[error("invalid authentication configuration: WorkloadIdentityCredential requires tenant_id")]
    MissingWorkloadTenantId,
    #[error("invalid authentication configuration: WorkloadIdentityCredential requires client_id")]
    MissingWorkloadClientId,
    #[error(
        "invalid authentication configuration: WorkloadIdentityCredential requires azure_federated_token_file"
    )]
    MissingWorkloadTokenFile,
    #[error(
        "invalid authentication configuration: credential reference requires a host credential resolver"
    )]
    MissingCredentialResolver,
    #[error("invalid authentication configuration: unsupported OIDC reference")]
    UnsupportedOidcReference,
    #[error("invalid authentication configuration: OIDC environment reference cannot be empty")]
    EmptyOidcEnvironmentReference,
    #[error(
        "invalid authentication configuration: OIDC environment path reference cannot be empty"
    )]
    EmptyOidcEnvironmentPathReference,
    #[error("invalid authentication configuration: OIDC file reference cannot be empty")]
    EmptyOidcFileReference,
    #[error("invalid authentication configuration: {name} must be a string or null")]
    InvalidAzureConfigType { name: String },
    #[error(
        "invalid authentication configuration: Missing Azure AI credentials - set AZURE_AI_API_KEY or configure Entra ID"
    )]
    MissingAzureAiCredentials,
    #[error(
        "invalid authentication configuration: Missing Azure Document Intelligence credentials - set AZURE_DOCUMENT_INTELLIGENCE_API_KEY or configure Entra ID"
    )]
    MissingAzureDocumentIntelligenceCredentials,
    #[error(
        "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
    )]
    MissingReductoApiKey,

    #[error("invalid authentication configuration: {0}")]
    AzureCredentialConfiguration(String),
    #[error("credential acquisition failed: {0}")]
    Acquisition(String),
    #[error("credential caller failed: {0}")]
    Caller(String),
    #[error(
        "Missing {provider} API Key - A call is being made to {provider} but no key is set either in the environment variables or via params"
    )]
    MissingApiKey { provider: &'static str },
    #[error(
        "Missing {provider} API Base - Set {environment_variable} environment variable or pass api_base parameter"
    )]
    MissingApiBase {
        provider: &'static str,
        environment_variable: &'static str,
    },
    #[error("{0}")]
    Message(String),
}

#[derive(Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("{context} extra_headers.{name} must be a string, got {actual}")]
    InvalidHeaderType {
        context: &'static str,
        name: String,
        actual: &'static str,
    },
    #[error("invalid HTTP header name")]
    InvalidHeaderName,
    #[error("invalid value for HTTP header {name}")]
    InvalidHeaderValue { name: String },
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
    #[error(transparent)]
    Auth(#[from] AuthError),
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    /// The provider was never reached: DNS, TCP, TLS or proxy setup failed
    /// before any byte of the request went out. Nothing was billed, so a host
    /// that keeps a reference implementation can serve the request itself.
    /// A timeout is deliberately not this, since the provider may have received
    /// and answered the request already.
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
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
