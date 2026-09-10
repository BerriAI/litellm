use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AuthError {
    #[error("invalid authentication configuration: {0}")]
    Configuration(#[from] AuthConfigurationError),
    #[error("credential acquisition failed: {0}")]
    AzureTokenAcquisition(String),
    #[error("credential acquisition failed: {}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    CredentialChain(Vec<AuthError>),
    #[error("credential caller failed: credential caller returned an empty credential")]
    EmptyCallerCredential,
    #[error("credential caller failed: Azure AD token provider returned an empty token")]
    EmptyAzureToken,
    #[error("credential acquisition failed: Azure OIDC reference did not resolve to a value")]
    UnresolvedOidcReference,
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
    MissingCredential(#[from] MissingCredential),
    #[error("{0}")]
    Aws(#[from] AwsAuthError),
    #[error("invalid authentication header")]
    InvalidHeader,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AuthConfigurationError {
    #[error("credential header already exists")]
    ExistingCredentialHeader,
    #[error("credential plan is not allowed by the provider auth policy")]
    DisallowedCredentialPlan,
    #[error("credential cannot be empty")]
    EmptyCredential,
    #[error("invalid Azure credential selector")]
    InvalidAzureSelector,
    #[error("ClientSecretCredential requires tenant_id, client_id, and client_secret")]
    MissingClientSecretFields,
    #[error("WorkloadIdentityCredential requires tenant_id")]
    MissingWorkloadTenant,
    #[error("WorkloadIdentityCredential requires client_id")]
    MissingWorkloadClient,
    #[error("WorkloadIdentityCredential requires azure_federated_token_file")]
    MissingWorkloadTokenFile,
    #[error("credential reference requires a host credential resolver")]
    MissingHostResolver,
    #[error("caller credential plan requires provider-specific inputs")]
    MissingCallerInputs,
    #[error("credential header {0} already exists")]
    DuplicateHeader(&'static str),
    #[error("{0} must be a string or null")]
    InvalidFieldType(String),
    #[error("unsupported OIDC reference")]
    UnsupportedOidcReference,
    #[error("{0} cannot be empty")]
    EmptyReference(String),
    #[error("Azure credential initialization failed: {0}")]
    AzureCredentialInitialization(String),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MissingCredential {
    #[error(
        "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY environment variable"
    )]
    AnthropicApiKey,
    #[error("Missing Azure API Key - Set `api_key` or the AZURE_API_KEY environment variable")]
    AzureApiKey,
    #[error(
        "Missing Azure API Base - Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic"
    )]
    AzureApiBase,
    #[error(
        "Missing OpenAI API Key - a realtime call is being made but no key was passed via params or the OPENAI_API_KEY environment variable"
    )]
    OpenAiRealtimeApiKey,
    #[error(
        "Missing OpenAI API Key - a Responses WebSocket call is being made but no key was passed via params or the OPENAI_API_KEY environment variable"
    )]
    OpenAiResponsesApiKey,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AwsAuthError {
    #[error("AWS profile credentials failed: {0}")]
    Profile(String),
    #[error("AWS default credentials failed: {0}")]
    DefaultChain(String),
    #[error("AWS role credentials failed: {0}")]
    AssumeRole(String),
    #[error("AWS web identity credentials failed: {0}")]
    WebIdentity(String),
    #[error("AWS web identity expiration was invalid: {0}")]
    WebIdentityExpiration(String),
    #[error("AWS signing parameters failed: {0}")]
    SigningParameters(String),
    #[error("AWS signable request failed: {0}")]
    SignableRequest(String),
    #[error("AWS request signing failed: {0}")]
    Signing(String),
    #[error("AWS web identity response had no credentials")]
    MissingWebIdentityCredentials,
}
