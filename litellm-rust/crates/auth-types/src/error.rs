use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("invalid authentication configuration: credential header already exists")]
    ExistingCredentialHeader,
    #[error(
        "invalid authentication configuration: credential plan is not allowed by the provider auth policy"
    )]
    DisallowedCredentialPlan,
    #[error("invalid authentication configuration: credential cannot be empty")]
    EmptyCredential,
    #[error("invalid authentication configuration: invalid Azure credential selector")]
    InvalidAzureSelector,
    #[error(
        "invalid authentication configuration: ClientSecretCredential requires tenant_id, client_id, and client_secret"
    )]
    MissingClientSecretFields,
    #[error("invalid authentication configuration: WorkloadIdentityCredential requires tenant_id")]
    MissingWorkloadTenant,
    #[error("invalid authentication configuration: WorkloadIdentityCredential requires client_id")]
    MissingWorkloadClient,
    #[error(
        "invalid authentication configuration: WorkloadIdentityCredential requires azure_federated_token_file"
    )]
    MissingWorkloadTokenFile,
    #[error(
        "invalid authentication configuration: credential reference requires a host credential resolver"
    )]
    MissingHostResolver,
    #[error(
        "invalid authentication configuration: caller credential plan requires provider-specific inputs"
    )]
    MissingCallerInputs,
    #[error("invalid authentication configuration: credential header {0} already exists")]
    DuplicateHeader(&'static str),
    #[error("invalid authentication configuration: {0} must be a string or null")]
    InvalidFieldType(String),
    #[error("invalid authentication configuration: unsupported OIDC reference")]
    UnsupportedOidcReference,
    #[error("invalid authentication configuration: {0} cannot be empty")]
    EmptyReference(String),
    #[error("invalid authentication configuration: Azure credential initialization failed: {0}")]
    AzureCredentialInitialization(String),
    #[error(
        "invalid authentication configuration: Azure authority must be an HTTPS origin without credentials, query, or fragment"
    )]
    InvalidAzureAuthority,
    #[error(
        "invalid authentication configuration: request-controlled Azure auth inputs cannot be combined with host credentials"
    )]
    MixedAzureCredentialSources,
    #[error(
        "invalid authentication configuration: request-controlled Azure credential references are not allowed"
    )]
    RequestAzureCredentialReference,
    #[error(
        "invalid authentication configuration: host credentials cannot be sent to a request-controlled Azure endpoint"
    )]
    RequestAzureCredentialDestination,
    #[error(
        "invalid authentication configuration: credentials cannot be sent to a request-controlled Vertex AI endpoint"
    )]
    RequestVertexCredentialDestination,
    #[error(
        "invalid authentication configuration: request-controlled Vertex credentials must use the canonical Google OAuth token endpoint"
    )]
    RequestVertexTokenEndpoint,
    #[error("credential acquisition failed: {0}")]
    AzureTokenAcquisition(String),
    #[error("credential acquisition failed: Vertex AI credentials: {0}")]
    VertexTokenAcquisition(String),
    #[error("{0}")]
    ProviderAuthentication(String),
    #[error("credential acquisition failed: {}", .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    CredentialChain(Vec<Error>),
    #[error("credential caller failed: credential caller returned an empty credential")]
    EmptyCallerCredential,
    #[error("credential caller failed: Azure AD token provider returned an empty token")]
    EmptyAzureToken,
    #[error("credential acquisition failed: Azure OIDC reference did not resolve to a value")]
    UnresolvedOidcReference,
    #[error(
        "Missing {provider} API Key - Set `api_key` or the {environment_variable} environment variable"
    )]
    MissingApiKey {
        provider: &'static str,
        environment_variable: &'static str,
    },
    #[error(
        "Missing {provider} API Base - Set {environment_variable} environment variable or pass api_base parameter"
    )]
    MissingApiBase {
        provider: &'static str,
        environment_variable: &'static str,
    },
    #[error(
        "Missing Azure API Base - Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic"
    )]
    MissingAzureApiBase,
    #[error("invalid authentication header")]
    InvalidHeader,
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn missing_api_key_names_provider_and_environment_variable() {
        assert_eq!(
            Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            }
            .to_string(),
            "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY environment variable"
        );
    }
}
