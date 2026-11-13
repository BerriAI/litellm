use rustify::errors::ClientError as RustifyClientError;
use vaultrs::error::ClientError;

#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error("HashiCorp Vault requires an enterprise license")]
    EnterpriseRequired,
    #[error("invalid secret name")]
    InvalidSecretName(#[source] litellm_secrets_types::Error),
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
    #[error("HashiCorp Vault client failed")]
    Client(
        #[source]
        #[redact]
        vaultrs::error::ClientError,
    ),
    #[error("HashiCorp Vault request failed")]
    Transport(
        #[source]
        #[redact]
        litellm_http::Error,
    ),
    #[error("HashiCorp Vault client settings are invalid")]
    ClientSettings(#[source] vaultrs::client::VaultClientSettingsBuilderError),
    #[error("HashiCorp Vault TLS identity could not be configured for {path}")]
    TlsIdentity {
        path: std::path::PathBuf,
        source: litellm_error::Source,
    },
    #[error("HashiCorp Vault login returned HTTP {status}")]
    LoginStatus { status: u16 },
    #[error("HashiCorp Vault login response is malformed")]
    MalformedLogin(
        #[source]
        #[redact]
        vaultrs::error::ClientError,
    ),
    #[error("HashiCorp Vault authentication is not configured")]
    NoAuthConfigured,
    #[error("HashiCorp Vault returned HTTP {status}")]
    Status { status: u16 },
    #[error("HashiCorp Vault response payload is malformed")]
    MalformedPayload(
        #[source]
        #[redact]
        vaultrs::error::ClientError,
    ),
    #[error("HashiCorp Vault secret value is not a string")]
    NonStringValue,
    #[error("HashiCorp Vault data key conflicts with description")]
    DataKeyConflictsWithDescription,
    #[error("HashiCorp Vault secret version exceeds CAS range")]
    CasVersionOverflow(#[source] std::num::TryFromIntError),
    #[error("HashiCorp Vault operation timed out")]
    Timeout,
    #[error("invalid HashiCorp Vault refresh interval")]
    RefreshInterval(#[source] Option<std::num::ParseIntError>),
}

impl Error {
    pub fn login_status(&self) -> Option<u16> {
        match self {
            Self::LoginStatus { status } => Some(*status),
            Self::EnterpriseRequired
            | Self::InvalidSecretName(_)
            | Self::Operation(_)
            | Self::Client(_)
            | Self::Transport(_)
            | Self::ClientSettings(_)
            | Self::TlsIdentity { .. }
            | Self::MalformedLogin(_)
            | Self::NoAuthConfigured
            | Self::Status { .. }
            | Self::MalformedPayload(_)
            | Self::NonStringValue
            | Self::DataKeyConflictsWithDescription
            | Self::CasVersionOverflow(_)
            | Self::Timeout
            | Self::RefreshInterval(_) => None,
        }
    }

    pub fn is_unsafe_secret_name(&self) -> bool {
        match self {
            Self::InvalidSecretName(error) | Self::Operation(error) => {
                error.is_unsafe_secret_name()
            }
            Self::EnterpriseRequired
            | Self::Client(_)
            | Self::Transport(_)
            | Self::ClientSettings(_)
            | Self::TlsIdentity { .. }
            | Self::LoginStatus { .. }
            | Self::MalformedLogin(_)
            | Self::NoAuthConfigured
            | Self::Status { .. }
            | Self::MalformedPayload(_)
            | Self::NonStringValue
            | Self::DataKeyConflictsWithDescription
            | Self::CasVersionOverflow(_)
            | Self::Timeout
            | Self::RefreshInterval(_) => false,
        }
    }
}

impl From<ClientError> for Error {
    fn from(error: ClientError) -> Self {
        match error {
            ClientError::RestClientError { source } => match transport(source) {
                Ok(error) => Self::Transport(error),
                Err(source) => Self::Client(ClientError::RestClientError { source }),
            },
            ClientError::RestClientBuildError { source } => Self::Transport(source.into()),
            error => Self::Client(error),
        }
    }
}

pub(crate) fn transport(
    error: RustifyClientError,
) -> Result<litellm_http::Error, RustifyClientError> {
    match error {
        RustifyClientError::RequestError {
            source,
            url,
            method,
        } => source
            .downcast::<reqwest::Error>()
            .map(litellm_http::Error::from)
            .map_err(|source| RustifyClientError::RequestError {
                source,
                url,
                method,
            }),
        RustifyClientError::ResponseError { source } => source
            .downcast::<reqwest::Error>()
            .map(litellm_http::Error::from)
            .map_err(|source| RustifyClientError::ResponseError { source }),
        RustifyClientError::ReqwestBuildError { source } => Ok(source.into()),
        error => Err(error),
    }
}
