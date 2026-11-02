#[derive(veil::Redact, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Connect(#[from] litellm_redis_client::ConnectError),
    #[error(
        "Redis setting credential_provider names a Python object, which this host cannot supply"
    )]
    NoHostCredentialProvider,
    #[error(
        "Redis credential setting {setting} must be set in the process environment, not only environment_variables"
    )]
    ProcessEnvironmentRequired { setting: &'static str },
    #[error("{stage} failed{}", http_status_suffix(*http_status))]
    Credential {
        stage: CredentialStage,
        http_status: Option<u16>,
        #[source]
        #[redact(with = "[REDACTED]")]
        cause: litellm_error::Source,
    },
    #[error("credential acquisition exceeded {timeout:?}")]
    CredentialTimeout {
        timeout: std::time::Duration,
        #[source]
        cause: tokio::time::error::Elapsed,
    },
    #[error("Redis credential has expired")]
    ExpiredCredential,
    #[error("no {provider} credential is available for Redis")]
    MissingCredential { provider: &'static str },
}

#[derive(Clone, Copy, Debug, strum::Display, strum::VariantArray)]
pub enum CredentialStage {
    #[strum(serialize = "caller identity")]
    Identity,
    #[strum(serialize = "Redis token signing")]
    Signing,
    #[strum(serialize = "Redis token exchange")]
    Exchange,
    #[strum(serialize = "Redis token response")]
    Response,
}

impl Error {
    pub(crate) fn credential<E: std::error::Error + Send + Sync + 'static>(
        stage: CredentialStage,
    ) -> impl FnOnce(E) -> Self {
        move |cause| Self::Credential {
            stage,
            http_status: None,
            cause: litellm_error::Source::new(cause),
        }
    }

    pub(crate) fn credential_response(
        stage: CredentialStage,
        status: u16,
    ) -> impl FnOnce(litellm_http::Error) -> Self {
        move |cause| Self::Credential {
            stage,
            http_status: Some(status),
            cause: litellm_error::Source::new(cause),
        }
    }

    pub(crate) fn upstream_status(&self) -> Option<u16> {
        match self {
            Self::Credential { http_status, .. } => *http_status,
            Self::Connect(_)
            | Self::NoHostCredentialProvider
            | Self::ProcessEnvironmentRequired { .. }
            | Self::CredentialTimeout { .. }
            | Self::ExpiredCredential
            | Self::MissingCredential { .. } => None,
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Connect(_) => "connection",
            Self::NoHostCredentialProvider => "host-credential-provider",
            Self::ProcessEnvironmentRequired { .. } => "process-environment-required",
            Self::Credential { stage, .. } => match stage {
                CredentialStage::Identity => "caller-identity",
                CredentialStage::Signing => "token-signing",
                CredentialStage::Exchange => "token-exchange",
                CredentialStage::Response => "token-response",
            },
            Self::CredentialTimeout { .. } => "acquisition-timeout",
            Self::ExpiredCredential => "expired-credential",
            Self::MissingCredential { .. } => "missing-credential",
        }
    }
}

fn http_status_suffix(status: Option<u16>) -> String {
    status.map_or_else(String::new, |status| format!(" (HTTP {status})"))
}
