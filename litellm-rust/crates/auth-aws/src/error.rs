use litellm_error::Source;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("could not load AWS {flow} credentials")]
    Credentials {
        flow: CredentialFlow,
        #[source]
        source: Source,
    },
    #[error("AWS web identity response had no credentials")]
    MissingWebIdentityCredentials,
    #[error("AWS web identity response had an invalid expiration")]
    WebIdentityExpiration(#[source] Source),
    #[error("header {name} has a value that is not visible ASCII")]
    NonAsciiHeader { name: String },
    #[error("{stage}")]
    Signing {
        stage: SigningStage,
        #[source]
        source: Source,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CredentialFlow {
    #[error("profile")]
    Profile,
    #[error("default chain")]
    DefaultChain,
    #[error("assumed role")]
    AssumeRole,
    #[error("web identity")]
    WebIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SigningStage {
    #[error("SigV4 signing parameters are invalid")]
    Parameters,
    #[error("SigV4 cannot sign this request")]
    Request,
    #[error("SigV4 could not compute the signature")]
    Signature,
    #[error("SigV4 produced an invalid header")]
    Output,
}

impl Error {
    pub(crate) fn credentials<E: std::error::Error + Send + Sync + 'static>(
        flow: CredentialFlow,
    ) -> impl FnOnce(E) -> Self {
        move |source| Self::Credentials {
            flow,
            source: Source::new(source),
        }
    }

    pub(crate) fn signing<E: std::error::Error + Send + Sync + 'static>(
        stage: SigningStage,
    ) -> impl FnOnce(E) -> Self {
        move |source| Self::Signing {
            stage,
            source: Source::new(source),
        }
    }
}

impl From<Error> for litellm_auth_types::Error {
    fn from(error: Error) -> Self {
        Self::provider(error)
    }
}
