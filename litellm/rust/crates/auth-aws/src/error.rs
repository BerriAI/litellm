use aws_credential_types::provider::error::CredentialsError;
use litellm_error::Source;

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("AWS credential provider returned expired credentials")]
    ExpiredCredentials,
    #[error("could not load AWS {flow} credentials")]
    Credentials {
        flow: CredentialFlow,
        #[source]
        source: Source,
    },
    #[error(
        "the AWS web identity token must name a secret or an oidc/ provider, not an environment variable"
    )]
    WebIdentityTokenReference,
    #[error("the AWS web identity token reference resolved to no token")]
    MissingWebIdentityToken,
    #[error("could not read the AWS web identity token")]
    WebIdentityToken(#[source] Source),
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
    pub fn is_empty_default_chain(&self) -> bool {
        let Self::Credentials {
            flow: CredentialFlow::DefaultChain,
            source,
        } = self
        else {
            return false;
        };
        matches!(
            source.downcast_ref::<CredentialsError>(),
            Some(CredentialsError::CredentialsNotLoaded(_))
        )
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::empty_default_chain(
        CredentialFlow::DefaultChain,
        CredentialsError::not_loaded("no providers in chain provided credentials"),
        true
    )]
    #[case::failed_default_chain(
        CredentialFlow::DefaultChain,
        CredentialsError::provider_error("IMDS answered 500"),
        false
    )]
    #[case::empty_profile(
        CredentialFlow::Profile,
        CredentialsError::not_loaded("the profile has no credentials"),
        false
    )]
    fn only_a_default_chain_that_loaded_nothing_is_empty(
        #[case] flow: CredentialFlow,
        #[case] failure: CredentialsError,
        #[case] empty: bool,
    ) {
        assert_eq!(
            Error::credentials(flow)(failure).is_empty_default_chain(),
            empty
        );
    }

    #[test]
    fn a_web_identity_failure_is_not_an_empty_default_chain() {
        assert!(!Error::MissingWebIdentityToken.is_empty_default_chain());
    }
}
