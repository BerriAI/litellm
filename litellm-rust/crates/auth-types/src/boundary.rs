use crate::{InputSource, UntrustedCredentialContext};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CredentialBoundary {
    source: InputSource,
}

impl CredentialBoundary {
    pub fn new(sources: &[InputSource]) -> Result<Self, UntrustedCredentialContext> {
        if sources.is_empty() {
            return Err(UntrustedCredentialContext);
        }
        let request_owned = sources.contains(&InputSource::Request);
        if request_owned && sources.iter().any(|source| *source != InputSource::Request) {
            return Err(UntrustedCredentialContext);
        }
        let source = if request_owned {
            InputSource::Request
        } else if sources.contains(&InputSource::Deployment) {
            InputSource::Deployment
        } else {
            InputSource::Environment
        };
        Ok(Self { source })
    }

    pub fn host_only(sources: &[InputSource]) -> Result<Self, UntrustedCredentialContext> {
        let boundary = Self::new(sources)?;
        if boundary.source == InputSource::Request {
            return Err(UntrustedCredentialContext);
        }
        Ok(boundary)
    }

    pub fn source(&self) -> InputSource {
        self.source
    }

    pub fn validate_resources(
        &self,
        sources: impl IntoIterator<Item = InputSource>,
    ) -> Result<(), UntrustedCredentialContext> {
        if self.source != InputSource::Request
            && sources
                .into_iter()
                .any(|source| source == InputSource::Request)
        {
            return Err(UntrustedCredentialContext);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn credential_and_resource_source_combinations(
        #[values(
            InputSource::Request,
            InputSource::Deployment,
            InputSource::Environment
        )]
        identity: InputSource,
        #[values(
            InputSource::Request,
            InputSource::Deployment,
            InputSource::Environment
        )]
        secret: InputSource,
        #[values(
            InputSource::Request,
            InputSource::Deployment,
            InputSource::Environment
        )]
        resource: InputSource,
    ) {
        let result = CredentialBoundary::new(&[identity, secret])
            .and_then(|boundary| boundary.validate_resources([resource]));
        let request_owned = identity == InputSource::Request && secret == InputSource::Request;
        let host_owned = identity != InputSource::Request && secret != InputSource::Request;
        assert_eq!(
            result.is_ok(),
            request_owned || (host_owned && resource != InputSource::Request)
        );
    }

    #[rstest]
    #[case::environment(&[InputSource::Environment], Some(InputSource::Environment))]
    #[case::deployment(&[InputSource::Deployment, InputSource::Environment], Some(InputSource::Deployment))]
    #[case::request(&[InputSource::Request], None)]
    #[case::mixed(&[InputSource::Request, InputSource::Deployment], None)]
    fn host_credentials_require_host_sources(
        #[case] sources: &[InputSource],
        #[case] expected: Option<InputSource>,
    ) {
        assert_eq!(
            CredentialBoundary::host_only(sources)
                .ok()
                .map(|boundary| boundary.source()),
            expected
        );
    }

    #[rstest]
    fn an_empty_credential_bundle_cannot_authorize_resources() {
        assert_eq!(
            CredentialBoundary::new(&[]),
            Err(UntrustedCredentialContext)
        );
    }

    #[rstest]
    fn every_supplied_resource_is_checked_even_when_a_trusted_value_exists() {
        let trusted = crate::Sourced::new("configured", InputSource::Deployment);
        let supplied = crate::Sourced::new("override", InputSource::Request);
        let boundary = CredentialBoundary::new(&[InputSource::Environment]).unwrap();
        assert_eq!(
            boundary.validate_resources([trusted.source(), supplied.source()]),
            Err(UntrustedCredentialContext)
        );
        assert_eq!(
            boundary.validate_resources([supplied.source()]),
            Err(UntrustedCredentialContext)
        );
    }
}
