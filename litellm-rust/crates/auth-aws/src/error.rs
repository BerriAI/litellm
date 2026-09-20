use thiserror::Error as ThisError;

#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum Error {
    #[error("AWS profile credentials failed: {0}")]
    AwsProfile(String),
    #[error("AWS default credentials failed: {0}")]
    AwsDefaultChain(String),
    #[error("AWS role credentials failed: {0}")]
    AwsAssumeRole(String),
    #[error("AWS web identity credentials failed: {0}")]
    AwsWebIdentity(String),
    #[error("AWS web identity expiration was invalid: {0}")]
    AwsWebIdentityExpiration(String),
    #[error("AWS signing parameters failed: {0}")]
    AwsSigningParameters(String),
    #[error("AWS signable request failed: {0}")]
    AwsSignableRequest(String),
    #[error("AWS request signing failed: {0}")]
    AwsSigning(String),
    #[error("AWS web identity response had no credentials")]
    AwsMissingWebIdentityCredentials,
}

impl From<Error> for litellm_auth::Error {
    fn from(error: Error) -> Self {
        Self::ProviderAuthentication(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn converts_to_shared_auth_error_without_losing_context() {
        let error = litellm_auth::Error::from(Error::AwsProfile("profile not found".into()));

        assert_eq!(
            error,
            litellm_auth::Error::ProviderAuthentication(
                "AWS profile credentials failed: profile not found".into()
            )
        );
    }
}
