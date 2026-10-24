use crate::Error;
use litellm_auth::{CredentialBoundary, InputSource};
use litellm_core_utils::settings::resolve_non_empty;

pub const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
pub const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";

pub(crate) fn validate_destination(
    endpoint_source: Option<InputSource>,
    credential_source: InputSource,
) -> Result<(), litellm_auth::Error> {
    CredentialBoundary::new(&[credential_source])
        .and_then(|boundary| boundary.validate_resources(endpoint_source))
        .map_err(|_| {
            litellm_auth::Error::InvalidConfiguration(
                "host credentials cannot be sent to a request-controlled Azure endpoint".into(),
            )
        })
}

pub fn resolve_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_non_empty(api_key, env_lookup, &[AZURE_API_KEY_ENV]).ok_or_else(|| {
        Error::from(litellm_auth::Error::MissingApiKey {
            provider: "Azure",
            environment_variable: AZURE_API_KEY_ENV,
        })
    })
}

pub fn resolve_azure_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    resolve_non_empty(api_base, env_lookup, &[AZURE_API_BASE_ENV])
        .ok_or_else(|| Error::from(litellm_auth::Error::MissingApiBase { provider: "Azure", guidance: "Set `api_base` or the AZURE_API_BASE environment variable. Expected format: https://<resource-name>.services.ai.azure.com/anthropic" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn destination_source_combinations(
        #[values(
            InputSource::Request,
            InputSource::Deployment,
            InputSource::Environment
        )]
        credential: InputSource,
        #[values(
            None,
            Some(InputSource::Request),
            Some(InputSource::Deployment),
            Some(InputSource::Environment)
        )]
        endpoint: Option<InputSource>,
    ) {
        let result = validate_destination(endpoint, credential);
        if endpoint == Some(InputSource::Request) && credential != InputSource::Request {
            assert_eq!(
                result,
                Err(litellm_auth::Error::InvalidConfiguration(
                    "host credentials cannot be sent to a request-controlled Azure endpoint".into(),
                ))
            );
        } else {
            assert_eq!(result, Ok(()));
        }
    }
}
