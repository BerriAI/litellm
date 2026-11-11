use litellm_core_utils::url_utils::ApiUrl;

use crate::{
    Error,
    azure::common_utils::{self, AzureCredential},
    base_llm::auth::{Headers, ValidatedEnvironment},
};

const ROUTE: [&str; 3] = ["openai", "v1", "decisions"];

pub fn complete_url(api_base: &str) -> Result<String, Error> {
    ApiUrl::parse(api_base)
        .and_then(|url| url.complete_path(&ROUTE))
        .map(|url| url.into_string())
        .map_err(|error| {
            Error::InvalidRequest(crate::ErrorDetail::invalid("Azure api_base", error))
        })
}

pub fn environment(headers: Headers, credential: AzureCredential) -> ValidatedEnvironment {
    common_utils::environment(headers, credential, false)
}

#[cfg(test)]
mod tests {
    use litellm_auth::{CredentialPlacement, SecretValue};
    use rstest::rstest;

    use super::*;
    use crate::base_llm::auth::AuthScheme;

    #[rstest]
    #[case::resource(
        "https://r.openai.azure.com",
        "https://r.openai.azure.com/openai/v1/decisions"
    )]
    #[case::trailing_slash(
        "https://r.openai.azure.com/",
        "https://r.openai.azure.com/openai/v1/decisions"
    )]
    #[case::openai_prefix(
        "https://r.openai.azure.com/openai",
        "https://r.openai.azure.com/openai/v1/decisions"
    )]
    #[case::complete(
        "https://r.openai.azure.com/openai/v1/decisions",
        "https://r.openai.azure.com/openai/v1/decisions"
    )]
    #[case::mock_prefix(
        "http://mock:9000/d/az-0",
        "http://mock:9000/d/az-0/openai/v1/decisions"
    )]
    #[case::query_kept(
        "https://r.openai.azure.com?x=1",
        "https://r.openai.azure.com/openai/v1/decisions?x=1"
    )]
    fn the_url_ends_in_the_v1_decisions_route(#[case] api_base: &str, #[case] expected: &str) {
        assert_eq!(complete_url(api_base).unwrap(), expected);
    }

    #[rstest]
    #[case::key(AzureCredential::ApiKey(SecretValue::new("k".to_owned())), CredentialPlacement::Header("api-key"))]
    #[case::token(AzureCredential::AdToken(SecretValue::new("t".to_owned())), CredentialPlacement::Bearer)]
    fn a_key_goes_in_api_key_and_a_token_is_a_bearer(
        #[case] credential: AzureCredential,
        #[case] expected: CredentialPlacement,
    ) {
        let validated = environment(Vec::new(), credential);

        assert!(
            matches!(validated.auth, AuthScheme::Credential { placement, .. } if placement == expected)
        );
    }
}
