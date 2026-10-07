use crate::{
    bedrock::messages::mantle::transformation::AmazonMantleMessagesConfig,
    bedrock_mantle::common_utils::BEDROCK_MANTLE_SETTINGS,
};

pub const BEDROCK_MANTLE_MESSAGES_CONFIG: AmazonMantleMessagesConfig = AmazonMantleMessagesConfig {
    settings: BEDROCK_MANTLE_SETTINGS,
};

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_auth::CredentialPlacement;
    use litellm_auth_aws::constants::{AWS_BEARER_TOKEN_BEDROCK, AWS_REGION, AWS_REGION_NAME};
    use rstest::rstest;

    use super::*;
    use crate::{
        base_llm::{auth::AuthScheme, messages::transformation::BaseMessagesConfig},
        bedrock::messages::mantle::transformation::mantle_host_region,
        bedrock_mantle::common_utils::{
            BEDROCK_MANTLE_API_KEY_ENV, BEDROCK_MANTLE_DEFAULT_REGION, BEDROCK_MANTLE_REGION_ENV,
        },
    };

    const CONFIG: AmazonMantleMessagesConfig = BEDROCK_MANTLE_MESSAGES_CONFIG;

    fn env_from(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn bearer(api_key: Option<&str>, env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
        match CONFIG
            .validate_environment(Vec::new(), api_key, "m", env)
            .unwrap()
            .auth
        {
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret,
            } => Some(secret.expose().to_string()),
            AuthScheme::AwsSigV4 { .. } => None,
            other => panic!("unexpected auth {other:?}"),
        }
    }

    fn sigv4_region(env: &dyn Fn(&str) -> Option<String>) -> String {
        match CONFIG
            .validate_environment(Vec::new(), None, "m", env)
            .unwrap()
            .auth
        {
            AuthScheme::AwsSigV4 { region, .. } => region,
            other => panic!("expected SigV4, got {other:?}"),
        }
    }

    #[rstest]
    #[case::explicit_key(
        Some("key"),
        &[(BEDROCK_MANTLE_API_KEY_ENV, "mantle"), (AWS_BEARER_TOKEN_BEDROCK, "bedrock")],
        Some("key"),
    )]
    #[case::mantle_key_outranks_bedrock_token(
        None,
        &[(BEDROCK_MANTLE_API_KEY_ENV, "mantle"), (AWS_BEARER_TOKEN_BEDROCK, "bedrock")],
        Some("mantle"),
    )]
    #[case::bedrock_token(None, &[(AWS_BEARER_TOKEN_BEDROCK, "bedrock")], Some("bedrock"))]
    #[case::blank_mantle_key_falls_through(
        None,
        &[(BEDROCK_MANTLE_API_KEY_ENV, " "), (AWS_BEARER_TOKEN_BEDROCK, "bedrock")],
        Some("bedrock"),
    )]
    #[case::nothing_signs(None, &[], None)]
    fn bearer_token_precedence(
        #[case] api_key: Option<&str>,
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(bearer(api_key, &env_from(env)).as_deref(), expected);
    }

    #[rstest]
    #[case::mantle_region(
        &[(BEDROCK_MANTLE_REGION_ENV, "eu-west-3"), (AWS_REGION_NAME, "eu-west-1"), (AWS_REGION, "ap-south-1")],
        "eu-west-3",
    )]
    #[case::region_name(&[(AWS_REGION_NAME, "eu-west-1"), (AWS_REGION, "ap-south-1")], "eu-west-1")]
    #[case::region(&[(AWS_REGION, "ap-south-1")], "ap-south-1")]
    #[case::blank_mantle_region_falls_through(&[(BEDROCK_MANTLE_REGION_ENV, ""), (AWS_REGION, "ap-south-1")], "ap-south-1")]
    #[case::default(&[], BEDROCK_MANTLE_DEFAULT_REGION)]
    fn region_precedence_for_the_url_and_the_signature(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let env = env_from(env);
        let url = CONFIG.get_complete_url(None, "m", &env).unwrap();
        assert_eq!(
            (mantle_host_region(&url), sigv4_region(&env)),
            (Some(expected.to_string()), expected.to_string())
        );
    }

    #[test]
    fn secret_names_cover_every_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = CONFIG.validate_environment(Vec::new(), None, "m", &record);
        let _ = CONFIG.get_complete_url(None, "m", &record);
        let requested = requested.into_inner();
        assert!(requested.contains(&BEDROCK_MANTLE_API_KEY_ENV.to_string()));
        assert!(requested.contains(&BEDROCK_MANTLE_REGION_ENV.to_string()));
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| !CONFIG.secret_names().contains(&name.as_str()))
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
}
