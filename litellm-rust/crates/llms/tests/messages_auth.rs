use litellm_auth::{
    AuthServices, CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_llms::base_llm::auth::{Authenticated, ValidatedEnvironment, resolve_auth};
use rstest::rstest;

const X_API_KEY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const REJECT_X_API_KEY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key"],
    existing_header_behavior: ExistingHeaderBehavior::Reject,
    scope: None,
    audience: None,
};

const BEARER: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Bearer,
    }],
    accepted_existing_headers: &["authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const ENTRA_ONLY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Entra,
        placement: CredentialPlacement::Bearer,
    }],
    accepted_existing_headers: &["authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const MISSING: litellm_auth::Error = litellm_auth::Error::MissingApiKey {
    provider: "Test",
    environment_variable: "TEST_API_KEY",
};

fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect()
}

async fn resolved(
    policy: &ProviderAuthPolicy,
    forwarded: &[(&str, &str)],
    api_key: Option<&str>,
) -> Result<Authenticated, litellm_auth::Error> {
    let validated = ValidatedEnvironment::with_api_key(
        policy,
        headers(forwarded),
        api_key.map(SecretValue::new),
        MISSING,
    )?;
    resolve_auth(&AuthServices::default(), validated, &|_| None).await
}

#[rstest]
#[case::header_placement(
    X_API_KEY,
    &[("content-type", "application/json")],
    Some("sk"),
    &[("content-type", "application/json"), ("x-api-key", "sk")]
)]
#[case::bearer_placement(BEARER, &[], Some("sk"), &[("authorization", "Bearer sk")])]
#[tokio::test]
async fn the_api_key_lands_in_the_static_placement(
    #[case] policy: ProviderAuthPolicy,
    #[case] forwarded: &[(&str, &str)],
    #[case] api_key: Option<&str>,
    #[case] expected: &[(&str, &str)],
) {
    let authenticated = resolved(&policy, forwarded, api_key).await.unwrap();
    assert_eq!(authenticated.headers, headers(expected));
}

#[rstest]
#[case::static_key_with_api_key(
    X_API_KEY,
    &[("X-Api-Key", "caller")],
    Some("sk-deploy"),
    &[("X-Api-Key", "caller")]
)]
#[case::static_key_without_api_key(X_API_KEY, &[("X-Api-Key", "caller")], None, &[("X-Api-Key", "caller")])]
#[case::entra_forwarded_without_static_rule(
    ENTRA_ONLY,
    &[("Authorization", "Bearer caller")],
    None,
    &[("Authorization", "Bearer caller")]
)]
#[tokio::test]
async fn an_existing_credential_header_is_forwarded(
    #[case] policy: ProviderAuthPolicy,
    #[case] forwarded: &[(&str, &str)],
    #[case] api_key: Option<&str>,
    #[case] expected: &[(&str, &str)],
) {
    let authenticated = resolved(&policy, forwarded, api_key).await.unwrap();
    assert_eq!(authenticated.headers, headers(expected));
}

#[rstest]
#[case::existing_header_with_key(Some("sk"))]
#[tokio::test]
async fn a_policy_that_rejects_existing_credential_headers_returns_an_error(
    #[case] api_key: Option<&str>,
) {
    let error = resolved(&REJECT_X_API_KEY, &[("x-api-key", "caller")], api_key)
        .await
        .unwrap_err();
    assert_eq!(
        error,
        litellm_auth::Error::InvalidConfiguration("credential header already exists".into())
    );
}

#[rstest]
#[case::missing_key(None, &[])]
#[case::empty_key(Some(""), &[])]
#[case::whitespace_key(Some(" \t"), &[])]
#[case::missing_key_with_unrelated_header(None, &[("x-trace", "1")])]
#[tokio::test]
async fn a_missing_or_blank_key_is_the_supplied_error(
    #[case] api_key: Option<&str>,
    #[case] forwarded: &[(&str, &str)],
) {
    let error = resolved(&X_API_KEY, forwarded, api_key).await.unwrap_err();
    assert_eq!(error, MISSING);
}

#[rstest]
#[case::with_key(Some("sk"))]
#[case::without_key(None)]
#[tokio::test]
async fn a_policy_without_a_static_rule_is_invalid_configuration(#[case] api_key: Option<&str>) {
    let error = resolved(&ENTRA_ONLY, &[], api_key).await.unwrap_err();
    assert_eq!(
        error,
        litellm_auth::Error::InvalidConfiguration(
            "the provider auth policy has no static credential rule".into()
        )
    );
}
