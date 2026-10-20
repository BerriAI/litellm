use aws_credential_types::provider::ProvideCredentials;
use litellm_auth_aws::AwsAuthConfig;
use litellm_cache_s3::S3Credentials;
use rstest::rstest;

fn explicit(session_token: Option<&str>) -> AwsAuthConfig {
    AwsAuthConfig {
        access_key_id: Some("key".to_string()),
        secret_access_key: Some("secret".to_string()),
        session_token: session_token.map(str::to_string),
        region_name: Some("us-east-1".to_string()),
        ..Default::default()
    }
}

fn ambient_token(name: &str) -> Option<String> {
    (name == "AWS_SESSION_TOKEN").then(|| "ambient".to_string())
}

fn environment_keys(name: &str) -> Option<String> {
    match name {
        "AWS_ACCESS_KEY_ID" => Some("env-key".to_string()),
        "AWS_SECRET_ACCESS_KEY" => Some("env-secret".to_string()),
        "AWS_SESSION_TOKEN" => Some("env-token".to_string()),
        _ => None,
    }
}

#[rstest]
#[case::explicit_keys_ignore_an_ambient_session_token(
    explicit(None), ambient_token, ("key", "secret", None)
)]
#[case::explicit_keys_keep_their_session_token(
    explicit(Some("t")), ambient_token, ("key", "secret", Some("t"))
)]
#[case::environment_keys_resolve_with_their_session_token(
    AwsAuthConfig::default(), environment_keys, ("env-key", "env-secret", Some("env-token"))
)]
#[tokio::test]
async fn credentials_resolve(
    #[case] config: AwsAuthConfig,
    #[case] env: fn(&str) -> Option<String>,
    #[case] expected: (&str, &str, Option<&str>),
) {
    let credentials = S3Credentials::with_env(config, env)
        .provide_credentials()
        .await
        .unwrap();
    assert_eq!(
        (
            credentials.access_key_id(),
            credentials.secret_access_key(),
            credentials.session_token(),
        ),
        expected
    );
}
