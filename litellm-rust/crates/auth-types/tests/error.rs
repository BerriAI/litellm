use litellm_auth_types::Error;
use rstest::rstest;

#[rstest]
#[case::api_key(
    Error::MissingApiKey { provider: "Example", environment_variable: "EXAMPLE_API_KEY" },
    "Missing Example API Key - Set `api_key` or the EXAMPLE_API_KEY environment variable"
)]
#[case::another_api_key(
    Error::MissingApiKey { provider: "Custom", environment_variable: "CUSTOM_KEY" },
    "Missing Custom API Key - Set `api_key` or the CUSTOM_KEY environment variable"
)]
#[case::api_base(
    Error::MissingApiBase { provider: "Example", guidance: "Pass api_base" },
    "Missing Example API Base - Pass api_base"
)]
#[case::another_api_base(
    Error::MissingApiBase { provider: "Custom", guidance: "Set CUSTOM_ENDPOINT" },
    "Missing Custom API Base - Set CUSTOM_ENDPOINT"
)]
#[case::configuration(
    Error::InvalidConfiguration("credential selector is invalid".into()),
    "invalid authentication configuration: credential selector is invalid"
)]
#[case::acquisition(
    Error::CredentialAcquisition("token expired".into()),
    "credential acquisition failed: token expired"
)]
#[case::caller(
    Error::EmptyCallerCredential("empty token"),
    "credential caller failed: empty token"
)]
#[case::provider(
    Error::ProviderAuthentication("provider rejected credentials".into()),
    "provider rejected credentials"
)]
#[case::chain(
    Error::CredentialChain(vec![
        Error::CredentialAcquisition("token expired".into()),
        Error::EmptyCallerCredential("empty token"),
    ]),
    "credential acquisition failed: credential acquisition failed: token expired; credential caller failed: empty token"
)]
fn display_preserves_failure_phase_and_caller_context(
    #[case] error: Error,
    #[case] expected: &str,
) {
    assert_eq!(error.to_string(), expected);
}

#[rstest]
#[case::configuration(true)]
#[case::acquisition(false)]
fn contextual_failures_keep_the_original_source(#[case] configuration: bool) {
    use litellm_auth_types::ErrorDetail;

    let detail = ErrorDetail::failed(
        "test credential",
        std::io::Error::from(std::io::ErrorKind::PermissionDenied),
    );
    let error = if configuration {
        Error::InvalidConfiguration(detail)
    } else {
        Error::CredentialAcquisition(detail)
    };
    let source = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
        error.source()
    })
    .find_map(|error| error.downcast_ref::<std::io::Error>())
    .expect("the original credential error remains available");
    assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
    assert!(error.to_string().contains("test credential failed:"));
    assert!(error.to_string().ends_with(&source.to_string()));
}
