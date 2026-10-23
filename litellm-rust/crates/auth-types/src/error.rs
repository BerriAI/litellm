use litellm_error::{Report, Source};

use crate::{ParamKind, ParamSpec};

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid authentication configuration")]
    InvalidConfiguration(#[source] ErrorDetail),
    #[error("credential acquisition failed")]
    CredentialAcquisition(#[source] ErrorDetail),
    #[error("{caller} returned an empty credential")]
    EmptyCallerCredential { caller: &'static str },
    #[error("provider authentication failed")]
    Provider(#[source] Source),
    #[error("no credential in the chain succeeded: {}", reports(.0))]
    CredentialChain(Vec<Error>),
    #[error(
        "Missing {provider} API Key - Set `api_key` or the {environment_variable} environment variable"
    )]
    MissingApiKey {
        provider: &'static str,
        environment_variable: &'static str,
    },
    #[error("Missing {provider} {} - {}", .spec.setting, .spec.guidance())]
    MissingParam {
        provider: &'static str,
        spec: &'static ParamSpec,
    },
    #[error("Missing {provider} API Base - {guidance}")]
    MissingApiBase {
        provider: &'static str,
        guidance: &'static str,
    },
    #[error("invalid authentication header")]
    InvalidHeader,
}

fn reports(failures: &[Error]) -> String {
    failures
        .iter()
        .map(|failure| Report::new(failure).to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

impl Error {
    pub fn provider(cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Provider(Source::new(cause))
    }

    pub fn is_missing_credential(&self) -> bool {
        match self {
            Self::MissingApiKey { .. } => true,
            Self::MissingParam { spec, .. } => match spec.kind {
                ParamKind::Credential => true,
                ParamKind::Setting => false,
            },
            Self::InvalidConfiguration(_)
            | Self::CredentialAcquisition(_)
            | Self::EmptyCallerCredential { .. }
            | Self::Provider(_)
            | Self::CredentialChain(_)
            | Self::MissingApiBase { .. }
            | Self::InvalidHeader => false,
        }
    }

    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidConfiguration(_)
            | Self::CredentialAcquisition(_)
            | Self::EmptyCallerCredential { .. }
            | Self::Provider(_)
            | Self::CredentialChain(_)
            | Self::InvalidHeader => true,
            Self::MissingParam { spec, .. } => match spec.kind {
                ParamKind::Setting => true,
                ParamKind::Credential => false,
            },
            Self::MissingApiKey { .. } | Self::MissingApiBase { .. } => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ErrorDetail {
    #[error("{field} must be {expected}")]
    InvalidType {
        field: String,
        expected: &'static str,
    },
    #[error("{subject} cannot be empty")]
    Empty { subject: &'static str },
    #[error("credential header {0} already exists")]
    DuplicateHeader(&'static str),
    #[error("credential plan is not allowed by the provider auth policy")]
    PlanNotAllowed,
    #[error("{credential} requires {settings}")]
    MissingSettings {
        credential: &'static str,
        settings: &'static str,
    },
    #[error("credentials cannot be sent to a request-controlled {provider} endpoint")]
    RequestControlledEndpoint { provider: &'static str },
    #[error("request-controlled {provider} credential references are not allowed")]
    RequestControlledReference { provider: &'static str },
    #[error("request-controlled {provider} auth inputs cannot be combined with host credentials")]
    MixedInputSources { provider: &'static str },
    #[error("unsupported OIDC reference")]
    UnsupportedReference,
    #[error("credential reference requires a host credential resolver")]
    MissingResolver,
    #[error("{subject} did not resolve to a value")]
    Unresolved { subject: &'static str },
    #[error(
        "The `{field}` value is base64-encoded JSON, not the JSON itself. Set it to a readable file path, or to the credentials JSON itself."
    )]
    Base64EncodedJson { field: &'static str },
    #[error("{operation} failed")]
    Failed {
        operation: &'static str,
        #[source]
        cause: Source,
    },
}

impl ErrorDetail {
    pub fn failed(
        operation: &'static str,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Failed {
            operation,
            cause: Source::new(cause),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use litellm_error::{Report, Source};
    use rstest::rstest;

    use super::{Error, ErrorDetail};
    use crate::{AwsParams, VertexParams};

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
    fn missing_inputs_keep_the_python_text(#[case] error: Error, #[case] expected: &str) {
        assert_eq!(error.to_string(), expected);
    }

    #[rstest]
    #[case::configuration(
        Error::InvalidConfiguration(ErrorDetail::Empty { subject: "credential selector" }),
        "invalid authentication configuration: credential selector cannot be empty"
    )]
    #[case::acquisition(
        Error::CredentialAcquisition(ErrorDetail::Unresolved { subject: "OIDC reference" }),
        "credential acquisition failed: OIDC reference did not resolve to a value"
    )]
    #[case::empty_caller(
        Error::EmptyCallerCredential { caller: "token provider" },
        "token provider returned an empty credential"
    )]
    #[case::chain(
        Error::CredentialChain(vec![
            Error::CredentialAcquisition(ErrorDetail::Unresolved { subject: "OIDC reference" }),
            Error::EmptyCallerCredential { caller: "token provider" },
        ]),
        "no credential in the chain succeeded: credential acquisition failed: OIDC reference did not resolve \
         to a value; token provider returned an empty credential"
    )]
    fn the_report_names_every_layer_in_order(#[case] error: Error, #[case] expected: &str) {
        assert!(
            Report::new(&error).to_string().contains(expected),
            "{}",
            Report::new(&error)
        );
    }

    #[rstest]
    #[case::configuration(Error::InvalidConfiguration(ErrorDetail::failed(
        "test credential",
        denied()
    )))]
    #[case::acquisition(Error::CredentialAcquisition(ErrorDetail::failed(
        "test credential",
        denied()
    )))]
    fn a_failure_keeps_its_cause_as_the_source_and_renders_it_once(#[case] error: Error) {
        let cause =
            std::iter::successors(std::error::Error::source(&error), |cause| cause.source())
                .find_map(|cause| cause.downcast_ref::<Source>()?.downcast_ref::<io::Error>())
                .expect("the io cause is in the source chain");
        assert_eq!(cause.kind(), io::ErrorKind::PermissionDenied);
        let report = Report::new(&error).to_string();
        assert_eq!(report.matches("access denied").count(), 1, "{report}");
    }

    #[rstest]
    #[case::configuration(Error::InvalidConfiguration(ErrorDetail::failed(
        "test credential",
        denied()
    )))]
    #[case::acquisition(Error::CredentialAcquisition(ErrorDetail::failed(
        "test credential",
        denied()
    )))]
    fn a_contextual_failure_leaves_its_cause_out_of_its_own_message(#[case] error: Error) {
        assert!(!error.to_string().contains("access denied"), "{error}");
        assert!(
            Report::new(&error)
                .to_string()
                .contains("test credential failed: access denied")
        );
    }

    #[test]
    fn a_provider_failure_keeps_the_provider_error_as_its_source() {
        let error = Error::provider(denied());
        assert_eq!(
            litellm_error::find::<io::Error>(&error).map(io::Error::kind),
            Some(io::ErrorKind::PermissionDenied)
        );
        assert_eq!(
            Report::new(&error).to_string(),
            "provider authentication failed: access denied"
        );
    }

    fn denied() -> io::Error {
        io::Error::new(io::ErrorKind::PermissionDenied, "access denied")
    }

    #[rstest]
    #[case::api_key(Error::MissingApiKey { provider: "Example", environment_variable: "EXAMPLE_API_KEY" }, true)]
    #[case::vertex_credentials(VertexParams::CREDENTIALS.missing("Vertex AI"), true)]
    #[case::aws_secret_access_key(AwsParams::SECRET_ACCESS_KEY.missing("AWS"), true)]
    #[case::vertex_project(VertexParams::PROJECT.missing("Vertex AI"), false)]
    #[case::aws_region(AwsParams::REGION.missing("AWS"), false)]
    #[case::api_base(Error::MissingApiBase { provider: "Example", guidance: "Pass api_base" }, false)]
    #[case::invalid_configuration(Error::InvalidConfiguration(ErrorDetail::PlanNotAllowed), false)]
    fn only_an_absent_credential_is_a_missing_credential(
        #[case] error: Error,
        #[case] expected: bool,
    ) {
        assert_eq!(error.is_missing_credential(), expected);
    }

    #[rstest]
    #[case::vertex_project(VertexParams::PROJECT.missing("Vertex AI"), true)]
    #[case::aws_region(AwsParams::REGION.missing("AWS"), true)]
    #[case::api_base(Error::MissingApiBase { provider: "Example", guidance: "Pass api_base" }, false)]
    #[case::vertex_credentials(VertexParams::CREDENTIALS.missing("Vertex AI"), false)]
    #[case::api_key(Error::MissingApiKey { provider: "Example", environment_variable: "EXAMPLE_API_KEY" }, false)]
    #[case::invalid_configuration(Error::InvalidConfiguration(ErrorDetail::PlanNotAllowed), true)]
    #[case::invalid_header(Error::InvalidHeader, true)]
    fn a_setting_the_caller_can_pass_is_theirs_to_fix_but_credentials_and_api_bases_are_not(
        #[case] error: Error,
        #[case] expected: bool,
    ) {
        assert_eq!(error.is_request(), expected);
    }
}
