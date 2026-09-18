use serde::Serialize;

/// The public LiteLLM classes built from a status code alone: every one takes the same
/// constructor arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, strum::EnumIter, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum StatusClass {
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    RateLimit,
    ContextWindowExceeded,
    ContentPolicyViolation,
    InternalServer,
    BadGateway,
    ServiceUnavailable,
    UnsupportedParams,
}

impl StatusClass {
    /// The `status_code` the Python class sets on itself.
    pub const fn status_code(self) -> u16 {
        match self {
            Self::BadRequest
            | Self::ContextWindowExceeded
            | Self::ContentPolicyViolation
            | Self::UnsupportedParams => 400,
            Self::Authentication => 401,
            Self::PermissionDenied => 403,
            Self::NotFound => 404,
            Self::RateLimit => 429,
            Self::InternalServer => 500,
            Self::BadGateway => 502,
            Self::ServiceUnavailable => 503,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct UpstreamResponse {
    pub status: u16,
    pub body: String,
    pub headers: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HttpStub {
    pub status: u16,
    pub method: &'static str,
    pub url: &'static str,
    pub content: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseArg {
    Upstream(UpstreamResponse),
    Stub(HttpStub),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PublicKind {
    Status {
        status_class: StatusClass,
        response: Option<ResponseArg>,
    },
    Timeout {
        status: Option<u16>,
    },
    ApiConnection,
    Api {
        status: u16,
        request_url: &'static str,
    },
}

/// Constructor arguments for the public LiteLLM exception, as `exception_type` passes them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PublicFailure {
    pub kind: PublicKind,
    pub message: String,
    pub model: String,
    pub llm_provider: Option<String>,
    pub litellm_debug_info: Option<String>,
    pub litellm_response_headers: Option<Vec<(String, String)>>,
    pub print_banner: bool,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use serde_json::Value;
    use strum::IntoEnumIterator;

    use super::*;

    const REGENERATE: &str = "LITELLM_REGENERATE_PUBLIC_FAILURE_FIXTURES";

    fn fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/test_litellm/rust_bridge/fixtures/public_failures")
    }

    fn upstream(status: u16) -> ResponseArg {
        ResponseArg::Upstream(UpstreamResponse {
            status,
            body: r#"{"message": "rejected"}"#.into(),
            headers: vec![("retry-after".into(), "7".into())],
        })
    }

    fn status_response(class: StatusClass) -> Option<ResponseArg> {
        match class {
            StatusClass::Authentication => None,
            StatusClass::PermissionDenied => Some(ResponseArg::Stub(HttpStub {
                status: 403,
                method: "POST",
                url: " https://cloud.google.com/vertex-ai/",
                content: None,
            })),
            StatusClass::InternalServer => Some(ResponseArg::Stub(HttpStub {
                status: 500,
                method: "completion",
                url: "https://github.com/BerriAI/litellm",
                content: Some("upstream text".into()),
            })),
            class => Some(upstream(class.status_code())),
        }
    }

    fn failure(kind: PublicKind, name: &str) -> PublicFailure {
        let headers = matches!(
            &kind,
            PublicKind::Status {
                response: Some(ResponseArg::Upstream(_)),
                ..
            }
        );
        PublicFailure {
            kind,
            message: format!("MistralException - {name}"),
            model: "ocr-model".into(),
            llm_provider: Some("mistral".into()),
            litellm_debug_info: Some("\nModel: ocr-model".into()),
            litellm_response_headers: headers.then(|| vec![("retry-after".into(), "7".into())]),
            print_banner: false,
        }
    }

    /// One payload per public class the constructor can build; `test_failures.py` reads the
    /// same files, so a shape change on either side fails there or here.
    fn fixtures() -> Vec<(String, PublicFailure)> {
        let statuses = StatusClass::iter().map(|class| {
            let name: &'static str = class.into();
            let name = format!("status_{name}");
            let built = failure(
                PublicKind::Status {
                    status_class: class,
                    response: status_response(class),
                },
                &name,
            );
            (name, built)
        });
        let others = [
            (
                "timeout_with_status",
                PublicFailure {
                    print_banner: true,
                    ..failure(
                        PublicKind::Timeout { status: Some(504) },
                        "timeout_with_status",
                    )
                },
            ),
            (
                "timeout_without_status",
                PublicFailure {
                    litellm_debug_info: None,
                    ..failure(
                        PublicKind::Timeout { status: None },
                        "timeout_without_status",
                    )
                },
            ),
            (
                "api_connection",
                PublicFailure {
                    llm_provider: None,
                    ..failure(PublicKind::ApiConnection, "api_connection")
                },
            ),
            (
                "api",
                failure(
                    PublicKind::Api {
                        status: 409,
                        request_url: "https://docs.litellm.ai/docs",
                    },
                    "api",
                ),
            ),
        ]
        .map(|(name, built)| (name.to_string(), built));
        statuses.chain(others).collect()
    }

    #[test]
    fn serialized_payloads_match_the_golden_fixtures_python_reads() {
        let directory = fixture_directory();
        let regenerate = std::env::var_os(REGENERATE).is_some();
        let expected = fixtures();
        for (name, built) in &expected {
            let path = directory.join(format!("{name}.json"));
            let serialized = serde_json::to_value(built).unwrap();
            if regenerate {
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(
                    &path,
                    format!("{}\n", serde_json::to_string_pretty(&serialized).unwrap()),
                )
                .unwrap();
            }
            let golden: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(serialized, golden, "{name}; set {REGENERATE}=1 to rewrite");
        }
        let on_disk: BTreeSet<String> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let generated: BTreeSet<String> = expected
            .iter()
            .map(|(name, _)| format!("{name}.json"))
            .collect();
        assert_eq!(on_disk, generated);
    }

    #[rstest::rstest]
    #[case(StatusClass::BadRequest, 400)]
    #[case(StatusClass::Authentication, 401)]
    #[case(StatusClass::PermissionDenied, 403)]
    #[case(StatusClass::NotFound, 404)]
    #[case(StatusClass::RateLimit, 429)]
    #[case(StatusClass::ContextWindowExceeded, 400)]
    #[case(StatusClass::ContentPolicyViolation, 400)]
    #[case(StatusClass::InternalServer, 500)]
    #[case(StatusClass::BadGateway, 502)]
    #[case(StatusClass::ServiceUnavailable, 503)]
    #[case(StatusClass::UnsupportedParams, 400)]
    fn status_codes_are_the_ones_the_python_classes_set(
        #[case] class: StatusClass,
        #[case] status: u16,
    ) {
        assert_eq!(class.status_code(), status);
    }
}
