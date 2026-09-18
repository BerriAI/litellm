//! A port of Python's `exception_type` for the routes that run in Rust.
//!
//! KNOWN_GAPS: differences from the Python mapper that no Rust route can reach today. Each
//! one stops being acceptable at its trigger.
//! - The Vertex partner-model API base for "claude" models is not built into
//!   `extra_information`. Trigger: a Vertex route whose models include Anthropic partner
//!   models; then `api_base` gets that branch and a table row.
//! - Python reports the provider `get_llm_provider` resolves for a stripped model name when
//!   that name happens to be in the model cost map. Trigger: a route whose model names
//!   overlap the cost map; that needs the provider resolution port, not a classifier change.
//! - The generic `APIConnectionError` fallback appends `traceback.format_exc()` to the
//!   message. Rust has no Python traceback and does not invent one; a sweep row that reaches
//!   it compares the message before the traceback.

use super::secret_redaction::{redact_string, secret_redaction_enabled};

mod cohere;
mod openai;
mod original;
mod public;
mod rules;
mod status;
mod vertex_ai;

pub use original::{ExceptionFamily, LocalClass, OriginalException};
pub use public::{HttpStub, PublicFailure, PublicKind, ResponseArg, StatusClass, UpstreamResponse};

const DOCS_URL: &str = "https://docs.litellm.ai/docs";

const TIMEOUT_MARKERS: &[&str] = &[
    "Request Timeout Error",
    "Request timed out",
    "Timed out generating response",
    "The read operation timed out",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExceptionContext {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub family: ExceptionFamily,
    pub asynchronous: bool,
    pub suppress_debug_info: bool,
    pub redact_messages_in_exceptions: bool,
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
    pub model_group: Option<String>,
    pub deployment: Option<String>,
    pub user_api_key_alias: Option<String>,
    pub user_api_key_team_alias: Option<String>,
}

/// The attributes `exception_type` reads off the Python exception: a provider error
/// (`BaseLLMException`) carries a status, a response and a request, a plain exception
/// carries only its text.
struct Raised {
    status: Option<u16>,
    status_is_synthesized: bool,
    message: String,
    response: Option<UpstreamResponse>,
}

impl Raised {
    fn provider(
        status: u16,
        message: String,
        body: String,
        headers: Vec<(String, String)>,
    ) -> Self {
        Self {
            status: Some(status),
            status_is_synthesized: false,
            message,
            response: Some(UpstreamResponse {
                status,
                body,
                headers,
            }),
        }
    }

    fn plain(message: String) -> Self {
        Self {
            status: None,
            status_is_synthesized: false,
            message,
            response: None,
        }
    }

    fn new(original: &OriginalException, asynchronous: bool) -> Self {
        match original {
            OriginalException::Http {
                status,
                body,
                headers,
            } => Self::provider(*status, body.clone(), body.clone(), headers.clone()),
            OriginalException::Connection { message } => Self {
                status_is_synthesized: true,
                ..Self::provider(500, message.clone(), String::new(), Vec::new())
            },
            OriginalException::Timeout {
                timeout_seconds,
                elapsed_seconds,
            } => Self::provider(
                408,
                timeout_message(asynchronous, *timeout_seconds, *elapsed_seconds),
                String::new(),
                Vec::new(),
            ),
            OriginalException::Response { message }
            | OriginalException::Local { message, .. }
            | OriginalException::Public { message, .. } => Self::plain(message.clone()),
        }
    }
}

/// The text `litellm.Timeout` carries when the Python HTTP handler times out: the sync
/// and async handlers word it differently.
fn timeout_message(
    asynchronous: bool,
    timeout_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
) -> String {
    let timeout = python_float(timeout_seconds);
    if asynchronous {
        let elapsed =
            python_float(elapsed_seconds.map(|seconds| (seconds * 1000.0).round() / 1000.0));
        format!(
            "litellm.Timeout: Connection timed out. Timeout passed={timeout}, time taken={elapsed} seconds"
        )
    } else {
        format!("litellm.Timeout: Connection timed out after {timeout} seconds.")
    }
}

fn python_float(value: Option<f64>) -> String {
    match value {
        None => "None".to_string(),
        Some(value) if value.fract() == 0.0 => format!("{value:.1}"),
        Some(value) => value.to_string(),
    }
}

/// Everything the rules read: the original as Python sees it and the text `exception_type`
/// derives from the context before any provider mapper runs.
struct Mapping<'a> {
    context: &'a ExceptionContext,
    original: Raised,
    provider: &'a str,
    error_str: String,
    exception_provider: String,
    extra_information: String,
}

impl<'a> Mapping<'a> {
    fn new(context: &'a ExceptionContext, original: &OriginalException) -> Self {
        let original = Raised::new(original, context.asynchronous);
        let error_str = if secret_redaction_enabled() {
            redact_string(&original.message)
        } else {
            original.message.clone()
        };
        Self {
            context,
            original,
            provider: context.custom_llm_provider.as_deref().unwrap_or_default(),
            error_str,
            exception_provider: match &context.custom_llm_provider {
                None => "None".to_string(),
                Some(provider) => exception_provider(provider),
            },
            extra_information: extra_information(context, api_base(context).as_deref()),
        }
    }

    fn failure(&self, kind: PublicKind, message: String, debug: bool) -> PublicFailure {
        PublicFailure {
            kind,
            message,
            model: self.context.model.clone(),
            llm_provider: self.context.custom_llm_provider.clone(),
            litellm_debug_info: debug.then(|| self.extra_information.clone()),
            litellm_response_headers: None,
            print_banner: false,
        }
    }
}

pub fn exception_type(context: &ExceptionContext, original: &OriginalException) -> PublicFailure {
    if let OriginalException::Public { class, message } = original {
        return PublicFailure {
            kind: PublicKind::Status {
                status_class: *class,
                response: None,
            },
            message: message.clone(),
            model: context.model.clone(),
            llm_provider: context.custom_llm_provider.clone(),
            litellm_debug_info: None,
            litellm_response_headers: None,
            print_banner: false,
        };
    }
    let mapping = Mapping::new(context, original);
    let litellm_response_headers = mapping
        .original
        .response
        .as_ref()
        .map(|response| response.headers.clone())
        .filter(|headers| !headers.is_empty());
    PublicFailure {
        litellm_response_headers,
        print_banner: !context.suppress_debug_info,
        ..map(&mapping)
    }
}

fn map(mapping: &Mapping<'_>) -> PublicFailure {
    if rules::contains_any(&mapping.error_str, TIMEOUT_MARKERS) {
        return mapping.failure(
            PublicKind::Timeout { status: None },
            format!(
                "APITimeoutError - Request timed out. Error_str: {}",
                mapping.error_str
            ),
            true,
        );
    }
    let provider_failure = match mapping.context.family {
        ExceptionFamily::OpenAiCompatible => openai::map(mapping),
        ExceptionFamily::VertexAi => vertex_ai::map(mapping),
        ExceptionFamily::Cohere => cohere::map(mapping),
        ExceptionFamily::Other => None,
    };
    provider_failure
        .or_else(|| status::map(mapping))
        .unwrap_or_else(|| unmapped(mapping))
}

/// The `APIConnectionError` Python raises when no mapper claimed the failure: with the
/// provider prefix for a provider error, with the bare text for a plain exception.
fn unmapped(mapping: &Mapping<'_>) -> PublicFailure {
    let message = match mapping.original.status {
        Some(_) => format!("{} - {}", mapping.exception_provider, mapping.error_str),
        None => mapping.original.message.clone(),
    };
    mapping.failure(PublicKind::ApiConnection, message, false)
}

fn exception_provider(provider: &str) -> String {
    let mut characters = provider.chars();
    match characters.next() {
        Some(first) => format!("{}{}Exception", first.to_uppercase(), characters.as_str()),
        None => String::new(),
    }
}

fn python_capitalize(value: &str) -> String {
    let mut characters = value.chars();
    match characters.next() {
        Some(first) => format!(
            "{}{}",
            first.to_uppercase(),
            characters.as_str().to_lowercase()
        ),
        None => String::new(),
    }
}

fn api_base(context: &ExceptionContext) -> Option<String> {
    match (&context.vertex_location, &context.vertex_project) {
        (Some(location), Some(project)) => Some(format!(
            "{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{}:generateContent",
            context.model
        )),
        _ => None,
    }
}

fn extra_information(context: &ExceptionContext, api_base: Option<&str>) -> String {
    let lines = [
        Some(format!("\nModel: {}", context.model)),
        api_base.map(|api_base| format!("\nAPI Base: `{api_base}`")),
        context
            .model_group
            .as_ref()
            .map(|value| format!("\nmodel_group: `{value}`\n")),
        context
            .deployment
            .as_ref()
            .map(|value| format!("\ndeployment: `{value}`\n")),
        context
            .vertex_project
            .as_ref()
            .map(|value| format!("\nvertex_project: `{value}`\n")),
        context
            .vertex_location
            .as_ref()
            .map(|value| format!("\nvertex_location: `{value}`\n")),
    ];
    let information: String = lines.into_iter().flatten().collect();
    match &context.user_api_key_alias {
        Some(alias) => format!(
            "\n\nKey Name: `{alias}`\nTeam: `{}`{information}",
            context.user_api_key_team_alias.as_deref().unwrap_or("None")
        ),
        None => information,
    }
}

#[cfg(test)]
mod testing {
    use super::*;

    pub(super) const DEBUG: &str = "\nModel: ocr-model";

    pub(super) fn context(provider: &str, family: ExceptionFamily) -> ExceptionContext {
        ExceptionContext {
            model: "ocr-model".into(),
            custom_llm_provider: Some(provider.into()),
            family,
            suppress_debug_info: true,
            ..ExceptionContext::default()
        }
    }

    pub(super) fn http(status: u16, body: &str) -> OriginalException {
        OriginalException::Http {
            status,
            body: body.into(),
            headers: vec![("retry-after".into(), "7".into())],
        }
    }

    pub(super) fn upstream(status: u16, body: &str) -> Option<ResponseArg> {
        Some(ResponseArg::Upstream(UpstreamResponse {
            status,
            body: body.into(),
            headers: vec![("retry-after".into(), "7".into())],
        }))
    }

    pub(super) fn status(class: StatusClass, response: Option<ResponseArg>) -> PublicKind {
        PublicKind::Status {
            status_class: class,
            response,
        }
    }

    /// The failure a rule builds before `exception_type` adds the response headers and the
    /// banner flag.
    pub(super) fn failure(kind: PublicKind, message: &str, provider: &str) -> PublicFailure {
        PublicFailure {
            kind,
            message: message.into(),
            model: "ocr-model".into(),
            llm_provider: Some(provider.into()),
            litellm_debug_info: None,
            litellm_response_headers: None,
            print_banner: false,
        }
    }

    pub(super) fn with_debug(failure: PublicFailure) -> PublicFailure {
        PublicFailure {
            litellm_debug_info: Some(DEBUG.into()),
            ..failure
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{DEBUG, context, failure, http, status, upstream, with_debug};
    use super::*;

    fn openai() -> ExceptionContext {
        context("mistral", ExceptionFamily::OpenAiCompatible)
    }

    #[test]
    fn a_public_original_passes_through_without_banner_debug_or_prefix() {
        let original = OriginalException::Public {
            class: StatusClass::UnsupportedParams,
            message: "Invalid `req_format`".into(),
        };
        let context = ExceptionContext {
            suppress_debug_info: false,
            ..openai()
        };
        assert_eq!(
            exception_type(&context, &original),
            failure(
                status(StatusClass::UnsupportedParams, None),
                "Invalid `req_format`",
                "mistral"
            )
        );
    }

    #[rstest::rstest]
    #[case::vertex_family_status_rule(ExceptionFamily::VertexAi, "vertex_ai", PublicFailure {
        litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
        ..with_debug(failure(status(StatusClass::BadRequest, upstream(409, "rejected")), "Vertex_aiException - rejected", "vertex_ai"))
    })]
    #[case::cohere_family(ExceptionFamily::Cohere, "cohere", PublicFailure {
        litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
        ..with_debug(failure(status(StatusClass::BadRequest, upstream(409, "rejected")), "CohereException - rejected", "cohere"))
    })]
    #[case::other_family(ExceptionFamily::Other, "reducto", PublicFailure {
        litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
        ..with_debug(failure(status(StatusClass::BadRequest, upstream(409, "rejected")), "ReductoException - rejected", "reducto"))
    })]
    fn families_without_a_409_rule_reach_the_status_table(
        #[case] family: ExceptionFamily,
        #[case] provider: &str,
        #[case] expected: PublicFailure,
    ) {
        assert_eq!(
            exception_type(&context(provider, family), &http(409, "rejected")),
            expected
        );
    }

    #[test]
    fn the_openai_family_claims_a_409_before_the_status_table() {
        assert_eq!(
            exception_type(&openai(), &http(409, "rejected")),
            PublicFailure {
                litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
                ..with_debug(failure(
                    PublicKind::Api {
                        status: 409,
                        request_url: DOCS_URL
                    },
                    "APIError: MistralException - rejected",
                    "mistral"
                ))
            }
        );
    }

    #[rstest::rstest]
    #[case::request_timeout_error("Request Timeout Error")]
    #[case::request_timed_out("Request timed out")]
    #[case::timed_out_generating("Timed out generating response")]
    #[case::read_operation("The read operation timed out")]
    fn timeout_markers_win_over_every_family(#[case] marker: &str) {
        let body = format!("rate limit {marker}");
        for family in [
            ExceptionFamily::OpenAiCompatible,
            ExceptionFamily::VertexAi,
            ExceptionFamily::Cohere,
            ExceptionFamily::Other,
        ] {
            assert_eq!(
                exception_type(&context("mistral", family), &http(429, &body)),
                PublicFailure {
                    litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
                    ..with_debug(failure(
                        PublicKind::Timeout { status: None },
                        &format!("APITimeoutError - Request timed out. Error_str: {body}"),
                        "mistral"
                    ))
                }
            );
        }
    }

    #[rstest::rstest]
    #[case::provider_error_keeps_the_prefix(http(409, "rejected"), "ReductoException - rejected")]
    #[case::synthesized_status_skips_the_status_table(
        OriginalException::Connection { message: "refused".into() },
        "ReductoException - refused"
    )]
    #[case::plain_exception_keeps_its_text(
        OriginalException::Local { class: LocalClass::FileNotFound, message: "File not found: /a".into() },
        "File not found: /a"
    )]
    fn unmapped_failures_are_connection_errors(
        #[case] original: OriginalException,
        #[case] message: &str,
    ) {
        let context = context("reducto", ExceptionFamily::Other);
        let expected = match original {
            OriginalException::Http { .. } => with_debug(failure(
                status(StatusClass::BadRequest, upstream(409, "rejected")),
                message,
                "reducto",
            )),
            OriginalException::Connection { .. } | OriginalException::Local { .. } => {
                failure(PublicKind::ApiConnection, message, "reducto")
            }
            _ => unreachable!(),
        };
        let actual = exception_type(&context, &original);
        assert_eq!(
            PublicFailure {
                litellm_response_headers: None,
                ..actual
            },
            expected
        );
    }

    #[test]
    fn a_missing_provider_renders_like_python_none() {
        let context = ExceptionContext {
            custom_llm_provider: None,
            family: ExceptionFamily::Other,
            ..openai()
        };
        assert_eq!(
            exception_type(&context, &http(401, "rejected")),
            PublicFailure {
                llm_provider: None,
                litellm_response_headers: Some(vec![("retry-after".into(), "7".into())]),
                ..with_debug(failure(
                    status(StatusClass::Authentication, upstream(401, "rejected")),
                    "None - rejected",
                    "unused"
                ))
            }
        );
    }

    #[rstest::rstest]
    #[case::suppressed(true, false)]
    #[case::printed(false, true)]
    fn the_banner_prints_unless_debug_info_is_suppressed(
        #[case] suppress_debug_info: bool,
        #[case] print_banner: bool,
    ) {
        let context = ExceptionContext {
            suppress_debug_info,
            ..openai()
        };
        assert_eq!(
            exception_type(&context, &http(400, "rejected")).print_banner,
            print_banner
        );
    }

    #[test]
    fn empty_upstream_headers_are_not_reported() {
        let original = OriginalException::Http {
            status: 400,
            body: "rejected".into(),
            headers: Vec::new(),
        };
        assert_eq!(
            exception_type(&openai(), &original).litellm_response_headers,
            None
        );
    }

    #[test]
    fn messages_are_redacted_before_markers_and_prefixes() {
        let body = "rejected Bearer abcdefghijklmnop";
        assert_eq!(
            exception_type(
                &context("reducto", ExceptionFamily::Other),
                &http(400, body)
            )
            .message,
            "ReductoException - rejected REDACTED"
        );
    }

    const SYNC_TIMEOUT: &str = "litellm.Timeout: Connection timed out after 0.5 seconds.";

    #[rstest::rstest]
    #[case::sync(false, Some(0.5), Some(0.5031), SYNC_TIMEOUT)]
    #[case::async_rounds_the_elapsed_time(
        true,
        Some(0.5),
        Some(0.5031),
        "litellm.Timeout: Connection timed out. Timeout passed=0.5, time taken=0.503 seconds"
    )]
    #[case::whole_seconds_keep_a_decimal(
        true,
        Some(600.0),
        Some(2.0),
        "litellm.Timeout: Connection timed out. Timeout passed=600.0, time taken=2.0 seconds"
    )]
    #[case::unknown_values_render_as_none(
        true,
        None,
        None,
        "litellm.Timeout: Connection timed out. Timeout passed=None, time taken=None seconds"
    )]
    fn timeout_text_follows_the_delivery_mode(
        #[case] asynchronous: bool,
        #[case] timeout_seconds: Option<f64>,
        #[case] elapsed_seconds: Option<f64>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            timeout_message(asynchronous, timeout_seconds, elapsed_seconds),
            expected
        );
    }

    #[rstest::rstest]
    #[case::sync(false, SYNC_TIMEOUT)]
    #[case::async_(
        true,
        "litellm.Timeout: Connection timed out. Timeout passed=0.5, time taken=0.503 seconds"
    )]
    fn a_timeout_is_a_408_carrying_the_handler_text(
        #[case] asynchronous: bool,
        #[case] text: &str,
    ) {
        let context = ExceptionContext {
            asynchronous,
            ..openai()
        };
        let original = OriginalException::Timeout {
            timeout_seconds: Some(0.5),
            elapsed_seconds: Some(0.5031),
        };
        assert_eq!(
            exception_type(&context, &original),
            with_debug(failure(
                PublicKind::Timeout { status: None },
                &format!("Timeout Error: MistralException - {text}"),
                "mistral"
            ))
        );
    }

    #[test]
    fn a_refused_connection_is_a_500_with_an_empty_response() {
        assert_eq!(
            exception_type(
                &openai(),
                &OriginalException::Connection {
                    message: "refused".into()
                }
            ),
            with_debug(failure(
                status(
                    StatusClass::InternalServer,
                    Some(ResponseArg::Upstream(UpstreamResponse {
                        status: 500,
                        body: String::new(),
                        headers: Vec::new(),
                    }))
                ),
                "InternalServerError: MistralException - refused",
                "mistral"
            ))
        );
    }

    #[test]
    fn debug_information_follows_the_python_layout() {
        let context = ExceptionContext {
            vertex_project: Some("project".into()),
            vertex_location: Some("region".into()),
            model_group: Some("ocr".into()),
            deployment: Some("deployment".into()),
            user_api_key_alias: Some("key".into()),
            ..openai()
        };
        assert_eq!(
            extra_information(&context, api_base(&context).as_deref()),
            concat!(
                "\n\nKey Name: `key`\nTeam: `None`",
                "\nModel: ocr-model",
                "\nAPI Base: `region-aiplatform.googleapis.com/v1/projects/project/locations/region/publishers/google/models/ocr-model:generateContent`",
                "\nmodel_group: `ocr`\n",
                "\ndeployment: `deployment`\n",
                "\nvertex_project: `project`\n",
                "\nvertex_location: `region`\n",
            )
        );
    }

    #[rstest::rstest]
    #[case::bare(ExceptionContext::default(), "\nModel: ")]
    #[case::redacted_messages(ExceptionContext { redact_messages_in_exceptions: true, model: "m".into(), ..ExceptionContext::default() }, "\nModel: m")]
    #[case::team_alias(
        ExceptionContext { model: "m".into(), user_api_key_alias: Some("key".into()), user_api_key_team_alias: Some("team".into()), redact_messages_in_exceptions: true, ..ExceptionContext::default() },
        "\n\nKey Name: `key`\nTeam: `team`\nModel: m"
    )]
    #[case::team_alias_without_key_is_ignored(
        ExceptionContext { model: "m".into(), user_api_key_team_alias: Some("team".into()), redact_messages_in_exceptions: true, ..ExceptionContext::default() },
        "\nModel: m"
    )]
    #[case::project_without_location_has_no_api_base(
        ExceptionContext { model: "m".into(), vertex_project: Some("p".into()), redact_messages_in_exceptions: true, ..ExceptionContext::default() },
        "\nModel: m\nvertex_project: `p`\n"
    )]
    #[case::location_without_project_has_no_api_base(
        ExceptionContext { model: "m".into(), vertex_location: Some("l".into()), redact_messages_in_exceptions: true, ..ExceptionContext::default() },
        "\nModel: m\nvertex_location: `l`\n"
    )]
    fn each_optional_context_field_adds_its_own_line(
        #[case] context: ExceptionContext,
        #[case] expected: &str,
    ) {
        assert_eq!(
            extra_information(&context, api_base(&context).as_deref()),
            expected
        );
    }

    #[rstest::rstest]
    #[case::lowercase("mistral", "MistralException")]
    #[case::keeps_the_rest("azure_ai", "Azure_aiException")]
    #[case::empty("", "")]
    fn exception_provider_capitalizes_only_the_first_letter(
        #[case] provider: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(exception_provider(provider), expected);
    }

    #[rstest::rstest]
    #[case::lowers_the_rest("vERTEX_AI", "Vertex_ai")]
    #[case::empty("", "")]
    fn python_capitalize_lowers_the_rest(#[case] value: &str, #[case] expected: &str) {
        assert_eq!(python_capitalize(value), expected);
    }

    #[test]
    fn debug_constant_matches_the_default_test_context() {
        let context = openai();
        assert_eq!(extra_information(&context, None), DEBUG);
    }
}
