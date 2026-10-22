//! A port of Python's `exception_type` for the routes that run in Rust. Rust decides the
//! public class, the message and the debug text; Python only builds the class.
//!
//! DIVERGENCES: where the Python mapper is inconsistent, the port follows one rule instead.
//! - The message is always `{Provider}Exception - {redacted text}`. Python's per-branch
//!   labels (`RateLimitError: `, `litellm.RateLimitError: `, `Vertex_aiException BadRequestError`)
//!   are dropped because every public class already prefixes `litellm.{Class}: `.
//! - The upstream response is always the real one. Python swaps in made-up `httpx.Response`
//!   stubs on some Vertex branches, losing the body and `retry-after`.
//! - The debug text is always attached; Python passes it on some branches only.
//! - No family rule turns a status into a class; the shared status table owns that. So a
//!   Vertex 502 is a `BadGatewayError` and an OpenAI-family 403 is a `PermissionDeniedError`.
//!   Three rules read the status only to gate a text match, as Python does: the standalone
//!   `429`, Vertex's wrapped 429 behind a 5xx, and Cohere's rules for failures with no status.
//! - A timeout text marker on an HTTP failure keeps the upstream response. Python's `Timeout`
//!   carries none.
//! - Every family matches and reports the redacted text. Python's OpenAI mapper builds the
//!   message from the unredacted text.
//! - A refused connection is an `APIConnectionError`, not the 500 Python's HTTP handler
//!   synthesizes.
//! - Dropped Python rules: Vertex's bare `403` substring (it matches `4031 tokens`), Vertex's
//!   `IndexError` quota marker (a Python client crash), the OpenAI SDK's missing-`api_key`
//!   text and its `OPENAI` renaming, Cohere's `llm_provider="cohere"` override, and Cohere's
//!   `CohereConnectionError` check (a Python SDK class name).
//!
//! KNOWN_GAPS: differences from the Python mapper that no Rust route can reach today. Each
//! one stops being acceptable at its trigger.
//! - The Vertex partner-model API base for "claude" models is not built into the debug text.
//!   Trigger: a Vertex route whose models include Anthropic partner models.
//! - The debug text's `API Base` line is only the non-streaming Vertex URL. Python prefers an
//!   explicit or provider-resolved `api_base`, uses `:streamGenerateContent` when streaming,
//!   and has Gemini and OpenAI defaults. Trigger: the first route wired to this mapper, since
//!   every route knows its `api_base`.
//! - The debug text has no `Messages:` line, which Python adds when
//!   `redact_messages_in_exceptions` is off. Trigger: a wired route that carries messages.
//! - Python reports the provider `get_llm_provider` resolves for a stripped model name when
//!   that name happens to be in the model cost map. Trigger: a route whose model names
//!   overlap the cost map; that needs the provider resolution port, not a classifier change.
//! - `litellm_proxy` errors are not unwrapped into the proxied exception. Trigger: a Rust
//!   route that calls a LiteLLM proxy.
//! - Only the OpenAI-compatible, Vertex AI and Cohere mappers are ported; every other
//!   provider goes straight to the status table. Trigger: a Rust route for such a provider.

use super::secret_redaction::SecretRedactor;

mod cohere;
mod openai;
mod original;
mod public;
mod rules;
mod status;
mod vertex_ai;

pub use original::{ExceptionFamily, OriginalException};
pub use public::{MappedFailure, PublicError, UpstreamResponse};

use rules::{Rule, contains_any, first_match};

const TIMEOUT_MARKERS: &[&str] = &[
    "Request Timeout Error",
    "Request timed out",
    "Timed out generating response",
    "The read operation timed out",
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExceptionContext {
    pub model: String,
    pub custom_llm_provider: String,
    pub asynchronous: bool,
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
    pub model_group: Option<String>,
    pub deployment: Option<String>,
    pub user_api_key_alias: Option<String>,
    pub user_api_key_team_alias: Option<String>,
}

/// What the rules read: the status of a provider response, if any, and the redacted text.
struct Mapping {
    status: Option<u16>,
    error_str: String,
}

pub fn exception_type(
    context: &ExceptionContext,
    redactor: Option<&SecretRedactor>,
    original: &OriginalException,
) -> MappedFailure {
    let (status, text, upstream) = match original {
        OriginalException::Http {
            status,
            body,
            headers,
        } => (
            Some(*status),
            body.clone(),
            Some(UpstreamResponse {
                status: *status,
                body: body.clone(),
                headers: headers.clone(),
            }),
        ),
        OriginalException::Connection { message } | OriginalException::Plain { message } => {
            (None, message.clone(), None)
        }
        OriginalException::Timeout {
            timeout_seconds,
            elapsed_seconds,
        } => (
            None,
            timeout_message(context.asynchronous, *timeout_seconds, *elapsed_seconds),
            None,
        ),
    };
    let mapping = Mapping {
        status,
        error_str: match redactor {
            Some(redactor) => redactor.redact(&text),
            None => text,
        },
    };
    let family = ExceptionFamily::for_provider(&context.custom_llm_provider);
    let (error, hint) = classify(family, original, &mapping);
    MappedFailure {
        error,
        message: format!(
            "{} - {}{hint}",
            exception_provider(&context.custom_llm_provider),
            mapping.error_str
        ),
        upstream,
        debug_info: extra_information(context, api_base(context).as_deref()),
    }
}

fn classify(
    family: ExceptionFamily,
    original: &OriginalException,
    mapping: &Mapping,
) -> (PublicError, &'static str) {
    const TIMEOUT: PublicError = PublicError::Timeout { status: 408 };
    if matches!(original, OriginalException::Timeout { .. })
        || contains_any(&mapping.error_str, TIMEOUT_MARKERS)
    {
        return (TIMEOUT, "");
    }
    if let Some(rule) = first_match(family_rules(family), mapping) {
        return (rule.error, rule.hint);
    }
    let by_status = mapping.status.and_then(status::classify);
    (by_status.unwrap_or(PublicError::ApiConnection), "")
}

fn family_rules(family: ExceptionFamily) -> &'static [Rule] {
    match family {
        ExceptionFamily::OpenAiCompatible => openai::RULES,
        ExceptionFamily::VertexAi => vertex_ai::RULES,
        ExceptionFamily::Cohere => cohere::RULES,
        ExceptionFamily::Other => &[],
    }
}

/// The text the Python HTTP handler's timeout carries: the sync and async handlers word it
/// differently.
fn timeout_message(
    asynchronous: bool,
    timeout_seconds: Option<f64>,
    elapsed_seconds: Option<f64>,
) -> String {
    let timeout = python_float(timeout_seconds);
    if asynchronous {
        let elapsed =
            python_float(elapsed_seconds.map(|seconds| (seconds * 1000.0).round() / 1000.0));
        format!("Connection timed out. Timeout passed={timeout}, time taken={elapsed} seconds")
    } else {
        format!("Connection timed out after {timeout} seconds.")
    }
}

fn python_float(value: Option<f64>) -> String {
    match value {
        None => "None".to_string(),
        Some(value) if value.fract() == 0.0 => format!("{value:.1}"),
        Some(value) => value.to_string(),
    }
}

fn exception_provider(provider: &str) -> String {
    if provider == "openai" {
        return "OpenAIException".to_string();
    }
    let mut characters = provider.chars();
    match characters.next() {
        Some(first) => format!("{}{}Exception", first.to_uppercase(), characters.as_str()),
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
    use super::Mapping;

    pub(super) fn mapping(status: Option<u16>, text: &str) -> Mapping {
        Mapping {
            status,
            error_str: text.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEBUG: &str = "\nModel: ocr-model";

    fn context(provider: &str) -> ExceptionContext {
        ExceptionContext {
            model: "ocr-model".into(),
            custom_llm_provider: provider.into(),
            ..ExceptionContext::default()
        }
    }

    fn redactor() -> SecretRedactor {
        SecretRedactor::new(16)
    }

    fn headers() -> Vec<(String, String)> {
        vec![("retry-after".into(), "7".into())]
    }

    fn http(status: u16, body: &str) -> OriginalException {
        OriginalException::Http {
            status,
            body: body.into(),
            headers: headers(),
        }
    }

    fn upstream(status: u16, body: &str) -> Option<UpstreamResponse> {
        Some(UpstreamResponse {
            status,
            body: body.into(),
            headers: headers(),
        })
    }

    fn mapped(provider: &str, original: &OriginalException) -> MappedFailure {
        exception_type(&context(provider), Some(&redactor()), original)
    }

    #[rstest::rstest]
    #[case::openai_family("mistral", "rate limit reached", PublicError::RateLimit)]
    #[case::vertex_family("vertex_ai", "Resource exhausted", PublicError::RateLimit)]
    #[case::cohere_family("cohere", "too many tokens", PublicError::ContextWindowExceeded)]
    fn a_family_text_rule_beats_the_status_and_keeps_the_real_response(
        #[case] provider: &str,
        #[case] body: &str,
        #[case] expected: PublicError,
    ) {
        let failure = mapped(provider, &http(401, body));
        assert_eq!(failure.error, expected);
        assert_eq!(failure.upstream, upstream(401, body));
    }

    #[test]
    fn the_other_family_has_no_text_rules() {
        assert_eq!(
            mapped("reducto", &http(401, "rate limit reached")).error,
            PublicError::Authentication
        );
    }

    #[rstest::rstest]
    #[case::openai_403_is_permission_denied("mistral", 403, PublicError::PermissionDenied)]
    #[case::openai_409_is_bad_request("mistral", 409, PublicError::BadRequest)]
    #[case::vertex_502_is_bad_gateway("vertex_ai", 502, PublicError::BadGateway)]
    #[case::vertex_504_is_a_timeout("vertex_ai", 504, PublicError::Timeout { status: 504 })]
    #[case::cohere_498_is_bad_request("cohere", 498, PublicError::BadRequest)]
    #[case::other_503("reducto", 503, PublicError::ServiceUnavailable)]
    fn without_a_text_rule_every_family_uses_the_status_table(
        #[case] provider: &str,
        #[case] status: u16,
        #[case] expected: PublicError,
    ) {
        assert_eq!(
            mapped(provider, &http(status, "rejected")),
            MappedFailure {
                error: expected,
                message: format!("{} - rejected", exception_provider(provider)),
                upstream: upstream(status, "rejected"),
                debug_info: DEBUG.into(),
            }
        );
    }

    #[rstest::rstest]
    #[case::request_timeout_error("Request Timeout Error")]
    #[case::request_timed_out("Request timed out")]
    #[case::timed_out_generating("Timed out generating response")]
    #[case::read_operation("The read operation timed out")]
    fn timeout_markers_win_over_every_family(#[case] marker: &str) {
        let body = format!("rate limit invalid api token {marker}");
        for provider in ["mistral", "vertex_ai", "cohere", "reducto"] {
            assert_eq!(
                mapped(provider, &http(429, &body)).error,
                PublicError::Timeout { status: 408 },
                "{provider}"
            );
        }
    }

    #[test]
    fn a_handler_timeout_is_a_408_without_a_response() {
        let original = OriginalException::Timeout {
            timeout_seconds: Some(0.5),
            elapsed_seconds: Some(0.5031),
        };
        assert_eq!(
            mapped("mistral", &original),
            MappedFailure {
                error: PublicError::Timeout { status: 408 },
                message: "MistralException - Connection timed out after 0.5 seconds.".into(),
                upstream: None,
                debug_info: DEBUG.into(),
            }
        );
    }

    #[rstest::rstest]
    #[case::refused_connection(OriginalException::Connection { message: "refused".into() })]
    #[case::unparseable_response(OriginalException::Plain { message: "refused".into() })]
    #[case::informational_status(OriginalException::Http { status: 399, body: "refused".into(), headers: Vec::new() })]
    fn a_failure_no_rule_or_status_claims_is_a_connection_error(
        #[case] original: OriginalException,
    ) {
        let failure = mapped("reducto", &original);
        assert_eq!(failure.error, PublicError::ApiConnection);
        assert_eq!(failure.message, "ReductoException - refused");
    }

    #[test]
    fn a_timeout_marker_on_a_response_keeps_the_response() {
        let failure = mapped("reducto", &http(429, "Request timed out"));
        assert_eq!(failure.error, PublicError::Timeout { status: 408 });
        assert_eq!(failure.upstream, upstream(429, "Request timed out"));
    }

    #[test]
    fn family_text_rules_also_classify_failures_without_a_response() {
        let original = OriginalException::Plain {
            message: "Request too large".into(),
        };
        assert_eq!(mapped("mistral", &original).error, PublicError::RateLimit);
    }

    #[rstest::rstest]
    #[case::openai_family("mistral", "MistralException - rejected REDACTED")]
    #[case::vertex_family("vertex_ai", "Vertex_aiException - rejected REDACTED")]
    #[case::other_family("reducto", "ReductoException - rejected REDACTED")]
    fn every_family_reports_the_redacted_text(#[case] provider: &str, #[case] message: &str) {
        let failure = mapped(provider, &http(400, "rejected Bearer abcdefghijklmnop"));
        assert_eq!(failure.message, message);
    }

    #[test]
    fn redaction_runs_before_the_rules_see_the_text() {
        let body = "db_password=rate_limit";
        assert_eq!(
            mapped("mistral", &http(400, body)).error,
            PublicError::BadRequest
        );
        assert_eq!(
            exception_type(&context("mistral"), None, &http(400, body)).error,
            PublicError::RateLimit
        );
    }

    #[test]
    fn without_a_redactor_the_text_is_kept() {
        let body = "rejected Bearer abcdefghijklmnop";
        assert_eq!(
            exception_type(&context("reducto"), None, &http(400, body)).message,
            format!("ReductoException - {body}")
        );
    }

    #[test]
    fn a_rule_hint_follows_the_message() {
        let failure = mapped("mistral", &http(400, "invalid_encrypted_content"));
        assert_eq!(failure.error, PublicError::BadRequest);
        assert!(
            failure
                .message
                .starts_with("MistralException - invalid_encrypted_content\n\n This error occurs")
        );
    }

    #[rstest::rstest]
    #[case::sync(
        false,
        Some(0.5),
        Some(0.5031),
        "Connection timed out after 0.5 seconds."
    )]
    #[case::async_rounds_the_elapsed_time(
        true,
        Some(0.5),
        Some(0.5031),
        "Connection timed out. Timeout passed=0.5, time taken=0.503 seconds"
    )]
    #[case::whole_seconds_keep_a_decimal(
        true,
        Some(600.0),
        Some(2.0),
        "Connection timed out. Timeout passed=600.0, time taken=2.0 seconds"
    )]
    #[case::unknown_values_render_as_none(
        true,
        None,
        None,
        "Connection timed out. Timeout passed=None, time taken=None seconds"
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

    #[test]
    fn debug_information_follows_the_python_layout() {
        let context = ExceptionContext {
            vertex_project: Some("project".into()),
            vertex_location: Some("region".into()),
            model_group: Some("ocr".into()),
            deployment: Some("deployment".into()),
            user_api_key_alias: Some("key".into()),
            ..context("vertex_ai")
        };
        assert_eq!(
            exception_type(&context, None, &http(400, "rejected")).debug_info,
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
    #[case::team_alias(
        ExceptionContext { model: "m".into(), user_api_key_alias: Some("key".into()), user_api_key_team_alias: Some("team".into()), ..ExceptionContext::default() },
        "\n\nKey Name: `key`\nTeam: `team`\nModel: m"
    )]
    #[case::team_alias_without_key_is_ignored(
        ExceptionContext { model: "m".into(), user_api_key_team_alias: Some("team".into()), ..ExceptionContext::default() },
        "\nModel: m"
    )]
    #[case::project_without_location_has_no_api_base(
        ExceptionContext { model: "m".into(), vertex_project: Some("p".into()), ..ExceptionContext::default() },
        "\nModel: m\nvertex_project: `p`\n"
    )]
    #[case::location_without_project_has_no_api_base(
        ExceptionContext { model: "m".into(), vertex_location: Some("l".into()), ..ExceptionContext::default() },
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
    #[case::openai_keeps_its_brand("openai", "OpenAIException")]
    #[case::lowercase("mistral", "MistralException")]
    #[case::keeps_the_rest("azure_ai", "Azure_aiException")]
    #[case::empty("", "")]
    fn exception_provider_capitalizes_only_the_first_letter(
        #[case] provider: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(exception_provider(provider), expected);
    }
}
