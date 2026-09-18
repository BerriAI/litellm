use std::sync::LazyLock;

use fancy_regex::Regex;
use serde_json::Value;

use super::Mapping;
use super::public::{HttpStub, PublicFailure, PublicKind, ResponseArg, StatusClass};

const GITHUB_URL: &str = "https://github.com/BerriAI/litellm";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResponseChoice {
    Omitted,
    Provider,
    Stub { status: u16, url: &'static str },
    InternalServerStub,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ApiStatus {
    Fixed(u16),
    Original,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Status {
        class: StatusClass,
        response: ResponseChoice,
    },
    Timeout(Option<u16>),
    ApiConnection,
    Api {
        status: ApiStatus,
        request_url: &'static str,
    },
}

/// One branch of a Python `_map_*_exception` function: when it applies, the class it
/// raises, the message it builds, and whether it passes `litellm_debug_info`.
pub(super) struct Rule {
    pub(super) when: fn(&Mapping<'_>) -> bool,
    pub(super) kind: Kind,
    pub(super) message: fn(&Mapping<'_>) -> String,
    pub(super) debug: bool,
}

/// The first rule that applies decides the failure, as the `if`/`elif` chain does in Python.
pub(super) fn apply(rules: &[Rule], mapping: &Mapping<'_>) -> Option<PublicFailure> {
    rules
        .iter()
        .find(|rule| (rule.when)(mapping))
        .map(|rule| rule.build(mapping))
}

impl Rule {
    fn build(&self, mapping: &Mapping<'_>) -> PublicFailure {
        let kind = match self.kind {
            Kind::Status { class, response } => PublicKind::Status {
                status_class: class,
                response: response.resolve(mapping),
            },
            Kind::Timeout(status) => PublicKind::Timeout { status },
            Kind::ApiConnection => PublicKind::ApiConnection,
            Kind::Api {
                status,
                request_url,
            } => PublicKind::Api {
                status: match status {
                    ApiStatus::Fixed(status) => status,
                    ApiStatus::Original => mapping.original.status.unwrap_or(500),
                },
                request_url,
            },
        };
        PublicFailure {
            kind,
            message: (self.message)(mapping),
            model: mapping.context.model.clone(),
            llm_provider: mapping.context.custom_llm_provider.clone(),
            litellm_debug_info: self.debug.then(|| mapping.extra_information.clone()),
            litellm_response_headers: None,
            print_banner: false,
        }
    }
}

impl ResponseChoice {
    fn resolve(self, mapping: &Mapping<'_>) -> Option<ResponseArg> {
        match self {
            Self::Omitted => None,
            Self::Provider => mapping.original.response.clone().map(ResponseArg::Upstream),
            Self::Stub { status, url } => Some(ResponseArg::Stub(HttpStub {
                status,
                method: "POST",
                url,
                content: None,
            })),
            Self::InternalServerStub => Some(ResponseArg::Stub(HttpStub {
                status: 500,
                method: "completion",
                url: GITHUB_URL,
                content: Some(mapping.original.message.clone()),
            })),
        }
    }
}

pub(super) fn contains_any(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

static STANDALONE_429: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b429\b").expect("valid regex"));
static RATE_LIMIT_PHRASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"rate[\s_\-]*limit").expect("valid regex"));

/// `ExceptionCheckers.is_error_str_rate_limit`.
pub(super) fn is_rate_limit(error_str: &str, status: Option<u16>) -> bool {
    if STANDALONE_429.is_match(error_str).unwrap_or(false) && status == Some(429) {
        return true;
    }
    let lower = error_str.to_lowercase();
    RATE_LIMIT_PHRASE.is_match(&lower).unwrap_or(false)
        || lower.contains("service tier capacity exceeded")
}

/// `ExceptionCheckers.is_error_str_context_window_exceeded`.
pub(super) fn is_context_window_exceeded(error_str: &str) -> bool {
    let lower = error_str.to_lowercase();
    if lower.contains("string_above_max_length") {
        return false;
    }
    if lower.contains("invalid 'user'") && lower.contains("string too long") {
        return false;
    }
    contains_any(
        &lower,
        &[
            "exceed context limit",
            "this model's maximum context length is",
            "string too long. expected a string with maximum length",
            "model's maximum context limit",
            "is longer than the model's context length",
            "input tokens exceed the configured limit",
            "`inputs` tokens + `max_new_tokens` must be",
            "exceeds the available context size",
            "exceeds the maximum number of tokens allowed",
        ],
    ) || (lower.contains("current length is") && lower.contains("while limit is"))
        || (lower.contains("maximum input length is") && lower.contains("tokens"))
}

/// The integer `error.code` of a JSON error body, read the way Python's `int()` would.
pub(super) fn body_error_code(error_str: &str) -> Option<i64> {
    let body: Value = serde_json::from_str(error_str).ok()?;
    let Some(Value::Object(error)) = body.as_object()?.get("error") else {
        return None;
    };
    match error.get("code")? {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|value| value.trunc() as i64)),
        Value::String(code) => code.trim().replace('_', "").parse().ok(),
        Value::Bool(flag) => Some(i64::from(*flag)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::{context, failure, http};
    use super::super::{ExceptionFamily, OriginalException, UpstreamResponse};
    use super::*;

    fn first_marker(mapping: &Mapping<'_>) -> bool {
        mapping.error_str.contains("first")
    }

    fn always(_: &Mapping<'_>) -> bool {
        true
    }

    fn text(mapping: &Mapping<'_>) -> String {
        format!("seen {}", mapping.error_str)
    }

    const ORDERED: &[Rule] = &[
        Rule {
            when: first_marker,
            kind: Kind::Status {
                class: StatusClass::NotFound,
                response: ResponseChoice::Omitted,
            },
            message: text,
            debug: false,
        },
        Rule {
            when: always,
            kind: Kind::ApiConnection,
            message: text,
            debug: true,
        },
    ];

    fn apply_one(kind: Kind, debug: bool, original: &OriginalException) -> Option<PublicFailure> {
        let context = context("mistral", ExceptionFamily::OpenAiCompatible);
        let mapping = Mapping::new(&context, original);
        apply(
            &[Rule {
                when: always,
                kind,
                message: text,
                debug,
            }],
            &mapping,
        )
    }

    #[rstest::rstest]
    #[case::earlier_rule_wins("first and second", failure(
        PublicKind::Status { status_class: StatusClass::NotFound, response: None },
        "seen first and second",
        "mistral",
    ))]
    #[case::later_rule_when_the_earlier_does_not_apply("second", PublicFailure {
        litellm_debug_info: Some("\nModel: ocr-model\nMessages: `None`".into()),
        ..failure(PublicKind::ApiConnection, "seen second", "mistral")
    })]
    fn the_first_applicable_rule_decides(#[case] body: &str, #[case] expected: PublicFailure) {
        let context = context("mistral", ExceptionFamily::OpenAiCompatible);
        let original = http(400, body);
        assert_eq!(
            apply(ORDERED, &Mapping::new(&context, &original)),
            Some(expected)
        );
    }

    #[test]
    fn no_applicable_rule_leaves_the_failure_to_the_caller() {
        let context = context("mistral", ExceptionFamily::OpenAiCompatible);
        let original = http(400, "second");
        assert_eq!(
            apply(&ORDERED[..1], &Mapping::new(&context, &original)),
            None
        );
    }

    #[rstest::rstest]
    #[case::omitted(ResponseChoice::Omitted, None)]
    #[case::provider(ResponseChoice::Provider, Some(ResponseArg::Upstream(UpstreamResponse {
        status: 400,
        body: "body".into(),
        headers: vec![("retry-after".into(), "7".into())],
    })))]
    #[case::stub(
        ResponseChoice::Stub { status: 429, url: "https://stub.test" },
        Some(ResponseArg::Stub(HttpStub { status: 429, method: "POST", url: "https://stub.test", content: None }))
    )]
    #[case::internal_server_stub(
        ResponseChoice::InternalServerStub,
        Some(ResponseArg::Stub(HttpStub {
            status: 500,
            method: "completion",
            url: GITHUB_URL,
            content: Some("body".into()),
        }))
    )]
    fn response_choices_resolve_against_the_original(
        #[case] response: ResponseChoice,
        #[case] expected: Option<ResponseArg>,
    ) {
        let built = apply_one(
            Kind::Status {
                class: StatusClass::BadRequest,
                response,
            },
            false,
            &http(400, "body"),
        )
        .unwrap();
        assert_eq!(
            built.kind,
            PublicKind::Status {
                status_class: StatusClass::BadRequest,
                response: expected,
            }
        );
    }

    #[rstest::rstest]
    #[case::fixed(ApiStatus::Fixed(500), http(409, "body"), 500)]
    #[case::original(ApiStatus::Original, http(409, "body"), 409)]
    #[case::original_without_a_status(
        ApiStatus::Original,
        OriginalException::Response { message: "body".into() },
        500
    )]
    fn api_status_is_fixed_or_the_originals(
        #[case] status: ApiStatus,
        #[case] original: OriginalException,
        #[case] expected: u16,
    ) {
        let built = apply_one(
            Kind::Api {
                status,
                request_url: "https://api.test",
            },
            false,
            &original,
        )
        .unwrap();
        assert_eq!(
            built,
            failure(
                PublicKind::Api {
                    status: expected,
                    request_url: "https://api.test"
                },
                "seen body",
                "mistral"
            )
        );
    }

    #[rstest::rstest]
    #[case::with_debug(true, Some("\nModel: ocr-model\nMessages: `None`"))]
    #[case::without_debug(false, None)]
    fn debug_rules_carry_the_extra_information(
        #[case] debug: bool,
        #[case] expected: Option<&str>,
    ) {
        let built = apply_one(Kind::Timeout(Some(504)), debug, &http(504, "body")).unwrap();
        assert_eq!(
            built,
            PublicFailure {
                litellm_debug_info: expected.map(str::to_string),
                ..failure(
                    PublicKind::Timeout { status: Some(504) },
                    "seen body",
                    "mistral"
                )
            }
        );
    }

    #[rstest::rstest]
    #[case::standalone_429_with_429_status("got 429 back", Some(429), true)]
    #[case::standalone_429_with_other_status("got 429 back", Some(400), false)]
    #[case::embedded_429("token4290", Some(429), false)]
    #[case::phrase_spaced("Rate Limit reached", None, true)]
    #[case::phrase_underscored("rate_limit", None, true)]
    #[case::phrase_hyphenated("rate-limit", None, true)]
    #[case::service_tier("Service tier capacity exceeded", None, true)]
    #[case::unrelated("rejected", Some(429), false)]
    fn rate_limit_detection(
        #[case] text: &str,
        #[case] status: Option<u16>,
        #[case] expected: bool,
    ) {
        assert_eq!(is_rate_limit(text, status), expected);
    }

    #[rstest::rstest]
    #[case::exceed_context_limit("Exceed context limit", true)]
    #[case::maximum_context_length("This model's maximum context length is 10", true)]
    #[case::string_too_long("string too long. Expected a string with maximum length 5", true)]
    #[case::maximum_context_limit("the model's maximum context limit", true)]
    #[case::longer_than_context("prompt is longer than the model's context length", true)]
    #[case::configured_limit("input tokens exceed the configured limit", true)]
    #[case::max_new_tokens("`inputs` tokens + `max_new_tokens` must be <= 10", true)]
    #[case::available_context("exceeds the available context size", true)]
    #[case::maximum_tokens("exceeds the maximum number of tokens allowed", true)]
    #[case::current_and_limit("current length is 9 while limit is 8", true)]
    #[case::current_without_limit("current length is 9", false)]
    #[case::maximum_input_tokens("maximum input length is 8 tokens", true)]
    #[case::maximum_input_without_tokens("maximum input length is 8", false)]
    #[case::string_above_max_length_wins("string_above_max_length exceed context limit", false)]
    #[case::user_field_is_not_context(
        "invalid 'user': string too long. expected a string with maximum length",
        false
    )]
    #[case::unrelated("rejected", false)]
    fn context_window_detection(#[case] text: &str, #[case] expected: bool) {
        assert_eq!(is_context_window_exceeded(text), expected);
    }

    #[rstest::rstest]
    #[case::integer(r#"{"error": {"code": 429}}"#, Some(429))]
    #[case::float(r#"{"error": {"code": 429.9}}"#, Some(429))]
    #[case::string(r#"{"error": {"code": " 4_29 "}}"#, Some(429))]
    #[case::boolean(r#"{"error": {"code": true}}"#, Some(1))]
    #[case::unparseable_string(r#"{"error": {"code": "slow"}}"#, None)]
    #[case::null(r#"{"error": {"code": null}}"#, None)]
    #[case::no_code(r#"{"error": {}}"#, None)]
    #[case::error_not_an_object(r#"{"error": "429"}"#, None)]
    #[case::no_error(r#"{"code": 429}"#, None)]
    #[case::not_an_object("[429]", None)]
    #[case::not_json("429", None)]
    fn body_error_code_reads_the_nested_code(#[case] body: &str, #[case] expected: Option<i64>) {
        assert_eq!(body_error_code(body), expected);
    }
}
