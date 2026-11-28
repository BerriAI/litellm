//! How Python's logging reads the exception a call failed with into the payload's
//! `error_information` (`StandardLoggingPayloadSetup.get_error_information` in
//! `litellm/litellm_core_utils/litellm_logging.py`): the exception classes a Rust gateway call can
//! fail as, their rate-limit facts, and the stable `normalized_error` code of
//! `litellm/litellm_core_utils/error_normalization.py`.

use std::sync::LazyLock;

use regex::Regex;

use crate::ErrorInformation;

/// A Python exception class a call can fail as, named as `logged_class_name` logs it.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum ExceptionClass {
    #[strum(serialize = "BadRequestError")]
    BadRequest,
    #[strum(serialize = "ContextWindowExceededError")]
    ContextWindowExceeded,
    #[strum(serialize = "ContentPolicyViolationError")]
    ContentPolicyViolation,
    #[strum(serialize = "UnsupportedParamsError")]
    UnsupportedParams,
    #[strum(serialize = "EndpointNotSupportedByModelError")]
    EndpointNotSupportedByModel,
    #[strum(serialize = "AuthenticationError")]
    Authentication,
    #[strum(serialize = "PermissionDeniedError")]
    PermissionDenied,
    #[strum(serialize = "NotFoundError")]
    NotFound,
    #[strum(serialize = "Timeout")]
    Timeout,
    #[strum(serialize = "RateLimitError")]
    RateLimit,
    #[strum(serialize = "InternalServerError")]
    InternalServer,
    #[strum(serialize = "BadGatewayError")]
    BadGateway,
    #[strum(serialize = "ServiceUnavailableError")]
    ServiceUnavailable,
    #[strum(serialize = "APIConnectionError")]
    ApiConnection,
    #[strum(serialize = "APIError")]
    Api,
    /// `litellm.proxy.common_utils.proxy_rate_limit_error.ProxyRateLimitError`, the proxy's own
    /// limiters' 429: both a `fastapi.HTTPException` and a `RateLimitError`.
    #[strum(serialize = "ProxyRateLimitError")]
    ProxyRateLimit,
    /// `litellm.types.router.RouterRateLimitError`, a `ValueError`.
    #[strum(serialize = "RouterRateLimitError")]
    RouterRateLimit,
    #[strum(serialize = "ProxyException")]
    ProxyException,
    #[strum(serialize = "ModelAccessDeniedProxyException")]
    ModelAccessDeniedProxyException,
    #[strum(serialize = "HTTPException")]
    HttpException,
    /// `litellm.proxy.route_llm_request.ProxyModelNotFoundError`, an `HTTPException`.
    #[strum(serialize = "ProxyModelNotFoundError")]
    ProxyModelNotFound,
    #[strum(serialize = "ValueError")]
    ValueError,
    /// Pydantic's `ValidationError`, a `ValueError`.
    #[strum(serialize = "ValidationError")]
    ValidationError,
    #[strum(serialize = "TypeError")]
    TypeError,
    #[strum(serialize = "KeyError")]
    KeyError,
    #[strum(serialize = "IndexError")]
    IndexError,
    #[strum(serialize = "Exception")]
    Exception,
}

impl ExceptionClass {
    /// The classes `isinstance` finds this one to be, itself first: the part of its `__mro__`
    /// that `normalize_error` and `is_expected_client_error` read.
    fn lineage(self) -> &'static [Self] {
        use ExceptionClass as C;
        match self {
            C::ContextWindowExceeded => &[C::ContextWindowExceeded, C::BadRequest],
            C::ContentPolicyViolation => &[C::ContentPolicyViolation, C::BadRequest],
            C::UnsupportedParams => &[C::UnsupportedParams, C::BadRequest],
            C::EndpointNotSupportedByModel => &[C::EndpointNotSupportedByModel, C::BadRequest],
            C::ProxyRateLimit => &[C::ProxyRateLimit, C::HttpException, C::RateLimit],
            C::RouterRateLimit => &[C::RouterRateLimit, C::ValueError],
            C::ModelAccessDeniedProxyException => {
                &[C::ModelAccessDeniedProxyException, C::ProxyException]
            }
            C::ProxyModelNotFound => &[C::ProxyModelNotFound, C::HttpException],
            C::ValidationError => &[C::ValidationError, C::ValueError],
            C::BadRequest => &[C::BadRequest],
            C::Authentication => &[C::Authentication],
            C::PermissionDenied => &[C::PermissionDenied],
            C::NotFound => &[C::NotFound],
            C::Timeout => &[C::Timeout],
            C::RateLimit => &[C::RateLimit],
            C::InternalServer => &[C::InternalServer],
            C::BadGateway => &[C::BadGateway],
            C::ServiceUnavailable => &[C::ServiceUnavailable],
            C::ApiConnection => &[C::ApiConnection],
            C::Api => &[C::Api],
            C::ProxyException => &[C::ProxyException],
            C::HttpException => &[C::HttpException],
            C::ValueError => &[C::ValueError],
            C::TypeError => &[C::TypeError],
            C::KeyError => &[C::KeyError],
            C::IndexError => &[C::IndexError],
            C::Exception => &[C::Exception],
        }
    }

    /// `isinstance(error, other)`.
    pub fn is_a(self, other: Self) -> bool {
        self.lineage().contains(&other)
    }

    /// The `.message` the class builds from the message it is raised with: every litellm
    /// exception puts `litellm.{Class}: ` before it, a `ContextWindowExceededError` its own before
    /// the `BadRequestError` one its base put, and a `ProxyRateLimitError` the `RateLimitError`
    /// one. Any other class keeps the message as it is.
    pub fn message(self, raised_with: &str) -> String {
        use ExceptionClass as C;
        match self {
            C::ContextWindowExceeded => {
                format!(
                    "litellm.ContextWindowExceededError: litellm.BadRequestError: {raised_with}"
                )
            }
            C::ProxyRateLimit => format!("litellm.RateLimitError: {raised_with}"),
            C::BadRequest
            | C::ContentPolicyViolation
            | C::UnsupportedParams
            | C::Authentication
            | C::PermissionDenied
            | C::NotFound
            | C::Timeout
            | C::RateLimit
            | C::InternalServer
            | C::BadGateway
            | C::ServiceUnavailable
            | C::ApiConnection
            | C::Api => format!("litellm.{}: {raised_with}", <&str>::from(self)),
            C::EndpointNotSupportedByModel
            | C::RouterRateLimit
            | C::ProxyException
            | C::ModelAccessDeniedProxyException
            | C::HttpException
            | C::ProxyModelNotFound
            | C::ValueError
            | C::ValidationError
            | C::TypeError
            | C::KeyError
            | C::IndexError
            | C::Exception => raised_with.to_owned(),
        }
    }

    /// The class Python names `name`, `Exception` for a class this table does not hold.
    pub fn named(name: &str) -> Self {
        name.parse().unwrap_or(Self::Exception)
    }
}

/// `litellm.exceptions.RateLimitErrorCategory`: who refused the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
pub enum RateLimitCategory {
    #[strum(serialize = "vendor_rate_limit")]
    VendorRateLimit,
    #[strum(serialize = "vendor_batch_rate_limit")]
    VendorBatchRateLimit,
    #[strum(serialize = "litellm_rate_limit")]
    LitellmRateLimit,
    #[strum(serialize = "litellm_batch_rate_limit")]
    LitellmBatchRateLimit,
}

impl RateLimitCategory {
    /// One of litellm's own limiters refused the call (`_is_litellm_limit_rejection`).
    fn is_litellm_limit(self) -> bool {
        matches!(self, Self::LitellmRateLimit | Self::LitellmBatchRateLimit)
    }
}

/// `litellm.exceptions.RateLimitType`: which limit refused the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
pub enum RateLimitType {
    #[strum(serialize = "requests")]
    Requests,
    #[strum(serialize = "tokens")]
    Tokens,
    #[strum(serialize = "concurrent_requests")]
    ConcurrentRequests,
    #[strum(serialize = "budget")]
    Budget,
    #[strum(serialize = "max_iterations")]
    MaxIterations,
}

impl RateLimitType {
    /// `map_v3_rate_limit_type`: the type a `parallel_request_limiter_v3` status names, as its
    /// `rate_limit_type` response header carries it.
    pub fn from_v3(value: &str) -> Option<Self> {
        match value {
            "tokens" => Some(Self::Tokens),
            "max_parallel_requests" => Some(Self::ConcurrentRequests),
            "requests" => Some(Self::Requests),
            _ => None,
        }
    }
}

/// The stable codes of `error_normalization.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub enum NormalizedError {
    #[strum(serialize = "429_RATE_LIMIT_EXCEEDED")]
    RateLimitExceeded,
    #[strum(serialize = "429_BUDGET_EXCEEDED")]
    BudgetExceeded,
    #[strum(serialize = "429_NO_HEALTHY_DEPLOYMENTS")]
    NoHealthyDeployments,
    #[strum(serialize = "401_AUTHENTICATION_FAILED")]
    AuthenticationFailed,
    #[strum(serialize = "403_MODEL_ACCESS_DENIED")]
    ModelAccessDenied,
    #[strum(serialize = "403_PERMISSION_DENIED")]
    PermissionDenied,
    #[strum(serialize = "400_MISSING_REQUIRED_PARAMETER")]
    MissingRequiredParameter,
    #[strum(serialize = "400_INVALID_PARAMETER_VALUE")]
    InvalidParameterValue,
    #[strum(serialize = "400_CONTEXT_WINDOW_EXCEEDED")]
    ContextWindowExceeded,
    #[strum(serialize = "400_CONTENT_POLICY_VIOLATION")]
    ContentPolicyViolation,
    #[strum(serialize = "400_INVALID_REQUEST")]
    InvalidRequest,
    #[strum(serialize = "404_RESOURCE_NOT_FOUND")]
    ResourceNotFound,
    #[strum(serialize = "408_UPSTREAM_TIMEOUT")]
    UpstreamTimeout,
    #[strum(serialize = "500_PROVIDER_CONNECTION_ERROR")]
    ProviderConnectionError,
    #[strum(serialize = "503_PROVIDER_OVERLOADED")]
    ProviderOverloaded,
    #[strum(serialize = "500_PROVIDER_INTERNAL_ERROR")]
    ProviderInternalError,
    #[strum(serialize = "500_ROUTER_NO_FALLBACK")]
    RouterNoFallback,
    #[strum(serialize = "500_ROUTER_FALLBACK_FAILURE")]
    RouterFallbackFailure,
    #[strum(serialize = "500_UPSTREAM_PASSTHROUGH")]
    UpstreamPassthrough,
    #[strum(serialize = "500_UNSUPPORTED_OPERATION")]
    UnsupportedOperation,
    #[strum(serialize = "500_INTERNAL_STATE_ERROR")]
    InternalStateError,
    #[strum(serialize = "UNCLASSIFIED")]
    Unclassified,
}

fn pattern(source: &str) -> Regex {
    Regex::new(&format!("(?i){source}")).expect("normalization patterns compile")
}

/// `_MESSAGE_PATTERNS`, in order.
static MESSAGE_PATTERNS: LazyLock<Vec<(Regex, NormalizedError)>> = LazyLock::new(|| {
    use NormalizedError as N;
    vec![
        (
            pattern("upstream passthrough request failed"),
            N::UpstreamPassthrough,
        ),
        (
            pattern("budget has been exceeded|max budget|crossed budget"),
            N::BudgetExceeded,
        ),
        (
            pattern("no healthy deployments?|no deployments available"),
            N::NoHealthyDeployments,
        ),
        (
            pattern("not allowed to access model due to tags configuration"),
            N::ModelAccessDenied,
        ),
        (
            pattern("is not supported for provider|not implemented"),
            N::UnsupportedOperation,
        ),
        (
            pattern(
                r"context window|context length|(prompt|input) is too long|tokens? ?> ?\d+ ?maximum",
            ),
            N::ContextWindowExceeded,
        ),
        (
            pattern("missing required parameter|field required"),
            N::MissingRequiredParameter,
        ),
        (
            pattern("overloaded|unable to process your request"),
            N::ProviderOverloaded,
        ),
        (
            pattern(
                "connection error|APIConnectionError|TransferEncodingError|payload is not completed|connection reset|peer closed connection|incomplete chunked read",
            ),
            N::ProviderConnectionError,
        ),
        (pattern("timed? ?out"), N::UpstreamTimeout),
    ]
});

/// `_ROUTER_WRAPPER_PATTERNS`, in order.
static ROUTER_WRAPPER_PATTERNS: LazyLock<Vec<(Regex, NormalizedError)>> = LazyLock::new(|| {
    vec![
        (
            pattern("no fallback model group found"),
            NormalizedError::RouterNoFallback,
        ),
        (
            pattern("error doing the fallback|MidStreamFallbackError"),
            NormalizedError::RouterFallbackFailure,
        ),
    ]
});

/// `_CLASS_CODE_TABLE`, in order, for the classes a Rust call can fail as.
const CLASS_CODES: &[(ExceptionClass, NormalizedError)] = &[
    (
        ExceptionClass::Authentication,
        NormalizedError::AuthenticationFailed,
    ),
    (
        ExceptionClass::PermissionDenied,
        NormalizedError::PermissionDenied,
    ),
    (
        ExceptionClass::ContextWindowExceeded,
        NormalizedError::ContextWindowExceeded,
    ),
    (
        ExceptionClass::ContentPolicyViolation,
        NormalizedError::ContentPolicyViolation,
    ),
    (
        ExceptionClass::UnsupportedParams,
        NormalizedError::InvalidParameterValue,
    ),
    (ExceptionClass::NotFound, NormalizedError::ResourceNotFound),
    (ExceptionClass::Timeout, NormalizedError::UpstreamTimeout),
    (
        ExceptionClass::ApiConnection,
        NormalizedError::ProviderConnectionError,
    ),
    (
        ExceptionClass::ServiceUnavailable,
        NormalizedError::ProviderOverloaded,
    ),
    (
        ExceptionClass::InternalServer,
        NormalizedError::ProviderInternalError,
    ),
    (
        ExceptionClass::BadGateway,
        NormalizedError::ProviderInternalError,
    ),
    (ExceptionClass::BadRequest, NormalizedError::InvalidRequest),
];

/// `_INTERNAL_STATE_EXCEPTIONS`, those of them a Rust call can fail as.
const INTERNAL_STATE: &[ExceptionClass] = &[
    ExceptionClass::TypeError,
    ExceptionClass::KeyError,
    ExceptionClass::IndexError,
];

/// `_PROXY_ERROR_TYPE_MAP`: a `ProxyException`'s `type`.
fn by_proxy_type(proxy_type: &str) -> Option<NormalizedError> {
    use NormalizedError as N;
    Some(match proxy_type {
        "budget_exceeded" => N::BudgetExceeded,
        "auth_error" | "expired_key" | "token_not_found_in_db" | "auth_provider_unavailable" => {
            N::AuthenticationFailed
        }
        "key_model_access_denied"
        | "team_model_access_denied"
        | "user_model_access_denied"
        | "org_model_access_denied"
        | "project_model_access_denied"
        | "agent_model_access_denied"
        | "customer_model_access_denied" => N::ModelAccessDenied,
        "key_vector_store_access_denied"
        | "team_vector_store_access_denied"
        | "org_vector_store_access_denied"
        | "user_vector_store_access_denied"
        | "key_search_tool_access_denied"
        | "team_search_tool_access_denied"
        | "user_search_tool_access_denied"
        | "tool_access_denied"
        | "team_member_permission_error" => N::PermissionDenied,
        "not_found_error" => N::ResourceNotFound,
        _ => return None,
    })
}

/// `_STATUS_CODE_MAP`, else `UNCLASSIFIED`.
fn by_status(status: &str) -> NormalizedError {
    use NormalizedError as N;
    match status {
        "400" => N::InvalidRequest,
        "401" => N::AuthenticationFailed,
        "403" => N::PermissionDenied,
        "404" => N::ResourceNotFound,
        "408" | "504" => N::UpstreamTimeout,
        "422" => N::InvalidParameterValue,
        "429" => N::RateLimitExceeded,
        "500" | "502" => N::ProviderInternalError,
        "503" => N::ProviderOverloaded,
        _ => N::Unclassified,
    }
}

/// `_exceeded_before_budget`: some line names `exceeded` and, after it, `budget`.
fn exceeded_before_budget(message: &str) -> bool {
    const EXCEEDED: &str = "exceeded";
    message.to_lowercase().split('\n').any(|line| {
        line.find(EXCEEDED)
            .is_some_and(|start| line[start + EXCEEDED.len()..].contains("budget"))
    })
}

fn first_pattern(patterns: &[(Regex, NormalizedError)], message: &str) -> Option<NormalizedError> {
    patterns
        .iter()
        .find(|(pattern, _)| pattern.is_match(message))
        .map(|(_, code)| *code)
}

/// `PROVIDER_REQUEST_ID_HEADERS` in `litellm/constants.py`, in the order they are preferred.
const PROVIDER_REQUEST_ID_HEADERS: &[&str] = &[
    "x-amzn-requestid",
    "x-request-id",
    "request-id",
    "x-ms-request-id",
    "apim-request-id",
    "x-goog-request-id",
    "cf-ray",
];

/// `_get_provider_request_id`: the first non-empty value of the most preferred request id
/// header the provider's response carried, its name matched in any case.
pub fn provider_request_id(headers: &[(String, String)]) -> Option<String> {
    PROVIDER_REQUEST_ID_HEADERS.iter().find_map(|expected| {
        headers
            .iter()
            .find(|(name, value)| name.eq_ignore_ascii_case(expected) && !value.is_empty())
            .map(|(_, value)| value.clone())
    })
}

/// What `get_error_information` reads off the exception a call failed with, before it redacts
/// the message and the traceback.
#[derive(Clone, Debug, PartialEq)]
pub struct LoggedException {
    pub class: ExceptionClass,
    /// `str()` of the exception's `code`, else of its `status_code`, empty when it has neither.
    pub code: String,
    /// `getattr(exception, "llm_provider", "")`.
    pub llm_provider: String,
    /// The exception's `.message`, else `str()` of it.
    pub message: String,
    pub provider_request_id: Option<String>,
    pub rate_limit_category: Option<RateLimitCategory>,
    pub rate_limit_type: Option<RateLimitType>,
    /// A `ProxyException`'s `type`, which `normalize_error` reads first.
    pub proxy_type: Option<String>,
    /// Where the failure came from, the Rust stand-in for the Python traceback: the error's
    /// cause chain. It is logged only for an error that is not an expected client error, as
    /// Python formats a traceback only then (`log_client_error_tracebacks` is off, and a config
    /// that turns it on is served by Python).
    pub traceback: String,
}

impl LoggedException {
    /// `normalize_error(exception, error_code, error_message)`.
    pub fn normalized_error(&self) -> NormalizedError {
        if let Some(code) = self.proxy_type.as_deref().and_then(by_proxy_type) {
            return code;
        }
        let by_message = match exceeded_before_budget(&self.message) {
            true => Some(NormalizedError::BudgetExceeded),
            false => first_pattern(&MESSAGE_PATTERNS, &self.message),
        };
        if let Some(code) = by_message {
            return code;
        }
        if self.code == "529" {
            return NormalizedError::ProviderOverloaded;
        }
        if let Some(code) = self.by_class() {
            return code;
        }
        first_pattern(&ROUTER_WRAPPER_PATTERNS, &self.message)
            .unwrap_or_else(|| by_status(&self.code))
    }

    /// `_classify_by_class`.
    fn by_class(&self) -> Option<NormalizedError> {
        if self.class.is_a(ExceptionClass::RateLimit) {
            return Some(match self.rate_limit_type {
                Some(RateLimitType::Budget) => NormalizedError::BudgetExceeded,
                _ => NormalizedError::RateLimitExceeded,
            });
        }
        CLASS_CODES
            .iter()
            .find(|(class, _)| self.class.is_a(*class))
            .map(|(_, code)| *code)
            .or_else(|| {
                INTERNAL_STATE
                    .iter()
                    .any(|class| self.class.is_a(*class))
                    .then_some(NormalizedError::InternalStateError)
            })
    }

    /// `is_expected_client_error`: the proxy itself refused the call with a 4xx. A failure a
    /// provider raised (it names its `llm_provider`) never is, unless the proxy raised it as one
    /// of its own refusals (an `HTTPException`, or a litellm limiter's category).
    pub fn is_expected_client_error(&self) -> bool {
        let proxy_rejection = self.class.is_a(ExceptionClass::HttpException)
            || self
                .rate_limit_category
                .is_some_and(RateLimitCategory::is_litellm_limit);
        if !proxy_rejection && !self.llm_provider.is_empty() {
            return false;
        }
        self.code
            .parse::<i64>()
            .is_ok_and(|status| (400..500).contains(&status))
    }

    /// The payload's `error_information`, with `redact` (Python's `redact_string`) over the
    /// message and the traceback.
    pub fn error_information(&self, redact: impl Fn(&str) -> String) -> ErrorInformation {
        let traceback = match self.is_expected_client_error() {
            true => "",
            false => self.traceback.as_str(),
        };
        ErrorInformation {
            error_code: Some(self.code.clone()),
            error_class: Some(<&str>::from(self.class).to_owned()),
            llm_provider: Some(self.llm_provider.clone()),
            traceback: Some(redact(traceback)),
            error_message: Some(redact(&self.message)),
            error_provider_request_id: self.provider_request_id.clone(),
            error_rate_limit_category: self.rate_limit_category.map(|c| <&str>::from(c).to_owned()),
            error_rate_limit_type: self.rate_limit_type.map(|t| <&str>::from(t).to_owned()),
            error_budget_entity_type: None,
            error_budget_entity_id: None,
            error_budget_limit: None,
            error_budget_spend: None,
            error_cooldown_deployments: None,
            normalized_error: Some(<&str>::from(self.normalized_error()).to_owned()),
        }
    }
}

/// The words Python logs for a client that left before its call ended.
pub const CLIENT_DISCONNECTED_MESSAGE: &str = "Client disconnected the request";

impl ErrorInformation {
    /// `_CLIENT_DISCONNECTED_ERROR_INFORMATION`, which
    /// `get_error_information_for_logging_payload` logs in place of the exception's for a call
    /// whose client disconnected: a 499 and nothing else.
    pub fn client_disconnected() -> Self {
        Self {
            error_code: Some("499".to_owned()),
            error_class: Some("ClientDisconnected".to_owned()),
            error_message: Some(CLIENT_DISCONNECTED_MESSAGE.to_owned()),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn exception(class: ExceptionClass, code: &str, message: &str) -> LoggedException {
        LoggedException {
            class,
            code: code.to_owned(),
            llm_provider: String::new(),
            message: message.to_owned(),
            provider_request_id: None,
            rate_limit_category: None,
            rate_limit_type: None,
            proxy_type: None,
            traceback: "the cause".to_owned(),
        }
    }

    fn provider(class: ExceptionClass, code: &str, message: &str) -> LoggedException {
        LoggedException {
            llm_provider: "anthropic".to_owned(),
            ..exception(class, code, message)
        }
    }

    fn limited(rate_limit_type: Option<RateLimitType>) -> LoggedException {
        LoggedException {
            llm_provider: "anthropic".to_owned(),
            rate_limit_category: Some(RateLimitCategory::LitellmRateLimit),
            rate_limit_type,
            ..exception(
                ExceptionClass::ProxyRateLimit,
                "429",
                "litellm.RateLimitError: Rate limit exceeded for api_key: hash. Limit type: requests.",
            )
        }
    }

    #[rstest]
    #[case::proxy_type_first(
        LoggedException {
            proxy_type: Some("key_model_access_denied".to_owned()),
            ..exception(ExceptionClass::ModelAccessDeniedProxyException, "403", "context window")
        },
        NormalizedError::ModelAccessDenied
    )]
    #[case::budget_split_across_lines(
        exception(ExceptionClass::Exception, "", "Exceeded the key's\nbudget"),
        NormalizedError::Unclassified
    )]
    #[case::budget_on_one_line(
        exception(ExceptionClass::Exception, "", "spend exceeded the budget"),
        NormalizedError::BudgetExceeded
    )]
    #[case::no_deployments(
        exception(
            ExceptionClass::RouterRateLimit,
            "429",
            "No deployments available for selected model, Try again in 5 seconds."
        ),
        NormalizedError::NoHealthyDeployments
    )]
    #[case::anthropic_prompt_too_long(
        provider(
            ExceptionClass::ContextWindowExceeded,
            "400",
            "litellm.ContextWindowExceededError: litellm.BadRequestError: AnthropicException - prompt is too long"
        ),
        NormalizedError::ContextWindowExceeded
    )]
    #[case::overloaded_by_message(
        provider(
            ExceptionClass::InternalServer,
            "500",
            "litellm.InternalServerError: AnthropicException - {\"type\":\"overloaded_error\"}"
        ),
        NormalizedError::ProviderOverloaded
    )]
    #[case::status_529(
        provider(ExceptionClass::Api, "529", "litellm.APIError: busy"),
        NormalizedError::ProviderOverloaded
    )]
    #[case::vendor_rate_limit(
        provider(ExceptionClass::RateLimit, "429", "litellm.RateLimitError: slow down"),
        NormalizedError::RateLimitExceeded
    )]
    #[case::limiter_rate_limit(
        limited(Some(RateLimitType::Requests)),
        NormalizedError::RateLimitExceeded
    )]
    #[case::limiter_budget(limited(Some(RateLimitType::Budget)), NormalizedError::BudgetExceeded)]
    #[case::subclass_takes_its_base_code(
        provider(
            ExceptionClass::EndpointNotSupportedByModel,
            "400",
            "litellm.BadRequestError: nope"
        ),
        NormalizedError::InvalidRequest
    )]
    #[case::internal_state(
        exception(ExceptionClass::KeyError, "", "'group'"),
        NormalizedError::InternalStateError
    )]
    #[case::router_wrapper(
        exception(
            ExceptionClass::Exception,
            "",
            "No fallback model group found for original model_group=x"
        ),
        NormalizedError::RouterNoFallback
    )]
    #[case::by_status(
        exception(
            ExceptionClass::ProxyModelNotFound,
            "400",
            "400: {'error': 'Invalid model name passed in'}"
        ),
        NormalizedError::InvalidRequest
    )]
    #[case::unclassified(
        exception(ExceptionClass::Exception, "", "boom"),
        NormalizedError::Unclassified
    )]
    fn normalizes_as_python_clusters_failures(
        #[case] exception: LoggedException,
        #[case] expected: NormalizedError,
    ) {
        assert_eq!(exception.normalized_error(), expected);
    }

    #[rstest]
    #[case::provider_4xx(provider(ExceptionClass::BadRequest, "400", "m"), false)]
    #[case::limiter_refusal(limited(None), true)]
    #[case::router_refusal(exception(ExceptionClass::RouterRateLimit, "429", "m"), true)]
    #[case::unknown_model(exception(ExceptionClass::ProxyModelNotFound, "400", "m"), true)]
    #[case::server_error(exception(ExceptionClass::Exception, "500", "m"), false)]
    #[case::no_status(exception(ExceptionClass::Exception, "", "m"), false)]
    fn only_the_proxys_own_4xx_is_an_expected_client_error(
        #[case] exception: LoggedException,
        #[case] expected: bool,
    ) {
        assert_eq!(exception.is_expected_client_error(), expected);
    }

    #[rstest]
    #[case::expected_client_error_drops_its_traceback(limited(None), "")]
    #[case::provider_error_keeps_it(provider(ExceptionClass::BadRequest, "400", "m"), "the cause")]
    fn the_traceback_is_logged_as_python_formats_one(
        #[case] exception: LoggedException,
        #[case] traceback: &str,
    ) {
        let information = exception.error_information(str::to_owned);
        assert_eq!(information.traceback.as_deref(), Some(traceback));
    }

    #[rstest]
    fn error_information_carries_every_fact_redacted() {
        let exception = LoggedException {
            provider_request_id: Some("req_1".to_owned()),
            rate_limit_category: Some(RateLimitCategory::VendorRateLimit),
            ..provider(
                ExceptionClass::RateLimit,
                "429",
                "litellm.RateLimitError: sk-secret",
            )
        };
        let information =
            exception.error_information(|text| text.replace("sk-secret", "[REDACTED]"));
        assert_eq!(
            information,
            ErrorInformation {
                error_code: Some("429".to_owned()),
                error_class: Some("RateLimitError".to_owned()),
                llm_provider: Some("anthropic".to_owned()),
                traceback: Some("the cause".to_owned()),
                error_message: Some("litellm.RateLimitError: [REDACTED]".to_owned()),
                error_provider_request_id: Some("req_1".to_owned()),
                error_rate_limit_category: Some("vendor_rate_limit".to_owned()),
                error_rate_limit_type: None,
                normalized_error: Some("429_RATE_LIMIT_EXCEEDED".to_owned()),
                ..ErrorInformation::default()
            }
        );
    }

    #[rstest]
    #[case::preferred_name_wins(
        vec![("Request-Id".to_owned(), "b".to_owned()), ("x-request-id".to_owned(), "a".to_owned())],
        Some("a")
    )]
    #[case::empty_value_is_skipped(
        vec![("x-request-id".to_owned(), String::new()), ("request-id".to_owned(), "b".to_owned())],
        Some("b")
    )]
    #[case::none(vec![("content-type".to_owned(), "json".to_owned())], None)]
    fn the_provider_request_id_is_the_most_preferred_header(
        #[case] headers: Vec<(String, String)>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(provider_request_id(&headers).as_deref(), expected);
    }

    #[rstest]
    #[case::tokens("tokens", Some(RateLimitType::Tokens))]
    #[case::parallel("max_parallel_requests", Some(RateLimitType::ConcurrentRequests))]
    #[case::requests("requests", Some(RateLimitType::Requests))]
    #[case::unknown("budget", None)]
    fn v3_limit_types_map_to_the_public_ones(
        #[case] value: &str,
        #[case] expected: Option<RateLimitType>,
    ) {
        assert_eq!(RateLimitType::from_v3(value), expected);
    }

    #[rstest]
    #[case::litellm_class(ExceptionClass::RateLimit, "litellm.RateLimitError: slow")]
    #[case::context_window(
        ExceptionClass::ContextWindowExceeded,
        "litellm.ContextWindowExceededError: litellm.BadRequestError: slow"
    )]
    #[case::proxy_limiter(ExceptionClass::ProxyRateLimit, "litellm.RateLimitError: slow")]
    #[case::router(ExceptionClass::RouterRateLimit, "slow")]
    fn a_litellm_class_prefixes_its_message(#[case] class: ExceptionClass, #[case] expected: &str) {
        assert_eq!(class.message("slow"), expected);
    }

    #[rstest]
    #[case::known("RouterRateLimitError", ExceptionClass::RouterRateLimit)]
    #[case::unknown("SomethingElse", ExceptionClass::Exception)]
    fn a_class_is_found_by_its_python_name(#[case] name: &str, #[case] expected: ExceptionClass) {
        assert_eq!(ExceptionClass::named(name), expected);
    }

    #[rstest]
    fn client_disconnect_logs_only_its_three_facts() {
        assert_eq!(
            ErrorInformation::client_disconnected(),
            ErrorInformation {
                error_code: Some("499".to_owned()),
                error_class: Some("ClientDisconnected".to_owned()),
                error_message: Some("Client disconnected the request".to_owned()),
                ..ErrorInformation::default()
            }
        );
    }
}
