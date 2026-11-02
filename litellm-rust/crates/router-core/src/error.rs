//! The router's errors: [`ConfigError`] when a router is built or changed, [`Error`] when a call
//! or a selection fails, and [`AttemptError`], the typed facts of one failed attempt.

use std::{any::Any, fmt, sync::Arc, time::Duration};

use crate::config::{Op, RoutingStrategy};

/// What a host cannot serve, found when the router is built rather than partway through a call.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum Unsupported {
    #[error("routing strategy `{0}` needs a host that can select")]
    Strategy(RoutingStrategy),
    #[error("routing plugin `{0}` needs a host that can select")]
    Plugin(String),
    #[error("the host cannot run operation `{op}` on provider `{provider}`")]
    Op { op: Op, provider: String },
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("the router config asks for something this host cannot serve")]
    Unsupported(#[from] Unsupported),
    #[error("deployment `{model_name}` has invalid litellm_params")]
    InvalidDeployment {
        model_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("deployment id `{id}` is configured more than once")]
    DuplicateId { id: String },
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the router config is invalid")]
    Config(#[from] ConfigError),
    #[error("no deployment serves model `{model}`")]
    UnknownModel { model: String },
    #[error("no deployment of `{model}` is available")]
    NoHealthyDeployments { model: String },
    #[error("the attempt failed")]
    Attempt(#[from] AttemptError),
    #[error("the routing cache failed")]
    Cache(#[from] litellm_cache::Error),
    #[error("the router was retired")]
    Retired,
    /// A blocked deployment named by its id (`model` is its litellm model).
    #[error("deployment model `{model}` is paused")]
    Paused { model: String },
    /// A pass-through pick named a deployment id without `use_in_pass_through`.
    #[error("deployment `{id}` does not support pass-through")]
    NotPassThrough { id: String },
    /// A pass-through pick of a group with healthy deployments, none of them pass-through.
    #[error("model `{model}` has no pass-through deployment")]
    NoPassThroughDeployments { model: String },
    /// Every deployment `model` resolved to is a pre-routing strategy marker (`auto_router/...`).
    #[error("every deployment of `{model}` is a strategy router marker")]
    OnlyMarkers { model: String },
    /// Every deployment `model` resolved to is split into PTU shares the caller holds none of.
    #[error("deployment `{model}` is reserved for the teams holding a PTU share of it")]
    PtuReserved { model: String },
    /// A proxy admin without a team named a public model name several teams use.
    #[error("model name `{model}` matches deployments from multiple teams")]
    MultipleTeams { model: String },
    /// Every deployment `model` resolved to is reserved for other teams by an active access
    /// window, the first of which ends at `until` (`HH:MM`) in `timezone`.
    #[error("deployment `{model}` is reserved for another team until {until} {timezone}")]
    Reserved {
        model: String,
        until: String,
        timezone: String,
    },
    /// The pre-call checks found no deployment whose context window fits the prompt; `detail`
    /// names each one dropped for it.
    #[error("no deployment of `{model}` has a context window large enough for this call")]
    ContextWindowExceeded { model: String, detail: String },
    /// The pre-call checks found no deployment of `model` that takes the requested `params`.
    #[error("no deployment of `{model}` supports {params:?}")]
    UnsupportedParams { model: String, params: Vec<String> },
    /// The routing plugins left none of the healthy deployments' models.
    #[error("no deployments left after routing-plugin filtering: {models:?}")]
    RoutingPluginsLeftNothing { models: Vec<String> },
    /// The request's `_router_weights` failed `validate_router_weights` when simple-shuffle picked.
    #[error("the request's router weights are invalid")]
    InvalidRouterWeights,
    /// A deployment's `litellm_params` `metric` (weight, rpm or tpm) is not a number.
    #[error("deployment {metric} must be numeric")]
    NonNumericWeight { metric: &'static str },
    /// A router with routing plugins was asked for a pick on the sync path, which never runs them.
    #[error("routing plugins only run on the async selection path")]
    RoutingPluginsNeedAsync,
    /// The stale routing-group member map (`_model_to_group`, kept as it was when a raw
    /// `routing_groups` write replaced the groups) names a group that is no longer configured:
    /// Python's `_get_routing_context` raises `KeyError(group)` for every pick of the name.
    #[error("routing group `{group}` is no longer configured")]
    MissingRoutingGroup { group: String },
    /// The router's `default_deployment` is not a value Python can copy for `model`
    /// (`_try_early_resolve_deployments_for_model_not_in_names` raises the copy's error).
    #[error("the default deployment cannot be served for `{model}`")]
    DefaultDeploymentRaises { model: String },
    /// A written model-name or team index names a position past the end of the model list:
    /// Python's lookup of `model` raises `IndexError`.
    #[error("a written index of `{model}` names a position past the model list")]
    IndexOutOfRange { model: String },
}

impl Error {
    /// The failed attempt behind this error, when a deployment was called.
    pub fn attempt(&self) -> Option<&AttemptError> {
        match self {
            Self::Attempt(attempt) => Some(attempt),
            Self::Config(_)
            | Self::UnknownModel { .. }
            | Self::NoHealthyDeployments { .. }
            | Self::Cache(_)
            | Self::Retired
            | Self::Paused { .. }
            | Self::NotPassThrough { .. }
            | Self::NoPassThroughDeployments { .. }
            | Self::OnlyMarkers { .. }
            | Self::PtuReserved { .. }
            | Self::MultipleTeams { .. }
            | Self::Reserved { .. }
            | Self::ContextWindowExceeded { .. }
            | Self::UnsupportedParams { .. }
            | Self::RoutingPluginsLeftNothing { .. }
            | Self::InvalidRouterWeights
            | Self::NonNumericWeight { .. }
            | Self::RoutingPluginsNeedAsync
            | Self::MissingRoutingGroup { .. }
            | Self::DefaultDeploymentRaises { .. }
            | Self::IndexOutOfRange { .. } => None,
        }
    }

    /// The Python exception a pick that failed with this error raises, as the retry loop
    /// decides by it (its class names, most specific first, and its status): `None` for an
    /// attempt's failure and for errors no pick raises.
    pub fn pick_failure(&self) -> Option<AttemptError> {
        const BAD_REQUEST: &[&str] = &["BadRequestError"];
        let (classes, status): (&[&str], Option<u16>) = match self {
            Self::UnknownModel { .. }
            | Self::OnlyMarkers { .. }
            | Self::PtuReserved { .. }
            | Self::NotPassThrough { .. }
            | Self::NoPassThroughDeployments { .. }
            | Self::MultipleTeams { .. }
            | Self::Reserved { .. } => (BAD_REQUEST, Some(400)),
            Self::ContextWindowExceeded { .. } => (
                &["ContextWindowExceededError", "BadRequestError"],
                Some(400),
            ),
            Self::UnsupportedParams { .. } => {
                (&["UnsupportedParamsError", "BadRequestError"], Some(400))
            }
            Self::Paused { .. } => (&["ServiceUnavailableError"], Some(503)),
            Self::NoHealthyDeployments { .. } => (&["RouterRateLimitError", "ValueError"], None),
            Self::RoutingPluginsLeftNothing { .. } | Self::RoutingPluginsNeedAsync => {
                (&["ValueError"], None)
            }
            Self::InvalidRouterWeights => (&["ValidationError", "ValueError"], None),
            Self::NonNumericWeight { .. } => (&["TypeError"], None),
            Self::MissingRoutingGroup { .. } => (&["KeyError", "LookupError"], None),
            Self::IndexOutOfRange { .. } => (&["IndexError", "LookupError"], None),
            Self::DefaultDeploymentRaises { .. } => (&["Exception"], None),
            Self::Attempt(_) | Self::Config(_) | Self::Cache(_) | Self::Retired => return None,
        };
        let mut failure = AttemptError::host_raised(classes);
        failure.status = status;
        Some(failure)
    }

    /// A copy of a pick's error, for the events that name it; `None` for any other error.
    pub fn pick_copy(&self) -> Option<Self> {
        Some(match self {
            Self::UnknownModel { model } => Self::UnknownModel {
                model: model.clone(),
            },
            Self::NoHealthyDeployments { model } => Self::NoHealthyDeployments {
                model: model.clone(),
            },
            Self::Paused { model } => Self::Paused {
                model: model.clone(),
            },
            Self::NotPassThrough { id } => Self::NotPassThrough { id: id.clone() },
            Self::NoPassThroughDeployments { model } => Self::NoPassThroughDeployments {
                model: model.clone(),
            },
            Self::OnlyMarkers { model } => Self::OnlyMarkers {
                model: model.clone(),
            },
            Self::PtuReserved { model } => Self::PtuReserved {
                model: model.clone(),
            },
            Self::MultipleTeams { model } => Self::MultipleTeams {
                model: model.clone(),
            },
            Self::Reserved {
                model,
                until,
                timezone,
            } => Self::Reserved {
                model: model.clone(),
                until: until.clone(),
                timezone: timezone.clone(),
            },
            Self::ContextWindowExceeded { model, detail } => Self::ContextWindowExceeded {
                model: model.clone(),
                detail: detail.clone(),
            },
            Self::UnsupportedParams { model, params } => Self::UnsupportedParams {
                model: model.clone(),
                params: params.clone(),
            },
            Self::RoutingPluginsLeftNothing { models } => Self::RoutingPluginsLeftNothing {
                models: models.clone(),
            },
            Self::InvalidRouterWeights => Self::InvalidRouterWeights,
            Self::NonNumericWeight { metric } => Self::NonNumericWeight { metric },
            Self::RoutingPluginsNeedAsync => Self::RoutingPluginsNeedAsync,
            Self::MissingRoutingGroup { group } => Self::MissingRoutingGroup {
                group: group.clone(),
            },
            Self::DefaultDeploymentRaises { model } => Self::DefaultDeploymentRaises {
                model: model.clone(),
            },
            Self::IndexOutOfRange { model } => Self::IndexOutOfRange {
                model: model.clone(),
            },
            Self::Attempt(_) | Self::Config(_) | Self::Cache(_) | Self::Retired => return None,
        })
    }
}

/// The class of a failed attempt, as litellm's exception mapping names it.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum FailureKind {
    #[strum(serialize = "RateLimitError")]
    RateLimit,
    #[strum(serialize = "ContextWindowExceededError")]
    ContextWindowExceeded,
    #[strum(serialize = "ContentPolicyViolationError")]
    ContentPolicyViolation,
    #[strum(serialize = "AuthenticationError")]
    Authentication,
    #[strum(serialize = "PermissionDeniedError")]
    PermissionDenied,
    #[strum(serialize = "NotFoundError")]
    NotFound,
    #[strum(serialize = "BadRequestError")]
    BadRequest,
    #[strum(serialize = "Timeout")]
    Timeout,
    #[strum(serialize = "APIConnectionError")]
    ApiConnection,
    #[strum(serialize = "InternalServerError")]
    InternalServer,
    #[strum(serialize = "ServiceUnavailableError")]
    ServiceUnavailable,
    #[strum(serialize = "BadGatewayError")]
    BadGateway,
    /// A guardrail blocked the request: never retried, never a fallback.
    #[strum(serialize = "GuardrailIntervention")]
    GuardrailIntervention,
    /// litellm rejected the request before calling the deployment (`MalformedRequestFieldError`,
    /// `UnsupportedEndpointError`): never cools the deployment down.
    #[strum(serialize = "Declined")]
    Declined,
    /// The caller's own credential failed, not the deployment's.
    #[strum(serialize = "CallerCredentialError")]
    CallerCredential,
    #[strum(serialize = "APIError")]
    Other,
}

/// An opaque value the host hands through the router unchanged, such as the original Python
/// exception or response object.
#[derive(Clone)]
pub struct Payload(pub Arc<dyn Any + Send + Sync>);

impl fmt::Debug for Payload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Payload(..)")
    }
}

/// One failed attempt, as typed facts the retry, fallback and cooldown rules decide by.
#[derive(Clone, Debug, thiserror::Error)]
#[error("the deployment call failed with {kind}")]
pub struct AttemptError {
    pub kind: FailureKind,
    pub status: Option<u16>,
    /// The provider's `retry-after` (or `retry-after-ms`) header.
    pub retry_after: Option<Duration>,
    /// A deployment-level `num_retries` the failure carries (Python's `exception.num_retries`).
    pub num_retries: Option<u32>,
    /// The provider's error text, kept only for the cooldown record Python writes
    /// (`exception_received`, masked past 50 characters).
    pub detail: Option<Arc<str>>,
    /// The exception's class names, most specific first (its `__mro__`), for the rules Python
    /// decides by `isinstance`. Empty when the host reports none; the kind then stands in.
    pub classes: Vec<Arc<str>>,
    /// The host already recorded what this failure does to the deployment's health (a routing
    /// pre-call check's failure, which Python's attempt handles itself: a rate limit cools the
    /// deployment down, anything else records nothing), so the core records nothing.
    pub recorded: bool,
    /// The exception names its provider (`llm_provider` is set), which marks a provider's own
    /// 400 for `fallback_on_unsupported_params`.
    pub names_provider: bool,
    pub payload: Option<Payload>,
}

impl AttemptError {
    pub fn new(kind: FailureKind, status: Option<u16>) -> Self {
        Self {
            kind,
            status,
            retry_after: None,
            num_retries: None,
            detail: None,
            classes: Vec::new(),
            recorded: false,
            names_provider: false,
            payload: None,
        }
    }

    /// The exception's class names, most specific first: the host's, else the kind's.
    pub fn class_names(&self) -> Vec<&str> {
        if self.classes.is_empty() {
            return crate::retry::kind_classes(self.kind).to_vec();
        }
        self.classes.iter().map(AsRef::as_ref).collect()
    }

    /// `isinstance(error, <class named name>)`.
    pub fn is_a(&self, name: &str) -> bool {
        self.class_names().contains(&name)
    }

    #[must_use]
    pub fn with_retry_after(self, retry_after: Duration) -> Self {
        Self {
            retry_after: Some(retry_after),
            ..self
        }
    }

    /// A plain Python exception the host raised outside any attempt (a `KeyError` from a stale
    /// routing-group lookup, an error a retry policy lookup raised): no status, no provider,
    /// nothing recorded against a deployment. `classes` are its class names, most specific first.
    pub fn host_raised(classes: &[&str]) -> Self {
        Self {
            classes: classes.iter().map(|class| (*class).into()).collect(),
            recorded: true,
            ..Self::new(FailureKind::Other, None)
        }
    }

    /// The status Python's cooldown rules read: the provider's, else 500.
    pub fn cooldown_status(&self) -> u16 {
        self.status.unwrap_or(500)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::rate_limit("RateLimitError", FailureKind::RateLimit)]
    #[case::context_window("ContextWindowExceededError", FailureKind::ContextWindowExceeded)]
    #[case::caller_credential("CallerCredentialError", FailureKind::CallerCredential)]
    #[case::other("APIError", FailureKind::Other)]
    fn failure_kind_round_trips_its_exception_name(#[case] wire: &str, #[case] kind: FailureKind) {
        assert_eq!(wire.parse::<FailureKind>().unwrap(), kind);
        assert_eq!(<&'static str>::from(kind), wire);
    }

    #[rstest]
    #[case::empty("")]
    #[case::lowercase("ratelimiterror")]
    fn failure_kind_rejects_unknown_names(#[case] wire: &str) {
        assert!(wire.parse::<FailureKind>().is_err());
    }
}
