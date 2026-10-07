use serde::Deserialize;

use crate::pyrepr::PyNumber;

/// The LiteLLM exception classes routing decisions branch on.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Hash)]
pub enum ExceptionClass {
    BadRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    Timeout,
    RateLimit,
    ContextWindowExceeded,
    ContentPolicyViolation,
    ServiceUnavailable,
    InternalServer,
    BadGateway,
    ApiConnection,
}

/// What the host learned about an exception an attempt raised. The host computes the facts
/// that depend on Python objects (header parsing, logging state); routing reads only these.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Classified {
    /// The exception's routing-relevant classes, most specific first (`type(e).__mro__` order).
    pub classes: Vec<ExceptionClass>,
    pub type_name: String,
    /// `str(e)`, read for cooldown decisions and the stored cooldown reason.
    pub message: String,
    pub status_code: Option<i64>,
    /// `e.num_retries` after the attempt stamped the deployment's own `num_retries`.
    pub num_retries: Option<u32>,
    /// Retry-After seconds from the headers the retry sleep reads, `-1` when absent.
    pub sleep_retry_after: i64,
    /// Retry-After seconds from the headers the cooldown reads, when present.
    pub cooldown_retry_after: Option<i64>,
    pub guardrail_intervention: bool,
    /// litellm raised it, so its failure callbacks (cooldowns, failure usage) ran for it. An
    /// error the attempt raised after litellm returned, such as a blocked-output content policy
    /// error, did not.
    pub callbacks_ran: bool,
    /// A failure the router must not blame on the deployment (advisor orchestration, a
    /// background cost poll's 404, a caller-set timeout's 408).
    pub exempt_from_cooldown: bool,
    /// `type(e) in litellm.LITELLM_EXCEPTION_TYPES`, which gates the exhausted-retry stamp.
    pub exact_litellm_type: bool,
}

impl Classified {
    pub fn is(&self, class: ExceptionClass) -> bool {
        self.classes.contains(&class)
    }
}

/// A rejection the router raises itself, before any deployment is called.
#[derive(Clone, Debug, PartialEq)]
pub enum Rejection {
    /// `RouterRateLimitError`: every candidate is cooling down or was filtered out.
    NoDeploymentsAvailable {
        model: String,
        cooldown_time: PyNumber,
        cooldown_list: Vec<String>,
        model_ids: Vec<String>,
        enable_pre_call_checks: bool,
    },
    /// `BadRequestError`: the model names no deployment.
    NoHealthyDeployments { model: String },
}

impl Rejection {
    pub fn classified(&self) -> Classified {
        match self {
            Self::NoDeploymentsAvailable { .. } => Classified {
                type_name: "RouterRateLimitError".into(),
                sleep_retry_after: -1,
                ..Classified::default()
            },
            Self::NoHealthyDeployments { model } => Classified {
                classes: vec![ExceptionClass::BadRequest],
                type_name: "BadRequestError".into(),
                message: format!(
                    "You passed in model={model}. There are no healthy deployments for this model"
                ),
                status_code: Some(400),
                sleep_retry_after: -1,
                exact_litellm_type: true,
                ..Classified::default()
            },
        }
    }
}

/// The origin of a failure: an exception the host raised, kept opaque so it can be raised
/// again unchanged, or a rejection the router produced.
#[derive(Clone, Debug)]
pub enum Raised<E> {
    Host(E),
    /// `id` is unique within one routed call, so the host can raise the same object for
    /// every reference to one rejection.
    Router {
        id: u64,
        rejection: Rejection,
    },
}

#[derive(Clone, Debug)]
pub struct Failure<E> {
    pub raised: Raised<E>,
    pub classified: Classified,
    /// The deployment whose attempt raised it, when one was picked.
    pub deployment_id: Option<String>,
}

impl<E> Failure<E> {
    pub fn host(error: E, classified: Classified, deployment_id: String) -> Self {
        Self {
            raised: Raised::Host(error),
            classified,
            deployment_id: Some(deployment_id),
        }
    }

    pub fn rejected(id: u64, rejection: Rejection) -> Self {
        Self {
            classified: rejection.classified(),
            raised: Raised::Router { id, rejection },
            deployment_id: None,
        }
    }
}
