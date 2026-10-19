use std::{
    ops::{BitOr, BitOrAssign},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime},
};

use bytes::Bytes;
use http::HeaderMap;
use serde_json::{Map, Value};

use litellm_gateway_auth::keys::{KeyBudgets, KeyLimits};

use crate::usage::Usage;

/// The facts a hook reads or an observer asks for. The gateway builds only what some hook or
/// observer declares, so a fact nobody reads costs nothing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Facts(u16);

impl Facts {
    pub const NONE: Self = Self(0);
    /// Who sent the request: the key, its user, team and organization.
    pub const KEY: Self = Self(1 << 0);
    pub const END_USER: Self = Self(1 << 1);
    pub const CLIENT_HEADERS: Self = Self(1 << 2);
    /// An estimate of the request's input tokens, made on first read.
    pub const INPUT_TOKENS: Self = Self(1 << 3);
    /// The deployment that served the request.
    pub const DEPLOYMENT: Self = Self(1 << 4);
    /// Token usage read from the response as it streams.
    pub const USAGE: Self = Self(1 << 5);
    /// The response's cost, from usage and the deployment's prices.
    pub const COST: Self = Self(1 << 6);
    pub const TIMINGS: Self = Self(1 << 7);
    /// Values earlier hooks attached for observers.
    pub const ANNOTATIONS: Self = Self(1 << 8);
    /// The raw request bytes, already held for retries. Hooks read them while the request is
    /// admitted; only an observer that asks for them keeps a copy for the rest of the request.
    pub const REQUEST_BODY: Self = Self(1 << 9);
    /// The response text. The only fact that makes the gateway keep a stream's bytes.
    pub const RESPONSE_BODY: Self = Self(1 << 10);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for Facts {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Facts {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

/// The endpoint a request came in on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    Messages,
    ChatCompletions,
}

impl Route {
    /// The proxy's `call_type` for this endpoint, as spend logs record it.
    pub fn call_type(self) -> &'static str {
        match self {
            Self::Messages => "anthropic_messages",
            Self::ChatCompletions => "acompletion",
        }
    }

    /// The endpoint the proxy's daily spend tables key on.
    pub fn endpoint(self) -> &'static str {
        match self {
            Self::Messages => "/v1/messages",
            Self::ChatCompletions => "/chat/completions",
        }
    }
}

/// Who sent a request, as the key store knows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyFacts {
    /// The sha256 hex of the key, `LiteLLM_VerificationToken.token`.
    pub hash: String,
    pub alias: Option<String>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub team_alias: Option<String>,
    pub org_id: Option<String>,
    pub metadata: Arc<Map<String, Value>>,
    pub team_metadata: Arc<Map<String, Value>>,
    pub budgets: KeyBudgets,
    pub limits: KeyLimits,
}

/// What the gateway knows about a request before it is sent, shared by every hook and observer.
#[derive(Debug)]
pub struct RequestFacts {
    pub request_id: Arc<str>,
    pub route: Route,
    pub key: Option<Arc<KeyFacts>>,
    pub end_user: Option<Arc<str>>,
    /// The public model name the client asked for.
    pub requested_model: Arc<str>,
    pub client_headers: Arc<HeaderMap>,
    pub stream: bool,
    /// The output budget the request set, if any.
    pub max_tokens: Option<u64>,
    pub started_at: SystemTime,
    /// The request bytes while hooks run. Afterwards a compact copy when an observer reads them,
    /// else empty: the raw bytes are a slice of the connection's read buffer, and holding them
    /// would hold the whole buffer for as long as the response streams.
    body: Mutex<Bytes>,
    body_len: usize,
    input_tokens: OnceLock<u64>,
}

impl RequestFacts {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: Arc<str>,
        route: Route,
        key: Option<Arc<KeyFacts>>,
        end_user: Option<Arc<str>>,
        requested_model: Arc<str>,
        client_headers: Arc<HeaderMap>,
        stream: bool,
        max_tokens: Option<u64>,
        body: Bytes,
    ) -> Self {
        Self {
            request_id,
            route,
            key,
            end_user,
            requested_model,
            client_headers,
            stream,
            max_tokens,
            started_at: SystemTime::now(),
            body_len: body.len(),
            body: Mutex::new(body),
            input_tokens: OnceLock::new(),
        }
    }

    /// The request bytes: in full while hooks run, afterwards only for observers that read
    /// [`Facts::REQUEST_BODY`], empty for everyone else.
    pub fn body(&self) -> Bytes {
        self.body
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    /// Lets go of the raw bytes once hooks have run, keeping an owned copy when `keep`.
    pub(crate) fn release_body(&self, keep: bool) {
        let mut body = self
            .body
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *body = match keep {
            true => Bytes::copy_from_slice(&body),
            false => Bytes::new(),
        };
    }

    /// The proxy's cheap input estimate, a token per four bytes of request, made once.
    pub fn input_tokens(&self) -> u64 {
        *self
            .input_tokens
            .get_or_init(|| (self.body_len as u64).div_ceil(4).max(1))
    }
}

/// The deployment a request was served by.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeploymentFacts {
    /// `model_info.id`, or the id the router derives for a deployment without one.
    pub id: Arc<str>,
    pub model_group: Arc<str>,
    pub model: String,
    pub provider: Option<String>,
    pub api_base: Option<String>,
}

impl DeploymentFacts {
    /// The provider, from `custom_llm_provider` or the model's `provider/` prefix.
    pub fn provider(&self) -> Option<&str> {
        self.provider
            .as_deref()
            .or_else(|| self.model.split_once('/').map(|(provider, _)| provider))
    }
}

/// Values hooks attach for observers, in the order they attached them. Later writes to the
/// same key replace earlier ones.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotations(Vec<(&'static str, Value)>);

impl Annotations {
    pub fn set(&mut self, key: &'static str, value: Value) {
        match self.0.iter_mut().find(|(name, _)| *name == key) {
            Some((_, slot)) => *slot = value,
            None => self.0.push((key, value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| value)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// How a request ended.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub deployment: Option<DeploymentFacts>,
    /// The HTTP status the client got.
    pub status: u16,
    pub usage: Option<Usage>,
    /// In dollars, from usage and the deployment's prices; `None` when either is missing.
    pub cost: Option<f64>,
    pub time_to_first_byte: Option<Duration>,
    /// When the attempt that served, or failed last, was sent to the provider.
    pub upstream_started_at: Option<SystemTime>,
    pub ended_at: SystemTime,
    /// The stream stopped before its end: the client went away or the upstream failed.
    pub interrupted: bool,
}

/// Everything an observer can be handed about one request.
#[derive(Debug)]
pub struct Record {
    pub request: Arc<RequestFacts>,
    pub outcome: Outcome,
    pub annotations: Annotations,
    /// Present only when an observer reads [`Facts::RESPONSE_BODY`].
    pub response_body: Option<Bytes>,
}

impl Record {
    pub fn key(&self) -> Option<&KeyFacts> {
        self.request.key.as_deref()
    }

    pub fn succeeded(&self) -> bool {
        (200..300).contains(&self.outcome.status) && !self.outcome.interrupted
    }

    /// The request reached a provider and was cut off before it reported usage, as when the
    /// client goes away while the provider works. The provider may still bill it, so what it
    /// cost is unknown and reservations should be kept rather than returned.
    pub fn cut_off_after_dispatch(&self) -> bool {
        self.outcome.interrupted
            && self.outcome.upstream_started_at.is_some()
            && self.outcome.usage.is_none()
    }

    pub fn duration(&self) -> Duration {
        self.outcome
            .ended_at
            .duration_since(self.request.started_at)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn facts_combine_and_contain() {
        let facts = Facts::KEY | Facts::USAGE;
        assert!(facts.contains(Facts::KEY));
        assert!(facts.contains(Facts::KEY | Facts::USAGE));
        assert!(!facts.contains(Facts::COST));
        assert!(Facts::NONE.is_empty());
    }

    #[rstest]
    fn later_annotations_replace_earlier_ones() {
        let mut annotations = Annotations::default();
        annotations.set("a", Value::from(1));
        annotations.set("b", Value::from(2));
        annotations.set("a", Value::from(3));
        assert_eq!(annotations.get("a"), Some(&Value::from(3)));
        assert_eq!(annotations.get("b"), Some(&Value::from(2)));
    }

    #[rstest]
    #[case::prefix(None, "anthropic/claude-x", Some("anthropic"))]
    #[case::explicit(Some("bedrock"), "anthropic.claude", Some("bedrock"))]
    #[case::bare(None, "gpt-x", None)]
    fn provider_comes_from_the_setting_or_the_model_prefix(
        #[case] provider: Option<&str>,
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        let deployment = DeploymentFacts {
            model: model.into(),
            provider: provider.map(str::to_owned),
            ..DeploymentFacts::default()
        };
        assert_eq!(deployment.provider(), expected);
    }
}
