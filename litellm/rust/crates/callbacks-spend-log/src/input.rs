//! The typed input of the builder, parsed once from the JSON document that the Python glue
//! (`litellm/proxy/spend_tracking/rust_spend_log_rows.py`) sends, or from the same document the
//! spend log callback receives.
//!
//! Scalars and flags become Rust types here, and odd values are read as absent rather than
//! failing the row. Documents the row stores as JSON, and whose shape the caller or provider
//! defines (request metadata, the request body, the response, usage counts, guardrail records),
//! stay [`Value`]s, since the builder copies them through rather than reading a fixed schema out
//! of them. The glue bounds every document with `safe_json_structure` before it crosses, so
//! depth and cycles are settled before this parse.

use litellm_python_compat::{
    Value,
    datetime::{IsoDateTime, fromisoformat},
    flags, json,
    truthy::truthy,
};

use crate::RowError;
use crate::py::{NONE, get, get_or_none, is_true, non_blank_str, to_str};
use crate::stamps::CALLER_SESSION_ID_METADATA_KEY;

fn required<'a>(value: &'a Value, key: &'static str) -> Result<&'a Value, RowError> {
    get(value, key).ok_or(RowError::Input(key))
}

fn owned(value: &Value, key: &str) -> Value {
    get_or_none(value, key).clone()
}

fn flag(value: &Value, key: &str) -> bool {
    matches!(get_or_none(value, key), Value::Bool(true))
}

/// A scalar read as text: `None` is absent, a string is itself, anything else is `str(value)`.
pub(crate) fn text_of(value: &Value) -> Option<String> {
    match value {
        Value::None => None,
        Value::Str(text) => Some(text.clone()),
        other => Some(to_str(other)),
    }
}

fn text_field(value: &Value, key: &str) -> Option<String> {
    text_of(get_or_none(value, key))
}

fn mapping(value: &Value) -> Option<Value> {
    matches!(value, Value::Dict(_)).then(|| value.clone())
}

/// A number read as a float: ints and floats are themselves, anything else is absent.
pub(crate) fn number_of(value: &Value) -> Option<f64> {
    match value {
        Value::Float(number) => Some(*number),
        Value::Int(number) => crate::py::big_to_f64(number),
        _ => None,
    }
}

/// One timestamp of the request, as the glue reads it from a Python `datetime`.
pub(crate) struct Timestamp {
    /// `value.astimezone(timezone.utc).isoformat()`, which the row stores.
    pub(crate) utc: String,
    /// `value.isoformat()` parsed back, for the request duration.
    pub(crate) raw: Option<IsoDateTime>,
}

impl Timestamp {
    fn parse(value: &Value, key: &'static str) -> Result<Self, RowError> {
        Ok(Self {
            utc: match get_or_none(value, "utc") {
                Value::Str(utc) => utc.clone(),
                _ => return Err(RowError::Input(key)),
            },
            raw: match get_or_none(value, "raw") {
                Value::Str(raw) => fromisoformat(raw),
                _ => None,
            },
        })
    }
}

/// A metadata bucket of `litellm_params` (`metadata` or `litellm_metadata`), projected to the
/// keys the row reads.
pub(crate) enum RequestMetadata {
    Absent,
    NotAMapping,
    Mapping(Value),
}

impl RequestMetadata {
    fn parse(params: &Value, key: &str) -> Self {
        match get(params, key) {
            None => Self::Absent,
            Some(found @ Value::Dict(_)) => Self::Mapping(found.clone()),
            Some(_) => Self::NotAMapping,
        }
    }

    pub(crate) fn mapping(&self) -> Option<&Value> {
        match self {
            Self::Mapping(value) => Some(value),
            _ => None,
        }
    }
}

/// The request's `litellm_params`, projected.
pub(crate) struct LitellmParams {
    pub(crate) api_base: Option<String>,
    pub(crate) litellm_call_id: Option<String>,
    /// Absent, `None`, or the proxy's record of the client request, whose `body` the row stores.
    /// The glue sends it only when the request's prompts are stored.
    pub(crate) proxy_server_request: Option<Value>,
    pub(crate) litellm_session_id: Option<String>,
    pub(crate) user_api_key_end_user_id: Option<String>,
    pub(crate) metadata: RequestMetadata,
    pub(crate) litellm_metadata: RequestMetadata,
}

impl LitellmParams {
    fn parse(value: &Value) -> Self {
        Self {
            api_base: text_field(value, "api_base"),
            litellm_call_id: text_field(value, "litellm_call_id"),
            proxy_server_request: get(value, "proxy_server_request").cloned(),
            litellm_session_id: text_field(value, "litellm_session_id"),
            user_api_key_end_user_id: text_field(value, "user_api_key_end_user_id"),
            metadata: RequestMetadata::parse(value, "metadata"),
            litellm_metadata: RequestMetadata::parse(value, "litellm_metadata"),
        }
    }

    /// The buckets that are mappings, `metadata` first.
    pub(crate) fn metadata_buckets(&self) -> impl Iterator<Item = &Value> {
        [self.metadata.mapping(), self.litellm_metadata.mapping()]
            .into_iter()
            .flatten()
    }
}

/// The `model` kwarg: a name, absent, or a value that is not text, which the row stores as the
/// unknown-model placeholder.
pub(crate) enum RequestedModel {
    Absent,
    Name(String),
    Malformed,
}

impl RequestedModel {
    fn parse(value: &Value) -> Self {
        match value {
            Value::None => Self::Absent,
            Value::Str(name) => Self::Name(name.clone()),
            _ => Self::Malformed,
        }
    }

    /// The name, or `""`.
    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Name(name) => name,
            _ => "",
        }
    }
}

/// The logging kwargs the row reads.
pub(crate) struct Kwargs {
    pub(crate) model: RequestedModel,
    pub(crate) custom_llm_provider: Option<String>,
    /// `kwargs.get("response_cost", 0)`: `None` when the kwarg is `None` or not a number.
    pub(crate) response_cost: Option<f64>,
    pub(crate) agent_id: Option<String>,
    pub(crate) litellm_call_id: Option<String>,
    pub(crate) litellm_trace_id: Option<String>,
    /// Sent only when prompts are stored.
    pub(crate) realtime_tools: Value,
    /// Sent only when prompts are stored.
    pub(crate) realtime_tool_calls: Value,
    /// The Azure spillover headers among the provider's response headers.
    pub(crate) response_headers: Option<Value>,
    /// `turn_off_message_logging` from the request's dynamic callback params, when it is set.
    pub(crate) turn_off_message_logging: Option<bool>,
}

impl Kwargs {
    fn parse(value: &Value) -> Self {
        Self {
            model: RequestedModel::parse(get_or_none(value, "model")),
            custom_llm_provider: text_field(value, "custom_llm_provider"),
            response_cost: match get(value, "response_cost") {
                None => Some(0.0),
                Some(cost) => number_of(cost),
            },
            agent_id: text_field(value, "agent_id"),
            litellm_call_id: text_field(value, "litellm_call_id"),
            litellm_trace_id: text_field(value, "litellm_trace_id"),
            realtime_tools: owned(value, "realtime_tools"),
            realtime_tool_calls: owned(value, "realtime_tool_calls"),
            response_headers: mapping(get_or_none(value, "response_headers")),
            turn_off_message_logging: match get_or_none(value, "turn_off_message_logging") {
                Value::Bool(flag) => Some(*flag),
                Value::Str(setting) => flags::str_to_bool(setting),
                _ => None,
            },
        }
    }
}

/// The request's `StandardLoggingPayload`, projected. Its fields are read the way
/// `get_logging_payload` reads them, so a missing field and a `None` one stay distinct.
pub(crate) struct LoggingPayload {
    pub(crate) fields: Value,
    pub(crate) metadata: Value,
    pub(crate) hidden_params: Value,
}

impl LoggingPayload {
    fn parse(value: &Value) -> Self {
        Self {
            fields: value.clone(),
            metadata: owned(value, "metadata"),
            hidden_params: owned(value, "hidden_params"),
        }
    }

    /// `payload.get(field)`.
    pub(crate) fn get(&self, field: &str) -> &Value {
        get_or_none(&self.fields, field)
    }

    /// `payload[field]` when present.
    pub(crate) fn field(&self, field: &str) -> Option<&Value> {
        get(&self.fields, field)
    }

    /// `payload["metadata"].get(field)`.
    pub(crate) fn metadata(&self, field: &str) -> &Value {
        get_or_none(&self.metadata, field)
    }

    /// `payload.get("hidden_params", {}).get(field)`.
    pub(crate) fn hidden(&self, field: &str) -> &Value {
        get(&self.hidden_params, field).unwrap_or(&NONE)
    }
}

/// What the row reads from the request's metadata about who made the call and how it was routed,
/// parsed once here. Each field keeps the Python lookup it mirrors: a fact that only matters as text
/// is a `String`, and a value whose type changes the row stays a [`Value`]. The metadata column
/// itself is still built from [`RowInput::metadata`].
#[derive(Clone, veil::Redact)]
pub(crate) struct RowAuthFacts {
    /// `metadata.get("user_api_key")`, `None` when absent: its type decides how it is redacted.
    /// `Debug` hides it, since it can be the raw key.
    #[redact(with = "[REDACTED]")]
    pub(crate) user_api_key: Option<Value>,
    /// `user_api_key_hash` when it is a str.
    pub(crate) user_api_key_hash: Option<String>,
    /// `user_api_key_alias` of any type, as the redaction reads it.
    pub(crate) user_api_key_alias: Value,
    /// `str()` of `user_api_key_user_id`, `user_api_key_team_id` and `user_api_key_org_id` when
    /// truthy, else empty.
    pub(crate) user_id: String,
    pub(crate) team_id: String,
    pub(crate) org_id: String,
    /// `str()` of `user_api_key_end_user_id` unless it is `None`.
    pub(crate) end_user_id: Option<String>,
    /// `user_api_key_user_email` when it is a str with more than whitespace.
    pub(crate) user_email: Option<String>,
    /// `str()` of `model_info["id"]` when truthy.
    pub(crate) model_id: Option<String>,
    /// `model_info["internal_router_model"] is True`.
    pub(crate) internal_router_model: bool,
    /// `model_group` and `deployment` when they are strs.
    pub(crate) model_group: Option<String>,
    pub(crate) deployment: Option<String>,
    /// `str()` of `agent_id` unless it is `None`.
    pub(crate) agent_id: Option<String>,
    /// `status == "failure"`.
    pub(crate) failed: bool,
    /// `metadata.get("tags")`, `None` when absent: only a list is stored.
    pub(crate) tags: Option<Value>,
    /// `litellm_session_id_omitted` is truthy.
    pub(crate) session_id_omitted: bool,
    /// `str()` of `session_id` when truthy.
    pub(crate) session_id: Option<String>,
    /// The caller session stamped beside the metadata, when it is a non-empty str.
    pub(crate) caller_session_id: Option<String>,
}

impl RowAuthFacts {
    fn parse(metadata: &Value) -> Self {
        let field = |key: &str| get_or_none(metadata, key);
        let str_field = |key: &str| match field(key) {
            Value::Str(found) => Some(found.clone()),
            _ => None,
        };
        let text_when_truthy =
            |value: &Value| truthy(value).then(|| text_of(value).unwrap_or_default());
        let model_info = field("model_info");
        Self {
            user_api_key: get(metadata, "user_api_key").cloned(),
            user_api_key_hash: str_field("user_api_key_hash"),
            user_api_key_alias: field("user_api_key_alias").clone(),
            user_id: text_when_truthy(field("user_api_key_user_id")).unwrap_or_default(),
            team_id: text_when_truthy(field("user_api_key_team_id")).unwrap_or_default(),
            org_id: text_when_truthy(field("user_api_key_org_id")).unwrap_or_default(),
            end_user_id: text_of(field("user_api_key_end_user_id")),
            user_email: non_blank_str(field("user_api_key_user_email")).map(str::to_owned),
            model_id: text_when_truthy(get_or_none(model_info, "id")),
            internal_router_model: is_true(get_or_none(model_info, "internal_router_model")),
            model_group: str_field("model_group"),
            deployment: str_field("deployment"),
            agent_id: text_of(field("agent_id")),
            failed: matches!(field("status"), Value::Str(status) if status == "failure"),
            tags: get(metadata, "tags").cloned(),
            session_id_omitted: truthy(field("litellm_session_id_omitted")),
            session_id: truthy(field("session_id")).then(|| to_str(field("session_id"))),
            caller_session_id: str_field(CALLER_SESSION_ID_METADATA_KEY)
                .filter(|session| !session.is_empty()),
        }
    }
}

/// Everything one spend log row is built from.
///
/// Build it with [`RowInput::from_json`] (the glue's JSON text) or [`RowInput::from_value`] (the
/// same document already parsed), then hand it to [`crate::SpendLogRowBuilder::build_row`].
pub struct RowInput {
    /// The row's base request id: `get_spend_logs_id(...)` in the glue, or the contract's
    /// `SpendLogId.id` in the spend log callback. The cache-hit suffix is added by the builder.
    pub(crate) request_id: Option<String>,
    pub(crate) call_type: Option<String>,
    /// The `cache_hit` kwarg, `None` when it is not a bool.
    pub(crate) cache_hit: Option<bool>,
    pub(crate) served_from_cache: bool,
    /// `should_store_prompts_and_responses_in_spend_logs(metadata)` for this request: the proxy
    /// setting with the key's and team's opt-outs applied.
    pub(crate) stores_prompts: bool,
    /// The spend log error message of a request rejected as an unknown model (`Some(None)` when
    /// the rejection carries none), or `None` for every other request.
    pub(crate) unknown_model_error_message: Option<Option<String>>,
    /// `time.time()` when the row is built, for the cache-hit id suffix.
    pub(crate) now: f64,
    /// A fresh `uuid4`, the session of a request that carries none.
    pub(crate) fallback_session_id: String,
    pub(crate) start_time: Timestamp,
    pub(crate) end_time: Timestamp,
    pub(crate) completion_start_time: Option<Timestamp>,
    /// Token and unit counts as the response reported them, in their reported order.
    pub(crate) usage: Value,
    /// The request's merged LiteLLM metadata, projected to the keys the row reads.
    pub(crate) metadata: Value,
    /// The facts the row reads from `metadata`, typed.
    pub(crate) auth: RowAuthFacts,
    pub(crate) litellm_params: LitellmParams,
    pub(crate) kwargs: Kwargs,
    pub(crate) payload: Option<LoggingPayload>,
    /// The cache key Python's cache computed for the request.
    pub(crate) cache_key: String,
    /// The provider the router registers for the model group, looked up in Python.
    pub(crate) model_group_provider: Option<String>,
    /// Whether the router knows the model group, looked up in Python for failed requests.
    pub(crate) is_configured_model_group: bool,
    /// The generateContent call type a chat completion served on a Google adapter route stands
    /// for, resolved from the route table in Python.
    pub(crate) route_call_type: Option<String>,
    /// The cost breakdown a billed row without a logging payload carries in its kwargs.
    pub(crate) snapshotless_cost_breakdown: Value,
    /// Whether a forwarded Claude subscription token ran in its allowance or overage, read from
    /// the logging payload or from the provider's rate-limit headers in Python.
    pub(crate) subscription_usage: Value,
}

impl RowInput {
    /// Parses the glue's input JSON text.
    pub fn from_json(text: &str) -> Result<Self, RowError> {
        Self::from_value(&json::loads(text)?)
    }

    /// Reads an input document already parsed into a [`Value`].
    pub fn from_value(value: &Value) -> Result<Self, RowError> {
        let dict_or_empty = |key: &str| match get_or_none(value, key) {
            found @ Value::Dict(_) => found.clone(),
            _ => Value::Dict(Vec::new()),
        };
        let rejected_as_unknown_model = flag(value, "rejected_as_unknown_model");
        let metadata = dict_or_empty("metadata");
        Ok(Self {
            request_id: match get_or_none(value, "request_id") {
                Value::Str(id) => Some(id.clone()),
                _ => None,
            },
            call_type: match get_or_none(value, "call_type") {
                Value::Str(kind) => Some(kind.clone()),
                _ => None,
            },
            cache_hit: match get_or_none(value, "cache_hit") {
                Value::Bool(flag) => Some(*flag),
                _ => None,
            },
            served_from_cache: flag(value, "served_from_cache"),
            stores_prompts: flag(value, "stores_prompts"),
            unknown_model_error_message: rejected_as_unknown_model
                .then(|| text_field(value, "unknown_model_error_message")),
            now: number_of(get_or_none(value, "now")).ok_or(RowError::Input("now"))?,
            fallback_session_id: match get_or_none(value, "fallback_session_id") {
                Value::Str(session) => session.clone(),
                _ => return Err(RowError::Input("fallback_session_id")),
            },
            start_time: Timestamp::parse(required(value, "start_time")?, "start_time")?,
            end_time: Timestamp::parse(required(value, "end_time")?, "end_time")?,
            completion_start_time: match get_or_none(value, "completion_start_time") {
                Value::None => None,
                found => Some(Timestamp::parse(found, "completion_start_time")?),
            },
            usage: dict_or_empty("usage"),
            auth: RowAuthFacts::parse(&metadata),
            metadata,
            litellm_params: LitellmParams::parse(get_or_none(value, "litellm_params")),
            kwargs: Kwargs::parse(get_or_none(value, "kwargs")),
            payload: match get_or_none(value, "standard_logging_object") {
                found @ Value::Dict(_) => Some(LoggingPayload::parse(found)),
                _ => None,
            },
            cache_key: text_field(value, "cache_key").unwrap_or_default(),
            model_group_provider: text_field(value, "model_group_provider"),
            is_configured_model_group: flag(value, "is_configured_model_group"),
            route_call_type: match get_or_none(value, "route_call_type") {
                Value::Str(call_type) => Some(call_type.clone()),
                _ => None,
            },
            snapshotless_cost_breakdown: owned(value, "snapshotless_cost_breakdown"),
            subscription_usage: owned(value, "subscription_usage"),
        })
    }
}
