//! A logging callback's kwargs and response, read as the limiter's settlement needs them.
//!
//! The objects are told apart by their Python class here; what the limiter charges from them is
//! decided in `litellm_callbacks_parallel_request_limiter::logged`.

use std::time::Duration;

use litellm_callbacks_parallel_request_limiter::{
    Backend, GuardedOp, LoggedCall, LoggedObject, Op, Resolved, UsageObject,
};
use litellm_host_python::json_from_py;
use litellm_redis_store::{
    Number,
    rate_limits::{Counter, Increment, Kind, Scope},
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};
use serde_json::{Map, Value};

/// The standard logging metadata members a settlement reads.
const METADATA_MEMBERS: [&str; 14] = [
    "user_api_key_app_id",
    "user_api_key_app_group_id",
    "user_api_key_hash",
    "user_api_key_user_id",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_project_id",
    "user_api_key_end_user_id",
    "agent_id",
    "session_id",
    "trace_id",
    "user_api_key_customer_org_id",
    "user_api_key_customer_team_id",
    "user_api_key_customer_member_id",
];
/// `INTERNAL_CALL_ORIGIN_METADATA_KEY`.
const ORIGIN_MEMBER: &str = "internal_call_origin";
/// How deep a dict is read before its members count only for their truthiness.
const DEPTH: usize = 8;

/// The classes a logging callback's objects are told apart by.
struct Classes<'py> {
    usage: Bound<'py, PyAny>,
    responses_usage: Bound<'py, PyAny>,
    modeled: Bound<'py, PyTuple>,
    extraction: Bound<'py, PyAny>,
    rerank: Bound<'py, PyAny>,
}

impl<'py> Classes<'py> {
    fn import(py: Python<'py>) -> PyResult<Self> {
        let utils = py.import("litellm.types.utils")?;
        let openai = py.import("litellm.types.llms.openai")?;
        Ok(Self {
            usage: utils.getattr("Usage")?,
            responses_usage: openai.getattr("ResponseAPIUsage")?,
            modeled: PyTuple::new(
                py,
                [
                    utils.getattr("ModelResponse")?,
                    utils.getattr("EmbeddingResponse")?,
                    utils.getattr("TextCompletionResponse")?,
                    openai.getattr("BaseLiteLLMOpenAIResponseObject")?,
                ],
            )?,
            extraction: py
                .import("litellm.types.extraction")?
                .getattr("ExtractionResponse")?,
            rerank: utils.getattr("RerankResponse")?,
        })
    }

    fn is_usage(&self, value: &Bound<'py, PyAny>) -> PyResult<bool> {
        Ok(value.is_instance(&self.usage)? || value.is_instance(&self.responses_usage)?)
    }
}

/// `mapping.get(name)`, None when it is absent, None, or the mapping cannot be read.
fn member<'py>(mapping: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    mapping
        .call_method1("get", (name,))
        .ok()
        .filter(|value| !value.is_none())
}

/// A Python value as the usage arithmetic reads it: scalars and dicts as JSON, a big int
/// saturated, and any other object only for its truthiness.
fn lenient(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if let Ok(int) = value.cast::<PyInt>() {
        return Ok(Value::from(int.extract::<i64>().unwrap_or_else(|_| {
            if int.lt(0).unwrap_or(false) {
                i64::MIN
            } else {
                i64::MAX
            }
        })));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(serde_json::Number::from_f64(float.value())
            .map_or_else(|| Value::String("nan".to_owned()), Value::Number));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::String(text.to_string_lossy().into_owned()));
    }
    if depth > 0
        && let Ok(dict) = value.cast::<PyDict>()
    {
        let mut members = Map::new();
        for (key, item) in dict.iter() {
            if let Ok(key) = key.cast::<PyString>() {
                members.insert(
                    key.to_string_lossy().into_owned(),
                    lenient(&item, depth - 1)?,
                );
            }
        }
        return Ok(Value::Object(members));
    }
    Ok(if value.is_truthy()? {
        Value::String("<object>".to_owned())
    } else {
        Value::Null
    })
}

fn attribute(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Value> {
    match object.getattr(name) {
        Ok(member) => lenient(&member, DEPTH),
        Err(_) => Ok(Value::Null),
    }
}

/// A usage object's counts, its cache reads read off its `details` member.
fn counts(
    usage: &Bound<'_, PyAny>,
    names: [&str; 3],
    details: &str,
) -> PyResult<Map<String, Value>> {
    let mut members = Map::new();
    for name in names {
        members.insert(name.to_owned(), attribute(usage, name)?);
    }
    let cached = match usage.getattr(details) {
        Ok(details) if !details.is_none() => attribute(&details, "cached_tokens")?,
        _ => Value::Null,
    };
    members.insert("cached_tokens".to_owned(), cached);
    Ok(members)
}

fn usage_object<'py>(
    classes: &Classes<'py>,
    usage: &Bound<'py, PyAny>,
    nests: bool,
) -> PyResult<UsageObject> {
    if usage.is_instance(&classes.usage)? {
        return Ok(UsageObject::Chat {
            members: counts(
                usage,
                ["prompt_tokens", "completion_tokens", "total_tokens"],
                "prompt_tokens_details",
            )?,
        });
    }
    if usage.is_instance(&classes.responses_usage)? {
        return Ok(UsageObject::Responses {
            members: counts(
                usage,
                ["input_tokens", "output_tokens", "total_tokens"],
                "input_tokens_details",
            )?,
        });
    }
    let Ok(dict) = usage.cast::<PyDict>() else {
        return Ok(UsageObject::Other);
    };
    let nested = match dict.get_item("usage")? {
        Some(nested)
            if nests && (classes.is_usage(&nested)? || nested.cast::<PyDict>().is_ok()) =>
        {
            Some(Box::new(usage_object(classes, &nested, false)?))
        }
        _ => None,
    };
    let Value::Object(members) = lenient(usage, DEPTH)? else {
        return Ok(UsageObject::Other);
    };
    Ok(UsageObject::Mapping { members, nested })
}

fn optional_usage<'py>(
    classes: &Classes<'py>,
    usage: Option<Bound<'py, PyAny>>,
) -> PyResult<Option<UsageObject>> {
    usage
        .filter(|usage| !usage.is_none())
        .map(|usage| usage_object(classes, &usage, true))
        .transpose()
}

/// A response or `combined_usage_object` by its class. `kwargs` is the callback's, for a
/// Responses WebSocket session's events to be combined into its usage.
fn logged_object<'py>(
    classes: &Classes<'py>,
    object: Option<Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyAny>>,
) -> PyResult<LoggedObject> {
    let Some(object) = object.filter(|object| !object.is_none()) else {
        return Ok(LoggedObject::Absent);
    };
    let py = object.py();
    if classes.is_usage(&object)? {
        return Ok(LoggedObject::Usage {
            usage: usage_object(classes, &object, true)?,
        });
    }
    if object.is_instance(classes.modeled.as_any())? {
        return Ok(LoggedObject::Modeled {
            usage: optional_usage(classes, object.getattr("usage").ok())?,
        });
    }
    if object.is_instance(&classes.extraction)? {
        let usage = py
            .import("litellm.cost_calculator")?
            .getattr("get_usage_object")?
            .call1((&object,))
            .ok();
        return Ok(LoggedObject::Extraction {
            usage: optional_usage(classes, usage)?,
        });
    }
    if object.is_instance(&classes.rerank)? {
        let meta = match object.getattr("meta") {
            Ok(meta) if meta.cast::<PyDict>().is_ok() => match lenient(&meta, DEPTH)? {
                Value::Object(meta) => Some(meta),
                _ => None,
            },
            _ => None,
        };
        return Ok(LoggedObject::Rerank { meta });
    }
    if object.cast::<PyDict>().is_ok() {
        return Ok(LoggedObject::Mapping {
            usage: usage_object(classes, &object, true)?,
        });
    }
    if let (Ok(_), Some(kwargs)) = (object.cast::<PyList>(), kwargs) {
        let usage = py
            .import("litellm.proxy.hooks.rate_limiter_utils")?
            .getattr("socket_session_usage")?
            .call1((kwargs, &object))
            .ok();
        return Ok(LoggedObject::Events {
            usage: optional_usage(classes, usage)?,
        });
    }
    Ok(LoggedObject::Other)
}

fn text(value: Option<Bound<'_, PyAny>>) -> Option<String> {
    let value = value?;
    let text = value.cast::<PyString>().ok()?;
    Some(text.to_string_lossy().into_owned())
}

/// `get_litellm_metadata_from_kwargs(kwargs).get(INTERNAL_CALL_ORIGIN_METADATA_KEY)`.
fn origin(kwargs: &Bound<'_, PyAny>) -> PyResult<Value> {
    let Some(params) = member(kwargs, "litellm_params") else {
        return Ok(Value::Null);
    };
    let mut metadata = None;
    for name in ["litellm_metadata", "metadata"] {
        if let Some(candidate) = member(&params, name)
            && candidate.is_truthy()?
        {
            metadata = Some(candidate);
            break;
        }
    }
    match metadata.and_then(|metadata| member(&metadata, ORIGIN_MEMBER)) {
        Some(origin) => lenient(&origin, DEPTH),
        None => Ok(Value::Null),
    }
}

/// A logging callback's call: its ids, the standard logging metadata, and its response and
/// `combined_usage_object` by their class. A failure is settled without its response.
pub(super) fn logged_call<'py>(
    kwargs: &Bound<'py, PyAny>,
    response: Option<&Bound<'py, PyAny>>,
) -> PyResult<LoggedCall> {
    let classes = Classes::import(kwargs.py())?;
    let mut metadata = Map::new();
    if let Some(logged) =
        member(kwargs, "standard_logging_object").and_then(|payload| member(&payload, "metadata"))
    {
        for name in METADATA_MEMBERS {
            if let Some(value) = member(&logged, name) {
                metadata.insert(name.to_owned(), lenient(&value, DEPTH)?);
            }
        }
    }
    Ok(LoggedCall {
        call_id: text(member(kwargs, "litellm_call_id")),
        call_type: text(member(kwargs, "call_type")),
        origin: origin(kwargs)?,
        user: match member(kwargs, "user") {
            Some(user) => lenient(&user, DEPTH)?,
            None => Value::Null,
        },
        metadata,
        response: logged_object(&classes, response.cloned(), Some(kwargs))?,
        combined: logged_object(&classes, member(kwargs, "combined_usage_object"), None)?,
        provider_billed: false,
        ended_before_slot_reuse: false,
    })
}

/// `_ended_before_slot_reuse`: a non-streamed round that ended at or before the moment a nested
/// call reused the request's slot. Naive and aware times never compare.
pub(super) fn ended_before_slot_reuse<'py>(
    kwargs: &Bound<'py, PyAny>,
    end_time: &Bound<'py, PyAny>,
    reused_at: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let datetime = end_time.py().import("datetime")?.getattr("datetime")?;
    if !end_time.is_instance(&datetime)? || !reused_at.is_instance(&datetime)? {
        return Ok(false);
    }
    if let Some(stream) = member(kwargs, "stream")
        && stream.is_truthy()?
    {
        return Ok(false);
    }
    if end_time.getattr("tzinfo")?.is_none() != reused_at.getattr("tzinfo")?.is_none() {
        return Ok(false);
    }
    end_time.le(reused_at)
}

/// What the proxy resolved for the settlement: its models and the live policy.
pub(super) fn resolved(value: &Bound<'_, PyAny>) -> PyResult<Resolved> {
    serde_json::from_value(json_from_py(value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// A reservation operation as the Python limiter writes one:
/// `(descriptor, value, kind, amount, ttl, expected_window_start, backend)`.
pub(super) type OperationRow<'py> = (
    String,
    String,
    String,
    Bound<'py, PyAny>,
    Option<i64>,
    Option<String>,
    Option<String>,
);

pub(super) fn operation(row: OperationRow<'_>) -> PyResult<Op> {
    let (descriptor, value, kind, amount, ttl, window_start, backend) = row;
    let kind: Kind = kind
        .parse()
        .map_err(|_| PyValueError::new_err(format!("{kind} is not a counter kind")))?;
    let amount = if amount.cast::<PyFloat>().is_ok() {
        Number::Float(amount.extract()?)
    } else {
        Number::Int(amount.extract()?)
    };
    let increment = Increment {
        counter: Counter {
            scope: Scope { descriptor, value },
            kind,
        },
        amount,
        ttl: ttl.map(|ttl| Duration::from_secs(u64::try_from(ttl).unwrap_or(0))),
    };
    Ok(match (window_start, backend) {
        (Some(expected_window_start), Some(backend)) => Op::Guarded(GuardedOp {
            increment,
            expected_window_start,
            backend: if backend == "local" {
                Backend::Local
            } else {
                Backend::Redis
            },
        }),
        _ => Op::Plain(increment),
    })
}
