//! Model group info and the configured-value reads: what the router reports a model group, or a
//! concrete deployment behind a listed name, can do.
//!
//! The rules are the Python router core's (`_set_model_group_info`, `get_model_listing_info`,
//! `get_configured_token_limits`, `get_configured_mode`, `get_configured_display_name`,
//! `get_configured_service_tiers`). The host picks the deployments (the registry lookups) and
//! answers what needs its objects as [`Host::model_info`] lookups asked while the query runs: a
//! deployment's validated params, its merged model info and provider, its capabilities, and the
//! group's first entry as its model validates it. The core decides which deployments count and
//! how their values combine.

use serde_json::{Map, Number, Value};

use crate::{
    error::Error,
    host::{Host, ModelInfoLookup},
    model_info::{
        discovered, is_opaque, lookup, malformed, python_float, python_int, raised, truthy,
        without_digit_underscores,
    },
};

/// One `_set_model_group_info` query.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct GroupInfoQuery {
    /// The model group the deployments serve (an alias already resolved).
    pub model_group: String,
    /// The name the group is reported as.
    pub user_facing_model_group_name: String,
    /// The group is a team's public model name the team's deployments serve.
    pub serves_team_deployments: bool,
    /// A wildcard pattern routes `model_group` (asked only when an entry needs it).
    pub pattern_routes: bool,
    /// The first string `model_info.description` among the deployments.
    pub description: Value,
    /// The deployments the group routes to, as stored.
    pub deployments: Vec<Value>,
}

/// The group info: `None` when no deployment matched, else the fields the merge assigned after
/// the host built the group from its first deployment ([`ModelInfoLookup::GroupSeed`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupInfo {
    pub assignments: Map<String, Value>,
}

/// The type name Python reports for a value, in its error messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) if is_opaque(value) => "object",
        Value::Object(_) => "dict",
    }
}

/// A number as Python's arithmetic reads it (a bool is 0 or 1), or `None`.
fn arithmetic(value: &Value) -> Option<Number> {
    match value {
        Value::Bool(flag) => Some(Number::from(u8::from(*flag))),
        Value::Number(number) => Some(number.clone()),
        _ => None,
    }
}

/// `total += value`, with `total` starting at `0`; adding a non-number raises `TypeError`.
fn add(total: Option<Value>, value: &Value) -> Result<Value, Error> {
    let total = total.unwrap_or_else(|| Value::Number(Number::from(0)));
    let (Some(left), Some(right)) = (arithmetic(&total), arithmetic(value)) else {
        return Err(raised(
            "TypeError",
            format!(
                "unsupported operand type(s) for +=: '{}' and '{}'",
                type_name(&total),
                type_name(value)
            ),
        ));
    };
    if let (Some(left), Some(right)) = (left.as_i64(), right.as_i64())
        && let Some(sum) = left.checked_add(right)
    {
        return Ok(Value::Number(Number::from(sum)));
    }
    let sum = left.as_f64().unwrap_or(0.0) + right.as_f64().unwrap_or(0.0);
    Ok(Number::from_f64(sum).map_or(Value::Null, Value::Number))
}

/// `left > right`, for numbers or strings; any other pair raises `TypeError`.
fn greater(left: &Value, right: &Value) -> Result<bool, Error> {
    if let (Some(left), Some(right)) = (arithmetic(left), arithmetic(right)) {
        return Ok(left.as_f64().unwrap_or(0.0) > right.as_f64().unwrap_or(0.0));
    }
    if let (Value::String(left), Value::String(right)) = (left, right) {
        return Ok(left > right);
    }
    Err(raised(
        "TypeError",
        format!(
            "'>' not supported between instances of '{}' and '{}'",
            type_name(left),
            type_name(right)
        ),
    ))
}

/// `_cost_value_as_float`: `float(value)`, or `None` when it is unset or not a number.
fn cost_as_float(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        Value::Number(number) => number.as_f64(),
        Value::String(text) => {
            without_digit_underscores(text.trim()).and_then(|digits| python_float(&digits))
        }
        _ => None,
    }
}

fn float(value: Option<f64>) -> Value {
    value
        .and_then(Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// `value.get(key)` on a dict; a value that is no dict raises `AttributeError`, as `.get` does.
fn get<'a>(value: &'a Value, key: &str) -> Result<Option<&'a Value>, Error> {
    match value {
        Value::Object(map) if !is_opaque(value) => Ok(map.get(key)),
        other => Err(raised(
            "AttributeError",
            format!("'{}' object has no attribute 'get'", type_name(other)),
        )),
    }
}

/// `model.get(key)`, then `litellm_params.get(key)`, then `model_info.get(key)`: the first that is
/// not `None`.
fn first_set(
    model: &Value,
    params: &Value,
    info: &Value,
    key: &str,
) -> Result<Option<Value>, Error> {
    for layer in [model, params, info] {
        if let Some(value) = get(layer, key)?.filter(|value| !value.is_null()) {
            return Ok(Some(value.clone()));
        }
    }
    Ok(None)
}

fn object(answer: Value, kind: &str) -> Result<Map<String, Value>, Error> {
    match answer {
        Value::Object(map) => Ok(map),
        _ => Err(malformed(kind)),
    }
}

/// The group being merged: the host's validated first entry, and what the merge assigned since.
struct Merged {
    current: Map<String, Value>,
    assignments: Map<String, Value>,
}

impl Merged {
    fn get(&self, field: &str) -> &Value {
        self.current.get(field).unwrap_or(&Value::Null)
    }

    fn set(&mut self, field: &str, value: Value) {
        self.current.insert(field.to_owned(), value.clone());
        self.assignments.insert(field.to_owned(), value);
    }

    /// `if info.get(field) is not None and info[field] > group.field (or group.field is None)`.
    fn widen(&mut self, field: &str, info: &Map<String, Value>) -> Result<(), Error> {
        let Some(value) = info.get(field).filter(|value| !value.is_null()) else {
            return Ok(());
        };
        let current = self.get(field);
        if current.is_null() || greater(value, current)? {
            self.set(field, value.clone());
        }
        Ok(())
    }

    /// The cost `field` widened to `float(info[field])` when that exceeds `group.field or 0.0`.
    fn widen_cost(&mut self, field: &str, info: &Map<String, Value>) {
        let Some(cost) = cost_as_float(info.get(field)) else {
            return;
        };
        let current = self.get(field);
        let floor = if truthy(current) {
            current.as_f64().unwrap_or(0.0)
        } else {
            0.0
        };
        if current.is_null() || cost > floor {
            self.set(field, float(Some(cost)));
        }
    }

    /// `if info.get(field) is True: group.field = True`.
    fn any(&mut self, field: &str, info: &Map<String, Value>) {
        if info.get(field) == Some(&Value::Bool(true)) {
            self.set(field, Value::Bool(true));
        }
    }
}

/// `tuple(dict.fromkeys((*current, *added))) or None`.
fn union(current: &Value, added: &[Value]) -> Value {
    let mut kept: Vec<Value> = Vec::new();
    let existing = current.as_array().map(Vec::as_slice).unwrap_or_default();
    for value in existing.iter().chain(added) {
        if !kept.contains(value) {
            kept.push(value.clone());
        }
    }
    if kept.is_empty() {
        Value::Null
    } else {
        Value::Array(kept)
    }
}

/// `_set_model_group_info`: the group's info merged across the deployments that serve it.
pub async fn model_group_info(
    host: &dyn Host,
    query: &GroupInfoQuery,
) -> Result<Option<GroupInfo>, Error> {
    let mut merged: Option<Merged> = None;
    let mut totals: [(&str, Option<Value>); 4] =
        [("tpm", None), ("rpm", None), ("itpm", None), ("otpm", None)];
    let mut clientside_auth_params = Value::Null;
    let mut efforts_initialized = false;
    let mut efforts_unknown = false;
    let empty = Value::Object(Map::new());

    for (index, model) in query.deployments.iter().enumerate() {
        let named = get(model, "model_name")?;
        let is_match = query.serves_team_deployments
            || named.is_some_and(|name| name.as_str() == Some(query.model_group.as_str()))
            || (named.is_some() && query.pattern_routes);
        if !is_match {
            continue;
        }
        let params = object(
            lookup(host, ModelInfoLookup::GroupParams { index }).await?,
            "group_params",
        )?;
        clientside_auth_params = params
            .get("configurable_clientside_auth_params")
            .cloned()
            .unwrap_or(Value::Null);
        let model_params = get(model, "litellm_params")?.unwrap_or(&empty);
        let model_info_dict = get(model, "model_info")?.unwrap_or(&empty);
        let mut deployment = [None, None, None, None];
        for (slot, (metric, _)) in deployment.iter_mut().zip(totals.iter()) {
            *slot = first_set(model, model_params, model_info_dict, metric)?;
        }

        let facts = object(
            lookup(
                host,
                ModelInfoLookup::GroupDeployment {
                    index,
                    model: params.get("model").cloned().unwrap_or(Value::Null),
                },
            )
            .await?,
            "group_deployment",
        )?;
        let mapped = facts.get("mapped") == Some(&Value::Bool(true));
        let litellm_model = facts.get("litellm_model").cloned().unwrap_or(Value::Null);
        let llm_provider = facts.get("llm_provider").cloned().unwrap_or(Value::Null);
        let model_info = match facts.get("model_info") {
            Some(Value::Object(info)) if !is_opaque(&Value::Object(info.clone())) => info.clone(),
            Some(Value::Null) | None => {
                // No model info: what the deployment's own (database) model_info says.
                let db_info = get(model, "model_info")?.unwrap_or(&empty);
                let mut info = Map::new();
                info.insert("key".to_owned(), Value::String(query.model_group.clone()));
                for key in ["max_tokens", "max_input_tokens", "max_output_tokens"] {
                    info.insert(key.to_owned(), Value::Null);
                }
                for key in ["input_cost_per_token", "output_cost_per_token"] {
                    info.insert(key.to_owned(), float(cost_as_float(get(db_info, key)?)));
                }
                info.insert("litellm_provider".to_owned(), llm_provider.clone());
                info.insert(
                    "mode".to_owned(),
                    match get(db_info, "mode")? {
                        Some(mode) => mode.clone(),
                        None => Value::String("chat".to_owned()),
                    },
                );
                info.insert(
                    "supported_openai_params".to_owned(),
                    match facts.get("supported_openai_params") {
                        Some(Value::Null) | None => Value::Array(Vec::new()),
                        Some(params) => params.clone(),
                    },
                );
                info.insert("supports_system_messages".to_owned(), Value::Null);
                info
            }
            Some(_) => return Err(malformed("group_deployment")),
        };

        match merged.as_mut() {
            None => {
                let mut seed = Map::new();
                seed.insert(
                    "model_group".to_owned(),
                    Value::String(query.user_facing_model_group_name.clone()),
                );
                seed.insert(
                    "providers".to_owned(),
                    Value::Array(vec![llm_provider.clone()]),
                );
                seed.insert("description".to_owned(), query.description.clone());
                seed.extend(model_info.clone());
                seed.insert("supports_fast_mode".to_owned(), Value::Bool(true));
                seed.insert("supported_reasoning_efforts".to_owned(), Value::Null);
                seed.insert("supported_endpoints".to_owned(), Value::Null);
                let current = object(
                    lookup(host, ModelInfoLookup::GroupSeed { info: seed }).await?,
                    "group_seed",
                )?;
                merged = Some(Merged {
                    current,
                    assignments: Map::new(),
                });
            }
            Some(group) => {
                let providers = group
                    .get("providers")
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
                if !providers.contains(&llm_provider) {
                    let mut providers = providers;
                    providers.push(llm_provider.clone());
                    group.set("providers", Value::Array(providers));
                }
                group.widen("max_input_tokens", &model_info)?;
                group.widen("max_output_tokens", &model_info)?;
                group.widen_cost("input_cost_per_token", &model_info);
                group.widen_cost("output_cost_per_token", &model_info);
                for field in [
                    "supports_parallel_function_calling",
                    "supports_vision",
                    "supports_function_calling",
                    "supports_web_search",
                    "supports_url_context",
                    "supports_reasoning",
                ] {
                    group.any(field, &model_info);
                }
                if let Some(params) = model_info
                    .get("supported_openai_params")
                    .filter(|params| !params.is_null())
                {
                    group.set("supported_openai_params", params.clone());
                }
                for (slot, metric) in deployment.iter_mut().zip(["tpm", "rpm"]) {
                    if slot.is_none()
                        && let Some(value) = model_info.get(metric).filter(|v| !v.is_null())
                    {
                        *slot = Some(value.clone());
                    }
                }
            }
        }
        let Some(group) = merged.as_mut() else {
            unreachable!("the group was just built");
        };

        let capabilities = object(
            lookup(
                host,
                ModelInfoLookup::GroupCapabilities {
                    index,
                    model_info: Value::Object(model_info),
                    litellm_model,
                    llm_provider,
                    mapped,
                },
            )
            .await?,
            "group_capabilities",
        )?;
        let endpoints = capabilities
            .get("supported_endpoints")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let endpoints = union(group.get("supported_endpoints"), &endpoints);
        group.set("supported_endpoints", endpoints);
        let fast_mode = truthy(group.get("supports_fast_mode"))
            && capabilities.get("supports_fast_mode") == Some(&Value::Bool(true));
        group.set("supports_fast_mode", Value::Bool(fast_mode));
        match capabilities
            .get("reasoning_efforts")
            .cloned()
            .unwrap_or(Value::Null)
        {
            Value::Null => {
                efforts_unknown = true;
                group.set("supported_reasoning_efforts", Value::Null);
            }
            resolved if !efforts_initialized => {
                efforts_initialized = true;
                if !efforts_unknown {
                    group.set("supported_reasoning_efforts", resolved);
                }
            }
            resolved if !efforts_unknown => {
                let intersected = lookup(
                    host,
                    ModelInfoLookup::ReasoningIntersect {
                        current: group.get("supported_reasoning_efforts").clone(),
                        resolved,
                    },
                )
                .await?;
                group.set("supported_reasoning_efforts", intersected);
            }
            _ => {}
        }

        for ((_, total), value) in totals.iter_mut().zip(deployment) {
            if let Some(value) = value {
                *total = Some(add(total.take(), &value)?);
            }
        }
    }

    let Some(mut group) = merged else {
        return Ok(None);
    };
    for (metric, total) in totals {
        if let Some(total) = total {
            group.set(metric, total);
        }
    }
    if !clientside_auth_params.is_null() {
        group.set(
            "configurable_clientside_auth_params",
            clientside_auth_params,
        );
    }
    Ok(Some(GroupInfo {
        assignments: group.assignments,
    }))
}

/// `coerce_token_limit`: a token limit as an int, `None` when it is unset or not a usable number
/// (a bool never is).
pub fn coerce_token_limit(value: Option<&Value>) -> Option<i64> {
    match value? {
        Value::Number(number) => number.as_i64().or_else(|| {
            number
                .as_f64()
                .filter(|n| n.is_finite())
                .map(|n| n.trunc() as i64)
        }),
        Value::String(text) => without_digit_underscores(text.trim())
            .and_then(|digits| python_int(&digits))
            .and_then(|number| number.as_i64()),
        _ => None,
    }
}

/// The deployment's discovered limits under its configured `model_info` values that are set.
async fn effective_model_info(
    host: &dyn Host,
    id: Value,
    configured: &Map<String, Value>,
) -> Result<Map<String, Value>, Error> {
    let mut info = discovered(host, id).await?;
    info.extend(
        configured
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    Ok(info)
}

/// `get_configured_token_limits`: the (max input, max output) tokens a concrete deployment
/// configures, or its refresh discovered, from its `model_info` (`exclude_none` dump) and id.
pub async fn configured_token_limits(
    host: &dyn Host,
    id: Value,
    configured: &Map<String, Value>,
) -> Result<(Option<i64>, Option<i64>), Error> {
    let info = effective_model_info(host, id, configured).await?;
    Ok((
        coerce_token_limit(info.get("max_input_tokens")),
        coerce_token_limit(info.get("max_output_tokens")),
    ))
}

/// `get_configured_mode` / `get_configured_display_name`: a configured string that is not blank.
pub fn configured_text(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
}

/// `get_configured_service_tiers`: each deployment's `model_info.service_tiers`, unvalidated.
pub fn configured_service_tiers(model_infos: &[Map<String, Value>]) -> Vec<Value> {
    model_infos
        .iter()
        .map(|info| info.get("service_tiers").cloned().unwrap_or(Value::Null))
        .collect()
}

/// What a listed name's deployments contribute to its `/v1/models` entry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ListingInfo {
    pub cost_map_keys: Vec<String>,
    pub max_input_tokens: Option<i64>,
    pub max_output_tokens: Option<i64>,
}

fn named_text(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

/// `deployment_pricing_models(deployment)[0]`: `model_info.base_model`, then
/// `litellm_params.base_model`, then `litellm_params.model`; malformed blocks count as unset.
fn pricing_key(deployment: &Value) -> Option<String> {
    let block = |key: &str| {
        deployment
            .get(key)
            .and_then(Value::as_object)
            .filter(|_| !is_opaque(deployment.get(key).unwrap_or(&Value::Null)))
    };
    let params = block("litellm_params");
    let info = block("model_info");
    let base_model = info
        .and_then(|info| info.get("base_model"))
        .filter(|base| truthy(base))
        .or_else(|| params.and_then(|params| params.get("base_model")));
    named_text(base_model)
        .or_else(|| named_text(params.and_then(|params| params.get("model"))))
        .map(str::to_owned)
}

/// `get_model_listing_info` over the deployments a listed name names: the cost-map keys of their
/// models, deduplicated, and the widest token limits they configure or discovered.
pub async fn model_listing_info(
    host: &dyn Host,
    deployments: &[Value],
) -> Result<ListingInfo, Error> {
    let mut listing = ListingInfo::default();
    let empty = Value::Object(Map::new());
    for deployment in deployments {
        let info = match get(deployment, "model_info")? {
            Some(info) if truthy(info) => info,
            _ => &empty,
        };
        let id = get(info, "id")?.cloned().unwrap_or(Value::Null);
        let configured = match info {
            Value::Object(map) => map.clone(),
            _ => Map::new(),
        };
        let effective = effective_model_info(host, id, &configured).await?;
        for (field, widest) in [
            ("max_input_tokens", &mut listing.max_input_tokens),
            ("max_output_tokens", &mut listing.max_output_tokens),
        ] {
            if let Some(limit) = coerce_token_limit(effective.get(field)) {
                *widest = Some(widest.map_or(limit, |current| current.max(limit)));
            }
        }
    }
    for key in deployments.iter().filter_map(pricing_key) {
        if !listing.cost_map_keys.contains(&key) {
            listing.cost_map_keys.push(key);
        }
    }
    Ok(listing)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn coerce_token_limit_reads_python_ints() {
        assert_eq!(coerce_token_limit(Some(&json!(128))), Some(128));
        assert_eq!(coerce_token_limit(Some(&json!(12.9))), Some(12));
        assert_eq!(coerce_token_limit(Some(&json!(" 1_000 "))), Some(1000));
        assert_eq!(coerce_token_limit(Some(&json!("128,000"))), None);
        assert_eq!(coerce_token_limit(Some(&json!(true))), None);
        assert_eq!(coerce_token_limit(None), None);
    }

    #[test]
    fn totals_add_like_python() {
        assert_eq!(add(None, &json!(5)).unwrap(), json!(5));
        assert_eq!(add(Some(json!(5)), &json!(true)).unwrap(), json!(6));
        assert_eq!(add(Some(json!(5)), &json!(0.5)).unwrap(), json!(5.5));
        assert!(add(Some(json!(5)), &json!("1")).is_err());
    }

    #[test]
    fn union_keeps_first_seen_order_and_empty_reads_as_none() {
        assert_eq!(
            union(&json!(["a", "b"]), &[json!("b"), json!("c")]),
            json!(["a", "b", "c"])
        );
        assert_eq!(union(&Value::Null, &[]), Value::Null);
    }

    #[test]
    fn pricing_key_prefers_the_base_model() {
        let deployment = json!({
            "litellm_params": {"model": "azure/x", "base_model": "params-base"},
            "model_info": {"base_model": "info-base"},
        });
        assert_eq!(pricing_key(&deployment).as_deref(), Some("info-base"));
        let deployment = json!({"litellm_params": {"model": "azure/x"}, "model_info": "junk"});
        assert_eq!(pricing_key(&deployment).as_deref(), Some("azure/x"));
    }
}
