//! Model info: what the router reports a deployment can do and costs, merged from the process
//! cost map, what the background refresh discovered, and the deployment's own `model_info`.
//!
//! The merge rules are the Python router core's (`get_router_model_info`,
//! `get_deployment_model_info`); every cost-map read is a [`Host::model_info`] lookup asked while
//! the query runs, so a price reload or a registration the host made since the router was built is
//! what the next query reads. The host keeps choosing the keys a lookup tries (`get_model_info`);
//! the core chooses which lookups a query makes.
//!
//! Values the host cannot spell as JSON travel as opaque tokens ([`is_opaque`]): the merge moves
//! them between layers without reading them, and the host restores them.

use serde_json::{Map, Number, Value};

use crate::{
    error::Error,
    host::{Host, LogLevel, ModelInfoLookup},
};

/// The key the host wraps a value it cannot spell as JSON in.
pub const OPAQUE_KEY: &str = "__litellm_python_value__";

/// A value the host passed as an opaque token: the merge moves it, never reads it.
pub fn is_opaque(value: &Value) -> bool {
    matches!(value, Value::Object(map) if map.len() == 1 && map.contains_key(OPAQUE_KEY))
}

/// One `get_router_model_info` query, as the host reads it off the deployment.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct RouterModelInfoQuery {
    /// `model_info.id`.
    pub deployment_id: Value,
    /// `model_info.base_model`, then `litellm_params.base_model`: the first that is not `None`
    /// names the model whose limits and costs the deployment reports.
    pub model_info_base_model: Value,
    pub params_base_model: Value,
    /// The validated `litellm_params.model`.
    pub params_model: String,
    /// `base_model` among the validated params' extra fields.
    pub extra_base_model: Value,
    /// The name the caller asked for, which a wildcard deployment matches.
    pub received_model_name: String,
}

/// A merged model info, and the keys whose values are the deployment's own (the last layer).
#[derive(Clone, Debug, PartialEq)]
pub struct MergedModelInfo {
    pub info: Map<String, Value>,
    pub user_keys: Vec<String>,
}

pub(crate) fn raised(exception: &'static str, detail: impl Into<String>) -> Error {
    Error::ModelInfo {
        exception,
        detail: detail.into(),
    }
}

pub(crate) fn malformed(lookup: &str) -> Error {
    raised(
        "ValueError",
        format!("the host answered a `{lookup}` lookup malformed"),
    )
}

/// Python truthiness of a JSON value.
pub(crate) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => is_opaque(value) || !map.is_empty(),
    }
}

/// `(value or 0) > 0`, for a value that is a number (anything else reads as not positive).
fn positive_or_zero(value: Option<&Value>) -> bool {
    value
        .filter(|value| truthy(value))
        .and_then(Value::as_f64)
        .is_some_and(|n| n > 0.0)
}

pub(crate) async fn lookup(host: &dyn Host, lookup: ModelInfoLookup) -> Result<Value, Error> {
    host.model_info(&lookup).await
}

/// `get_model_info(model=...)`: the row, or `None` when the lookup tolerates a failure and failed.
async fn catalog_row(
    host: &dyn Host,
    model: Value,
    tolerate: bool,
) -> Result<Option<Map<String, Value>>, Error> {
    match lookup(
        host,
        ModelInfoLookup::ModelInfo {
            model,
            tolerate,
            exact: false,
        },
    )
    .await?
    {
        Value::Null => Ok(None),
        Value::Object(row) if tolerate && row.get("failed") == Some(&Value::Bool(true)) => Ok(None),
        Value::Object(row) => Ok(Some(row)),
        _ => Err(malformed("model_info")),
    }
}

/// `model_cost.get(key)`: `None` when the key is absent, else the entry as stored.
async fn cost_entry(host: &dyn Host, key: &str, process: bool) -> Result<Option<Value>, Error> {
    match lookup(
        host,
        ModelInfoLookup::CostEntry {
            key: key.to_owned(),
            process,
        },
    )
    .await?
    {
        Value::Null => Ok(None),
        Value::Object(mut answer) => Ok(Some(answer.remove("entry").unwrap_or(Value::Null))),
        _ => Err(malformed("cost_entry")),
    }
}

pub(crate) async fn discovered(host: &dyn Host, id: Value) -> Result<Map<String, Value>, Error> {
    match lookup(host, ModelInfoLookup::Discovered { id }).await? {
        Value::Null => Ok(Map::new()),
        Value::Object(limits) => Ok(limits),
        _ => Err(malformed("discovered")),
    }
}

async fn log(host: &dyn Host, level: LogLevel, message: String) -> Result<(), Error> {
    lookup(host, ModelInfoLookup::Log { level, message })
        .await
        .map(|_| ())
}

/// `_provider_model_id`: the model id a cost-map key or base model names, bedrock base model ids
/// normalized.
async fn provider_model_id(host: &dyn Host, name: &str, provider: &str) -> Result<String, Error> {
    let model_id = name.rsplit('/').next().unwrap_or(name).to_owned();
    if !provider.starts_with("bedrock") {
        return Ok(model_id);
    }
    match lookup(host, ModelInfoLookup::BedrockBase { model_id }).await? {
        Value::String(base) => Ok(base),
        _ => Err(malformed("bedrock_base")),
    }
}

/// `_exact_base_model_max_input_tokens`: the base model's own window, unless its row came from a
/// fallback-generalization rule or it is unmapped.
async fn exact_base_model_max_input_tokens(
    host: &dyn Host,
    base_model: &str,
) -> Result<Option<Value>, Error> {
    let answer = lookup(
        host,
        ModelInfoLookup::ModelInfo {
            model: Value::String(base_model.to_owned()),
            tolerate: true,
            exact: true,
        },
    )
    .await?;
    let Value::Object(mut answer) = answer else {
        return Ok(None);
    };
    if answer.get("failed") == Some(&Value::Bool(true))
        || answer.get("generalized") == Some(&Value::Bool(true))
    {
        return Ok(None);
    }
    let Some(Value::Object(row)) = answer.remove("row") else {
        return Ok(None);
    };
    Ok(row
        .get("max_input_tokens")
        .filter(|tokens| !tokens.is_null())
        .cloned())
}

/// `_base_model_widened_window`: a base model naming the same model as the route can only widen
/// its context window.
async fn base_model_widened_window(
    host: &dyn Host,
    route: &Map<String, Value>,
    base_model: Option<&str>,
    provider: &str,
) -> Result<Option<Value>, Error> {
    let Some(route_max) = route
        .get("max_input_tokens")
        .filter(|tokens| !tokens.is_null())
    else {
        return Ok(None);
    };
    let Some(base_model) = base_model else {
        return Ok(None);
    };
    let route_key = match route.get("key") {
        Some(Value::String(key)) => key.clone(),
        Some(_) => return Err(raised("AttributeError", "key")),
        None => return Err(raised("KeyError", "key")),
    };
    let route_id = provider_model_id(host, &route_key, provider).await?;
    let base_id = provider_model_id(host, base_model, provider).await?;
    let unversioned = base_id
        .split_once('@')
        .map_or(base_id.as_str(), |(id, _)| id);
    if route_id != base_id && route_id != unversioned {
        return Ok(None);
    }
    let Some(base_max) = exact_base_model_max_input_tokens(host, base_model).await? else {
        return Ok(None);
    };
    match (base_max.as_f64(), route_max.as_f64()) {
        (Some(base), Some(route)) if base > route => Ok(Some(base_max)),
        (Some(_), Some(_)) => Ok(None),
        _ => Err(raised(
            "TypeError",
            "'<=' not supported between the base model's and the route's max_input_tokens",
        )),
    }
}

/// `_declared_base_model`: a `base_model` that is a string.
fn declared_base_model(value: &Value) -> Option<&str> {
    value.as_str()
}

/// `get_router_model_info` after the host validated the deployment's params: the cost-map row of
/// the model the deployment serves, widened by its base model, overlaid with what the background
/// refresh discovered and then with every value the deployment sets itself. `None` when the
/// lookup answers `None`; an unmapped model raises as the lookup raises.
pub async fn router_model_info(
    host: &dyn Host,
    query: &RouterModelInfoQuery,
) -> Result<Option<MergedModelInfo>, Error> {
    let base_model = if query.model_info_base_model.is_null() {
        query.params_base_model.clone()
    } else {
        query.model_info_base_model.clone()
    };
    let mut model = base_model.clone();

    let (resolved_model, provider) = match lookup(
        host,
        ModelInfoLookup::Provider {
            model: query.params_model.clone(),
        },
    )
    .await?
    {
        Value::Array(pair) => match pair.as_slice() {
            [Value::String(model), Value::String(provider), ..] => {
                (model.clone(), provider.clone())
            }
            _ => return Err(malformed("provider")),
        },
        _ => return Err(malformed("provider")),
    };

    if provider == "azure" && base_model.is_null() {
        // Router init registers every deployment name into the cost map as a zeroed stub, so
        // membership alone cannot tell a resolvable name apart; it needs usable limits or costs.
        let fallback_key = if resolved_model.starts_with("azure/") {
            resolved_model.clone()
        } else {
            format!("azure/{resolved_model}")
        };
        let resolves = match cost_entry(host, &fallback_key, false).await? {
            Some(Value::Object(entry)) if !is_opaque(&Value::Object(entry.clone())) => {
                positive_or_zero(entry.get("max_input_tokens"))
                    || positive_or_zero(entry.get("input_cost_per_token"))
            }
            _ => false,
        };
        if resolves {
            log(
                host,
                LogLevel::Debug,
                format!(
                    "Azure deployment '{resolved_model}' has no base_model set; using '{fallback_key}' from the model cost map for max tokens, cost tracking, etc."
                ),
            )
            .await?;
        } else {
            log(
                host,
                LogLevel::Error,
                format!(
                    "Could not identify azure model '{resolved_model}'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models"
                ),
            )
            .await?;
        }
    } else if provider != "azure" {
        model = Value::String(resolved_model.clone());
        if resolved_model.contains('*') {
            // Only a wildcard deployment asks which model its pattern routed the name to.
            if let Value::Array(routed) = lookup(
                host,
                ModelInfoLookup::PatternModels {
                    name: query.received_model_name.clone(),
                },
            )
            .await?
                && let Some(routed) = routed
                    .iter()
                    .find(|routed| routed.get("id").unwrap_or(&Value::Null) == &query.deployment_id)
            {
                model = routed.get("model").cloned().unwrap_or(Value::Null);
            }
        }
    }

    if model.is_null() {
        // An azure deployment without a base_model is looked up by its own model.
        model = Value::String(resolved_model);
    }
    let Value::String(model) = model else {
        return Err(raised(
            "AttributeError",
            format!(
                "'{}' object has no attribute 'startswith'",
                crate::group_info::type_name(&model)
            ),
        ));
    };
    let model_info_name = if model.starts_with(&format!("{provider}/")) {
        model
    } else {
        format!("{provider}/{model}")
    };

    let Some(row) = catalog_row(host, Value::String(model_info_name), false).await? else {
        return Ok(None);
    };

    let user_model_info = match lookup(host, ModelInfoLookup::UserModelInfo).await? {
        Value::Object(info) => info,
        Value::Null => Map::new(),
        _ => return Err(malformed("user_model_info")),
    };
    let declared = user_model_info
        .get("base_model")
        .and_then(declared_base_model)
        .or_else(|| declared_base_model(&query.extra_base_model));
    let widened = base_model_widened_window(host, &row, declared, &provider).await?;
    let discovered = discovered(host, query.deployment_id.clone()).await?;

    let mut info = row;
    if let Some(max_input_tokens) = widened {
        info.insert("max_input_tokens".to_owned(), max_input_tokens);
    }
    info.extend(discovered);
    let mut user_keys = Vec::new();
    for (key, value) in &user_model_info {
        if !value.is_null() {
            info.insert(key.clone(), value.clone());
            user_keys.push(key.clone());
        }
    }
    Ok(Some(MergedModelInfo { info, user_keys }))
}

/// `_convert_stringified_numbers`: a string that reads as a number becomes one (a float when it
/// has an exponent or a point, else an int); any other string stays as written.
fn convert_stringified_number(text: &str) -> Value {
    let trimmed = text.trim();
    let digits = without_digit_underscores(trimmed);
    let lower = text.to_lowercase();
    if lower.contains('e') || text.contains('.') {
        return digits
            .and_then(|digits| python_float(&digits))
            .and_then(Number::from_f64)
            .map_or_else(|| Value::String(text.to_owned()), Value::Number);
    }
    digits
        .and_then(|digits| python_int(&digits))
        .map_or_else(|| Value::String(text.to_owned()), Value::Number)
}

/// The text with the underscores Python's `int()`/`float()` accept between digits removed, or
/// `None` when an underscore sits anywhere else.
pub(crate) fn without_digit_underscores(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (index, ch) in chars.iter().enumerate() {
        if *ch == '_' {
            let before = index.checked_sub(1).and_then(|i| chars.get(i));
            let after = chars.get(index + 1);
            if !(before.is_some_and(char::is_ascii_digit)
                && after.is_some_and(char::is_ascii_digit))
            {
                return None;
            }
            continue;
        }
        out.push(*ch);
    }
    Some(out)
}

pub(crate) fn python_float(text: &str) -> Option<f64> {
    let lower = text.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['+', '-']);
    if matches!(unsigned, "inf" | "infinity" | "nan") {
        // Not finite: JSON cannot carry it, so the value stays as written.
        return None;
    }
    let well_formed = text
        .chars()
        .all(|ch| ch.is_ascii_digit() || matches!(ch, '+' | '-' | '.' | 'e' | 'E'));
    if !well_formed {
        return None;
    }
    text.parse::<f64>().ok().filter(|n| n.is_finite())
}

pub(crate) fn python_int(text: &str) -> Option<Number> {
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    if unsigned.is_empty() || !unsigned.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let text = text.strip_prefix('+').unwrap_or(text);
    text.parse::<i64>()
        .map(Number::from)
        .ok()
        .or_else(|| text.parse::<u64>().map(Number::from).ok())
}

/// `update_dictionary`: `new`'s values over `existing`, skipping `None`s; a string that reads as a
/// number becomes one, and a dict merges one level into the dict it replaces.
pub fn update_dictionary(existing: &mut Map<String, Value>, new: &Map<String, Value>) {
    for (key, value) in new {
        match value {
            Value::Null => {}
            Value::String(text) => {
                existing.insert(key.clone(), convert_stringified_number(text));
            }
            Value::Object(nested) if !is_opaque(value) => {
                let merged = match existing.get(key) {
                    Some(current @ Value::Object(current_map)) if !is_opaque(current) => {
                        let mut merged = current_map.clone();
                        merged.extend(nested.clone());
                        merged
                    }
                    _ => nested.clone(),
                };
                existing.insert(key.clone(), Value::Object(merged));
            }
            _ => {
                existing.insert(key.clone(), value.clone());
            }
        }
    }
}

/// `cost_map_omits_token_price`: the cost-map entries under `keys` exist, and none carries a
/// per-token price.
async fn cost_map_omits_token_price(host: &dyn Host, keys: &[Option<&str>]) -> Result<bool, Error> {
    let mut entries = 0;
    for key in keys.iter().flatten() {
        if let Some(entry) = cost_entry(host, key, true).await?
            && !is_opaque(&entry)
            && let Value::Object(map) = entry
        {
            entries += 1;
            if map.contains_key("input_cost_per_token") || map.contains_key("output_cost_per_token")
            {
                return Ok(false);
            }
        }
    }
    Ok(entries > 0)
}

/// `get_deployment_model_info`: deployment `model_id`'s own cost-map entry (with what the refresh
/// discovered, over its base model's row), merged over the row of its litellm model `model_name`.
/// A lookup that fails leaves its layer out, as the Python router ignores it.
pub async fn deployment_model_info(
    host: &dyn Host,
    model_id: &str,
    model_name: &Value,
) -> Result<Option<Map<String, Value>>, Error> {
    let mut custom: Option<Map<String, Value>> = None;
    if let Some(entry) = cost_entry(host, model_id, false).await? {
        let entry = if truthy(&entry) {
            match entry {
                Value::Object(map) if !is_opaque(&Value::Object(map.clone())) => Some(map),
                // Not a mapping: the Python router's TypeError, which it ignores.
                _ => None,
            }
        } else {
            Some(Map::new())
        };
        if let Some(mut entry) = entry {
            entry.extend(discovered(host, Value::String(model_id.to_owned())).await?);
            custom = Some(entry);
        }
    }

    let builtin = catalog_row(host, model_name.clone(), true).await?;

    let mut base_model_key: Option<String> = None;
    if let Some(current) = &custom
        && let Some(base_model) = current.get("base_model").filter(|base| !base.is_null())
        && let Some(mut base_row) = catalog_row(host, base_model.clone(), true).await?
    {
        base_model_key = base_row
            .get("key")
            .and_then(Value::as_str)
            .map(str::to_owned);
        // The base model provides defaults; the deployment's own entry overrides them.
        update_dictionary(&mut base_row, current);
        custom = Some(base_row);
    }

    let builtin_key = builtin
        .as_ref()
        .and_then(|row| row.get("key"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let info = match (custom, builtin) {
        (Some(custom), Some(mut builtin)) => {
            update_dictionary(&mut builtin, &custom);
            builtin
        }
        (None, Some(builtin)) => builtin,
        (Some(custom), None) => custom,
        (None, None) => return Ok(None),
    };
    if cost_map_omits_token_price(
        host,
        &[
            Some(model_id),
            builtin_key.as_deref(),
            base_model_key.as_deref(),
        ],
    )
    .await?
    {
        let mut info = info;
        info.insert("input_cost_per_token".to_owned(), Value::Null);
        info.insert("output_cost_per_token".to_owned(), Value::Null);
        return Ok(Some(info));
    }
    Ok(Some(info))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{
        cache::{InMemoryRouterCache, RouterCache},
        error::AttemptError,
        host::{Attempt, Output, RouterEvent, RouterRequest, RoutingContext, SelectStage},
        registry::Deployment,
    };

    /// A cost map and a provider, answering lookups the way `PythonHost` does.
    #[derive(Default)]
    struct Catalog {
        rows: BTreeMap<String, Value>,
        provider: (String, String),
        user_model_info: Value,
        lookups: Mutex<Vec<Value>>,
    }

    #[async_trait]
    impl Host for Catalog {
        async fn invoke(&self, _attempt: &Attempt) -> Result<Output, AttemptError> {
            unreachable!("a model-info query runs no attempt")
        }
        async fn select(
            &self,
            _stage: &SelectStage,
            _ctx: &RoutingContext<'_>,
            candidates: Vec<Arc<Deployment>>,
        ) -> Result<Vec<Arc<Deployment>>, Error> {
            Ok(candidates)
        }
        async fn allow_fallback(
            &self,
            _request: &RouterRequest,
            _target: &str,
        ) -> Result<bool, crate::error::AttemptError> {
            Ok(true)
        }
        async fn sleep(&self, _duration: std::time::Duration) {}
        fn observe(&self, _event: &RouterEvent) {}
        fn cache(&self) -> Arc<dyn RouterCache> {
            Arc::new(InMemoryRouterCache::default())
        }
        async fn model_info(&self, lookup: &ModelInfoLookup) -> Result<Value, Error> {
            self.lookups
                .lock()
                .unwrap()
                .push(serde_json::to_value(lookup).unwrap());
            Ok(match lookup {
                ModelInfoLookup::ModelInfo {
                    model, tolerate, ..
                } => match model.as_str().and_then(|model| self.rows.get(model)) {
                    Some(row) => row.clone(),
                    None if *tolerate => json!({"failed": true}),
                    None => return Err(raised("ValueError", "unmapped")),
                },
                ModelInfoLookup::CostEntry { key, .. } => self
                    .rows
                    .get(key)
                    .map_or(Value::Null, |entry| json!({"entry": entry})),
                ModelInfoLookup::Provider { .. } => {
                    json!([self.provider.0, self.provider.1])
                }
                ModelInfoLookup::UserModelInfo => self.user_model_info.clone(),
                _ => Value::Null,
            })
        }
    }

    #[test]
    fn update_dictionary_converts_numbers_and_merges_nested_dicts_one_level() {
        let mut existing = json!({"a": 1, "n": {"x": 1, "y": 2}, "k": "keep"})
            .as_object()
            .unwrap()
            .clone();
        let new = json!({"a": None::<i64>, "n": {"y": 3}, "i": "8_192", "f": "3e-07", "s": "abc"})
            .as_object()
            .unwrap()
            .clone();
        update_dictionary(&mut existing, &new);
        assert_eq!(
            Value::Object(existing),
            json!({"a": 1, "n": {"x": 1, "y": 3}, "k": "keep", "i": 8192, "f": 3e-07, "s": "abc"})
        );
    }

    #[tokio::test]
    async fn router_model_info_overlays_the_deployments_own_values_on_the_row() {
        let host = Catalog {
            rows: BTreeMap::from([(
                "openai/gpt-4o".to_owned(),
                json!({"key": "gpt-4o", "max_input_tokens": 100, "input_cost_per_token": 1.0}),
            )]),
            provider: ("gpt-4o".to_owned(), "openai".to_owned()),
            user_model_info: json!({"id": "d", "input_cost_per_token": 2.0, "mode": None::<String>}),
            ..Catalog::default()
        };
        let query = RouterModelInfoQuery {
            deployment_id: json!("d"),
            params_model: "openai/gpt-4o".to_owned(),
            received_model_name: "g".to_owned(),
            ..RouterModelInfoQuery::default()
        };
        let merged = router_model_info(&host, &query).await.unwrap().unwrap();
        assert_eq!(
            Value::Object(merged.info),
            json!({"key": "gpt-4o", "max_input_tokens": 100, "input_cost_per_token": 2.0, "id": "d"})
        );
        assert_eq!(merged.user_keys, vec!["id", "input_cost_per_token"]);
    }

    #[tokio::test]
    async fn deployment_model_info_merges_the_entry_over_the_builtin_row() {
        let host = Catalog {
            rows: BTreeMap::from([
                (
                    "gpt-4o".to_owned(),
                    json!({"key": "gpt-4o", "max_tokens": 10, "input_cost_per_token": 1.0}),
                ),
                ("d".to_owned(), json!({"max_tokens": "20"})),
            ]),
            ..Catalog::default()
        };
        let info = deployment_model_info(&host, "d", &json!("gpt-4o"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            Value::Object(info),
            json!({"key": "gpt-4o", "max_tokens": 20, "input_cost_per_token": 1.0})
        );
    }

    #[tokio::test]
    async fn an_unpriced_entry_reports_no_token_prices() {
        let host = Catalog {
            rows: BTreeMap::from([("d".to_owned(), json!({"max_tokens": 5}))]),
            ..Catalog::default()
        };
        let info = deployment_model_info(&host, "d", &json!("unmapped"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            Value::Object(info),
            json!({"max_tokens": 5, "input_cost_per_token": null, "output_cost_per_token": null})
        );
    }
}
