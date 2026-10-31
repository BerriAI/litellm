//! The documents `RouterCoreHandle` reads: the router's static and dynamic config, a call's
//! request and an `apply` change set. A config key router-core does not read declines the
//! handle, so Python keeps the router on its own core.

use std::collections::BTreeMap;

use litellm_router_core::{
    ChangeSet, ConfigError, DeploymentSpec, RouterConfig, RouterDynamicConfig, RouterRequest,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};

/// The `RouterConfig` keys router-core reads.
const STATIC_KEYS: &[&str] = &[
    "model_list",
    "routing_strategy",
    "routing_groups",
    "plugins",
    "cooldown_time",
    "disable_cooldowns",
    "allowed_fails_policy",
    "retry_policy",
    "retry_after",
    "timeout",
    "request_timeout",
    "drop_params",
    "enable_pre_call_checks",
    "enable_health_check_routing",
    "provider_default_deployment_ids",
];

/// The `RouterDynamicConfig` keys router-core reads.
const DYNAMIC_KEYS: &[&str] = &[
    "version",
    "model_alias_map",
    "model_group_alias",
    "fallbacks",
    "default_fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "num_retries",
    "max_fallbacks",
    "allowed_fails",
    "fallback_keys",
];

/// Router arguments that do not change routing, accepted in either config and dropped.
const IGNORED_KEYS: &[&str] = &[
    "set_verbose",
    "debug_level",
    "alerting_config",
    "client_ttl",
];

/// Fallback lists Python treats as unset when empty (`context_window_fallbacks or None`).
const EMPTY_MEANS_UNSET: &[&str] = &["context_window_fallbacks", "content_policy_fallbacks"];

/// Which document a failure is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub(crate) enum Document {
    #[strum(serialize = "router config")]
    Static,
    #[strum(serialize = "dynamic router config")]
    Dynamic,
    #[strum(serialize = "router request")]
    Request,
    #[strum(serialize = "change set")]
    Changes,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("the {document} must be a JSON object")]
    NotObject { document: Document },
    #[error("router-core does not read `{key}` in the {document}")]
    UnknownKey { document: Document, key: String },
    #[error("router-core does not run `{key}` = {value}")]
    UnsupportedValue { key: &'static str, value: Value },
    #[error("the {document} does not have the shape router-core reads")]
    Shape {
        document: Document,
        #[source]
        source: serde_json::Error,
    },
    #[error("the request names no model")]
    MissingModel,
    #[error("router-core rejected the router config")]
    Core(#[from] ConfigError),
}

impl Error {
    /// Whether router-core cannot serve this config, so Python keeps its own core.
    pub(crate) fn declined(&self) -> bool {
        match self {
            Self::UnknownKey { .. } | Self::UnsupportedValue { .. } | Self::Core(_) => true,
            Self::Shape { document, .. } => {
                matches!(document, Document::Static | Document::Dynamic)
            }
            Self::NotObject { .. } | Self::MissingModel => false,
        }
    }
}

fn object(document: Document, value: Value) -> Result<Map<String, Value>, Error> {
    match value {
        Value::Object(object) => Ok(object),
        _ => Err(Error::NotObject { document }),
    }
}

fn shaped<T: DeserializeOwned>(document: Document, object: Map<String, Value>) -> Result<T, Error> {
    serde_json::from_value(Value::Object(object))
        .map_err(|source| Error::Shape { document, source })
}

/// Drop unset and ignored keys, then reject any key outside `known`.
fn routing_keys(
    document: Document,
    value: Value,
    known: &[&str],
) -> Result<Map<String, Value>, Error> {
    let mut object = object(document, value)?;
    object.retain(|key, value| {
        let empty_unset = EMPTY_MEANS_UNSET.contains(&key.as_str())
            && value.as_array().is_some_and(Vec::is_empty);
        !(value.is_null() || empty_unset || IGNORED_KEYS.contains(&key.as_str()))
    });
    if let Some(key) = object.keys().find(|key| !known.contains(&key.as_str())) {
        return Err(Error::UnknownKey {
            document,
            key: key.clone(),
        });
    }
    Ok(object)
}

pub(crate) fn router_config(value: Value) -> Result<RouterConfig, Error> {
    let object = routing_keys(Document::Static, value, STATIC_KEYS)?;
    // Pre-call checks reach router-core as plugins; the Python flag alone has no native filter.
    if let Some(value) = object.get("enable_pre_call_checks")
        && value != &Value::Bool(false)
    {
        return Err(Error::UnsupportedValue {
            key: "enable_pre_call_checks",
            value: value.clone(),
        });
    }
    shaped(Document::Static, object)
}

pub(crate) fn dynamic_config(value: Value) -> Result<RouterDynamicConfig, Error> {
    let object = routing_keys(Document::Dynamic, value, DYNAMIC_KEYS)?;
    shaped(Document::Dynamic, object)
}

/// A call's request as Python spells it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestSpec {
    model: Option<String>,
    team_id: Option<String>,
    specific_deployment: Option<bool>,
    num_retries: Option<u32>,
    disable_fallbacks: Option<bool>,
    router_weights: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    body: Option<Value>,
}

/// Read a request; `model` replaces the request's own when given.
pub(crate) fn request(value: Value, model: Option<&str>) -> Result<RouterRequest, Error> {
    let spec: RequestSpec = shaped(Document::Request, object(Document::Request, value)?)?;
    let model = model
        .map(str::to_owned)
        .or(spec.model)
        .ok_or(Error::MissingModel)?;
    Ok(RouterRequest {
        model,
        body: spec.body.unwrap_or_default(),
        team_id: spec.team_id,
        specific_deployment: spec.specific_deployment.unwrap_or(false),
        num_retries: spec.num_retries,
        disable_fallbacks: spec.disable_fallbacks.unwrap_or(false),
        router_weights: spec.router_weights,
        payload: None,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChangesSpec {
    #[serde(default)]
    upsert: Vec<DeploymentSpec>,
    #[serde(default)]
    delete: Vec<String>,
}

pub(crate) fn changes(value: Value) -> Result<ChangeSet, Error> {
    let spec: ChangesSpec = shaped(Document::Changes, object(Document::Changes, value)?)?;
    Ok(ChangeSet {
        upserts: spec.upsert,
        deletes: spec.delete,
        settings: None,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::unknown_key(json!({"routing_strategy_args": {"ttl": 1}}), "routing_strategy_args")]
    #[case::dynamic_key_in_static(json!({"num_retries": 2}), "num_retries")]
    fn an_unread_static_key_declines_naming_the_key(#[case] json: Value, #[case] key: &str) {
        let error = router_config(json).unwrap_err();
        assert!(error.declined());
        assert!(matches!(&error, Error::UnknownKey { key: named, .. } if named == key));
    }

    #[rstest]
    #[case::null_unknown(json!({"routing_strategy_args": null}))]
    #[case::ignored(json!({"set_verbose": true, "debug_level": "INFO"}))]
    #[case::pre_call_checks_off(json!({"enable_pre_call_checks": false}))]
    fn unset_and_ignored_keys_are_accepted(#[case] json: Value) {
        router_config(json).unwrap();
    }

    #[rstest]
    fn pre_call_checks_without_a_plugin_decline() {
        let error = router_config(json!({"enable_pre_call_checks": true})).unwrap_err();
        assert!(error.declined());
        assert!(matches!(
            error,
            Error::UnsupportedValue {
                key: "enable_pre_call_checks",
                ..
            }
        ));
    }

    #[rstest]
    #[case::not_object(json!([]), false)]
    #[case::bad_shape(json!({"retry_after": "soon"}), true)]
    fn a_config_that_is_not_an_object_is_a_value_error_and_a_wrong_shape_declines(
        #[case] json: Value,
        #[case] declined: bool,
    ) {
        assert_eq!(router_config(json).unwrap_err().declined(), declined);
    }

    #[rstest]
    fn python_router_defaults_read_as_unset() {
        let dynamic = dynamic_config(json!({
            "num_retries": null, "fallbacks": [], "context_window_fallbacks": [],
            "content_policy_fallbacks": [], "model_group_alias": {"gpt": "gpt-4"},
        }))
        .unwrap();
        assert_eq!(dynamic.num_retries, None);
        assert!(dynamic.context_window_fallbacks.is_none());
        assert!(dynamic.content_policy_fallbacks.is_none());
        assert_eq!(dynamic.model_alias_map["gpt"], "gpt-4");
    }

    #[rstest]
    #[case::own_model(json!({"model": "gpt", "num_retries": 1}), None, "gpt")]
    #[case::override_model(json!({"model": "gpt"}), Some("other"), "other")]
    fn a_request_reads_its_model(
        #[case] json: Value,
        #[case] model: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(request(json, model).unwrap().model, expected);
    }

    #[rstest]
    #[case::unknown_field(json!({"model": "gpt", "temperature": 1}))]
    #[case::no_model(json!({"team_id": "t"}))]
    fn a_bad_request_is_a_value_error(#[case] json: Value) {
        assert!(!request(json, None).unwrap_err().declined());
    }

    #[rstest]
    fn a_change_set_reads_upserts_and_deletes() {
        let changes = changes(json!({
            "upsert": [{"model_name": "m", "litellm_params": {"model": "openai/x"}}],
            "delete": ["old"],
        }))
        .unwrap();
        assert_eq!(changes.upserts.len(), 1);
        assert_eq!(changes.deletes, ["old"]);
    }
}
