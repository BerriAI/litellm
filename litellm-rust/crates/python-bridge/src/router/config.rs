//! The documents `RouterCoreHandle` reads: the router's static and dynamic config, a call's
//! request and an `apply` change set. A config key router-core does not read declines the
//! handle, so Python keeps the router on its own core.

use std::collections::BTreeMap;

use litellm_router_core::select::PassThrough;
use litellm_router_core::{
    ChangeSet, ConfigError, DeploymentSpec, RouterConfig, RouterDynamicConfig, RouterRequest,
    config::{Fallbacks, RetryPolicy},
    mock::MockFailure,
    registry::IndexOverrides,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value};

/// The `RouterConfig` keys router-core reads.
const STATIC_KEYS: &[&str] = &[
    "model_list",
    "routing_strategy",
    "routing_groups",
    "routing_group_members",
    "plugins",
    "cooldown_time",
    "disable_cooldowns",
    "allowed_fails_policy",
    "retry_policy",
    "model_group_retry_policy",
    "retry_after",
    "timeout",
    "request_timeout",
    "drop_params",
    "enable_pre_call_checks",
    "routing_plugins",
    "callback_filters",
    "enable_health_check_routing",
    "health_check_staleness_threshold",
    "background_health_check_model_groups",
    "enable_weighted_failover",
    "fallback_on_unsupported_params",
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
    "default_deployment",
    "default_deployment_raises",
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
            Self::UnknownKey { .. } | Self::Core(_) => true,
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
    proxy_admin: Option<bool>,
    allowed_models: Option<Vec<String>>,
    admin_model: Option<String>,
    specific_deployment: Option<bool>,
    num_retries: Option<u32>,
    disable_fallbacks: Option<bool>,
    router_weights: Option<Value>,
    pinned_deployment_id: Option<String>,
    routing_strategy: Option<String>,
    filter_tags: Option<bool>,
    /// `listed` or `healthy`: which pass-through pick the request makes.
    pass_through: Option<String>,
    target_order: Option<i64>,
    excluded_ids: Option<Vec<String>>,
    retry_skipped_ids: Option<Vec<String>>,
    pre_routes: Option<bool>,
    has_image: Option<bool>,
    unrouted: Option<bool>,
    model_group_retry_policy: Option<BTreeMap<String, RetryPolicy>>,
    retry_policy_raises: Option<Vec<String>>,
    #[serde(default, deserialize_with = "present")]
    fallbacks: Option<Option<Fallbacks>>,
    #[serde(default, deserialize_with = "present")]
    context_window_fallbacks: Option<Option<Fallbacks>>,
    #[serde(default, deserialize_with = "present")]
    content_policy_fallbacks: Option<Option<Fallbacks>>,
    max_fallbacks: Option<u32>,
    mock_failure: Option<MockFailure>,
    cooldown_time: Option<f64>,
    health_check_probe: Option<bool>,
    body: Option<Value>,
}

/// A key the request passed, even as `None`: `Some(None)`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
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
        proxy_admin: spec.proxy_admin.unwrap_or(false),
        allowed_models: spec.allowed_models.unwrap_or_default(),
        admin_model: spec.admin_model,
        specific_deployment: spec.specific_deployment.unwrap_or(false),
        num_retries: spec.num_retries,
        disable_fallbacks: spec.disable_fallbacks.unwrap_or(false),
        router_weights: spec.router_weights,
        pinned_deployment_id: spec.pinned_deployment_id,
        routing_strategy: spec.routing_strategy,
        filter_tags: spec.filter_tags.unwrap_or(false),
        pass_through: match spec.pass_through.as_deref() {
            None => None,
            Some("healthy") => Some(PassThrough::Healthy),
            Some(_) => Some(PassThrough::Listed),
        },
        target_order: spec.target_order,
        excluded_ids: spec.excluded_ids.unwrap_or_default(),
        retry_skipped_ids: spec.retry_skipped_ids.unwrap_or_default(),
        pre_routes: spec.pre_routes.unwrap_or(false),
        has_image: spec.has_image.unwrap_or(false),
        unrouted: spec.unrouted.unwrap_or(false),
        model_group_retry_policy: spec.model_group_retry_policy,
        retry_policy_raises: spec.retry_policy_raises.unwrap_or_default(),
        fallbacks: spec.fallbacks,
        context_window_fallbacks: spec.context_window_fallbacks,
        content_policy_fallbacks: spec.content_policy_fallbacks,
        max_fallbacks: spec.max_fallbacks,
        mock_failure: spec.mock_failure,
        cooldown_time: spec.cooldown_time,
        health_check_probe: spec.health_check_probe.unwrap_or(false),
        payload: None,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChangesSpec {
    /// Every deployment, replacing the current list before the upserts and deletes.
    #[serde(default)]
    replace: Option<Vec<DeploymentSpec>>,
    #[serde(default)]
    upsert: Vec<DeploymentSpec>,
    #[serde(default)]
    delete: Vec<String>,
    /// A new dynamic document (the same keys `RouterCoreHandle(static, dynamic)` takes).
    #[serde(default)]
    settings: Option<Value>,
    /// The registry indexes the host wrote (`IndexOverrides`); `{}` drops them.
    #[serde(default)]
    index: Option<IndexOverrides>,
}

pub(crate) fn changes(value: Value) -> Result<ChangeSet, Error> {
    let spec: ChangesSpec = shaped(Document::Changes, object(Document::Changes, value)?)?;
    Ok(ChangeSet {
        replace: spec.replace,
        upserts: spec.upsert,
        deletes: spec.delete,
        settings: spec.settings.map(dynamic_config).transpose()?,
        index: spec.index,
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
    #[case::pre_call_checks_on(json!({"enable_pre_call_checks": true}))]
    fn unset_and_ignored_keys_are_accepted(#[case] json: Value) {
        router_config(json).unwrap();
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
