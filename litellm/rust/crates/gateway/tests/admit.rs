//! Each file in `tests/admit/` is one config and the requests sent to the gateway built from it,
//! each with who answers it, `rust` or `python`. A request for Python names the `reason` the client
//! sees in `x-litellm-gateway`, and may name the setting or trait it goes to Python `over`.

mod keys;

use std::{collections::BTreeMap, path::Path, sync::Arc};

use axum::{body::Body, extract::Request, http::Method};
use litellm_config::{Config, Environment};
use litellm_gateway::{StoredConfig, host::auth::AuthStore};
use rstest::rstest;
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    config: serde_yaml_ng::Value,
    /// The virtual keys the database holds, each token with its model list.
    #[serde(default)]
    keys: BTreeMap<String, Vec<String>>,
    /// The `tpm_limit` of each key that holds one.
    #[serde(default)]
    tpm_limits: BTreeMap<String, i64>,
    /// What boot read of the config the proxy stores in its database.
    #[serde(default)]
    stored: Stored,
    requests: Vec<Sent>,
}

/// `unused` when `store_model_in_db` is off, `loaded` when the config holds what the database
/// stores, `unread` when the database could not be read, and `ui_settings_unread` when it is off
/// and the UI settings could not be read or layered in.
#[derive(Clone, Copy, Default, Deserialize)]
enum Stored {
    #[default]
    #[serde(rename = "unused")]
    Unused,
    #[serde(rename = "loaded")]
    Loaded,
    #[serde(rename = "unread")]
    Unread,
    #[serde(rename = "ui_settings_unread")]
    UiSettingsUnread,
}

impl From<Stored> for StoredConfig {
    fn from(stored: Stored) -> Self {
        match stored {
            Stored::Unused => Self::Unused,
            Stored::Loaded => Self::Loaded,
            Stored::Unread => Self::Unread,
            Stored::UiSettingsUnread => Self::UiSettingsUnread,
        }
    }
}

/// The database holding each key of the scenario, a plain row with the model list and the token
/// limit it gives.
async fn listed(
    keys: &BTreeMap<String, Vec<String>>,
    tpm_limits: &BTreeMap<String, i64>,
) -> keys::Held {
    let held = keys::Held::start().await;
    for (token, models) in keys {
        let columns = serde_json::json!({"models": models, "tpm_limit": tpm_limits.get(token)});
        held.key(token, columns).await;
    }
    held
}

#[derive(Deserialize)]
struct Sent {
    name: String,
    #[serde(default = "post")]
    method: String,
    #[serde(default = "messages")]
    path: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    /// The body as JSON, or `raw` for bytes that are not.
    body: Option<serde_json::Value>,
    raw: Option<String>,
    expect: Who,
    reason: Option<String>,
    over: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
enum Who {
    #[serde(rename = "rust")]
    Rust,
    #[serde(rename = "python")]
    Python,
}

fn post() -> String {
    Method::POST.to_string()
}

fn messages() -> String {
    "/v1/messages".to_owned()
}

impl Sent {
    fn request(&self) -> Request {
        let body = match (&self.body, &self.raw) {
            (Some(json), _) => Body::from(json.to_string()),
            (None, Some(raw)) => Body::from(raw.clone()),
            (None, None) => Body::empty(),
        };
        self.headers
            .iter()
            .fold(
                Request::builder()
                    .method(self.method.as_str())
                    .uri(&self.path),
                |request, (name, value)| request.header(name, value),
            )
            .body(body)
            .unwrap()
    }
}

#[rstest]
#[case::mixed_groups("mixed_groups")]
#[case::honored_settings("honored_settings")]
#[case::setting_for_every_request("setting_for_every_request")]
#[case::setting_for_messages_only("setting_for_messages_only")]
#[case::fallback_shapes("fallback_shapes")]
#[case::drop_params_set("drop_params_set")]
#[case::database("database")]
#[case::store_model_in_db("store_model_in_db")]
#[case::store_model_in_db_unread("store_model_in_db_unread")]
#[case::ui_settings_unread("ui_settings_unread")]
#[case::store_model_in_db_objects("store_model_in_db_objects")]
#[case::router_settings_by_group("router_settings_by_group")]
#[case::groups_and_fallbacks("groups_and_fallbacks")]
#[case::model_aliases("model_aliases")]
#[case::router_setting_for_every_request("router_setting_for_every_request")]
#[case::router_host_steps("router_host_steps")]
#[case::callers("callers")]
#[case::virtual_key_models("virtual_key_models")]
#[case::virtual_key_token_limits("virtual_key_token_limits")]
#[case::dynamic_rate_limiter("dynamic_rate_limiter")]
#[case::dynamic_rate_limiter_borrowing("dynamic_rate_limiter_borrowing")]
#[case::pre_call_checks("pre_call_checks")]
#[tokio::test(flavor = "multi_thread")]
async fn admit_answers_each_request_as_its_scenario_says(#[case] name: &str) {
    litellm_gateway::install_crypto_provider();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/admit/{name}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let scenario: Scenario =
        serde_yaml_ng::from_value(litellm_yaml::from_str(&text).unwrap()).unwrap();
    let config = Config::from_value(scenario.config).unwrap();
    let resources =
        litellm_gateway::Resources::over(&config, Arc::new(|_: &str| None::<String>)).unwrap();
    let state = litellm_gateway::AppState::over(&config, &Environment::default(), &resources)
        .await
        .unwrap();
    let held = match scenario.keys.is_empty() {
        true => None,
        false => Some(listed(&scenario.keys, &scenario.tpm_limits).await),
    };
    let state = match &held {
        None => state,
        Some(held) => litellm_gateway::AppState {
            keys: Some(Arc::new(AuthStore::new(
                keys::memory_cache(),
                held.database(),
            ))),
            ..state
        },
    };
    let admit = Arc::new(litellm_gateway::Admit::new(
        &config,
        &Environment::default(),
        &state,
        scenario.stored.into(),
    ));

    let answered = futures_util::future::join_all(scenario.requests.iter().map(|sent| {
        let admit = admit.clone();
        async move {
            let (who, reason, over) = match admit.admit(sent.request()).await {
                Ok(_) => (Who::Rust, None, None),
                Err((_, reason)) => (
                    Who::Python,
                    Some(<&str>::from(reason.why).to_owned()),
                    Some(String::from(reason.over)),
                ),
            };
            (
                sent.name.as_str(),
                who,
                reason,
                over.filter(|_| sent.over.is_some()),
            )
        }
    }))
    .await;
    state.close().await;

    let expected: Vec<_> = scenario
        .requests
        .iter()
        .map(|sent| {
            (
                sent.name.as_str(),
                sent.expect,
                sent.reason.clone(),
                sent.over.clone(),
            )
        })
        .collect();
    assert_eq!(answered, expected);
}
