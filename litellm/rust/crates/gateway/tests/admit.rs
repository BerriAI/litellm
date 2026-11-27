//! Each file in `tests/admit/` is one config and the requests sent to the gateway built from it,
//! each with who answers it, `rust` or `python`. A request for Python names the `reason` the client
//! sees in `x-litellm-gateway`, and may name the setting or trait it goes to Python `over`.

mod keys;

use std::{collections::BTreeMap, net::SocketAddr, path::Path, sync::Arc};

use axum::{
    body::Body,
    extract::{ConnectInfo, Request},
    http::Method,
};
use litellm_config::{Config, Environment};
use litellm_gateway::{StoredConfig, host::auth::AuthStore};
use litellm_model_catalog::{CostMap, IntegrityLimits};
use litellm_repositories::StoredObjects;
use rstest::rstest;
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    config: serde_yaml_ng::Value,
    /// The virtual keys the database holds, each token with the columns set on its row.
    #[serde(default)]
    keys: BTreeMap<String, Columns>,
    /// The other rows the database holds, each under its table's name, stored in this order.
    #[serde(default)]
    rows: Vec<BTreeMap<String, serde_json::Value>>,
    /// Whether the config reads the database that holds the keys, as its
    /// `general_settings.database_url`.
    #[serde(default)]
    database: bool,
    /// What boot read of the config the proxy stores in its database.
    #[serde(default)]
    stored: Stored,
    /// Which other stored objects the database holds, when `stored` is `loaded`.
    #[serde(default)]
    held: Held,
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

/// Each field of `StoredObjects`, `false` when the scenario leaves it out.
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Held {
    default_on_guardrails: bool,
    policy_attachments: bool,
    tool_policies: bool,
    cache_settings: bool,
    config_overrides: bool,
    anthropic_beta_headers_reload: bool,
    model_cost_map_reload: bool,
}

impl From<Held> for StoredObjects {
    fn from(held: Held) -> Self {
        let Held {
            default_on_guardrails,
            policy_attachments,
            tool_policies,
            cache_settings,
            config_overrides,
            anthropic_beta_headers_reload,
            model_cost_map_reload,
        } = held;
        Self {
            default_on_guardrails,
            policy_attachments,
            tool_policies,
            cache_settings,
            config_overrides,
            anthropic_beta_headers_reload,
            model_cost_map_reload,
        }
    }
}

fn stored_config(stored: Stored, held: Held) -> StoredConfig {
    match stored {
        Stored::Unused => StoredConfig::Unused,
        Stored::Loaded => StoredConfig::Loaded(held.into()),
        Stored::Unread => StoredConfig::Unread,
        Stored::UiSettingsUnread => StoredConfig::UiSettingsUnread,
    }
}

type Columns = serde_json::Map<String, serde_json::Value>;

/// The database holding each key of the scenario, a plain row with its columns set on it, and
/// each of its other rows.
async fn listed(
    keys: &BTreeMap<String, Columns>,
    rows: &[BTreeMap<String, serde_json::Value>],
) -> keys::Held {
    let held = keys::Held::start().await;
    for (token, columns) in keys {
        held.key(token, serde_json::Value::Object(columns.clone()))
            .await;
    }
    for (table, row) in rows.iter().flatten() {
        held.insert(table, row).await;
    }
    held
}

/// `config` reading the database at `url`.
fn with_database(mut config: serde_yaml_ng::Value, url: &str) -> serde_yaml_ng::Value {
    config["general_settings"]["database_url"] = url.into();
    config
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
    /// The client's TCP peer, which the IP allowlist reads.
    peer: Option<SocketAddr>,
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
                self.peer.into_iter().fold(
                    Request::builder()
                        .method(self.method.as_str())
                        .uri(&self.path),
                    |request, peer| request.extension(ConnectInfo(peer)),
                ),
                |request, (name, value)| request.header(name, value),
            )
            .body(body)
            .unwrap()
    }
}

#[rstest]
#[case::mixed_groups("mixed_groups")]
#[case::honored_settings("honored_settings")]
#[case::allowed_routes("allowed_routes")]
#[case::allowed_routes_unwritten("allowed_routes_unwritten")]
#[case::allowed_ips("allowed_ips")]
#[case::setting_for_every_request("setting_for_every_request")]
#[case::setting_for_messages_only("setting_for_messages_only")]
#[case::fallback_shapes("fallback_shapes")]
#[case::drop_params_set("drop_params_set")]
#[case::database("database")]
#[case::store_model_in_db("store_model_in_db")]
#[case::store_model_in_db_unread("store_model_in_db_unread")]
#[case::ui_settings_unread("ui_settings_unread")]
#[case::environment_variables("environment_variables")]
#[case::store_model_in_db_objects("store_model_in_db_objects")]
#[case::store_model_in_db_nothing_held("store_model_in_db_nothing_held")]
#[case::router_settings_by_group("router_settings_by_group")]
#[case::groups_and_fallbacks("groups_and_fallbacks")]
#[case::model_aliases("model_aliases")]
#[case::router_setting_for_every_request("router_setting_for_every_request")]
#[case::router_host_steps("router_host_steps")]
#[case::callers("callers")]
#[case::virtual_key_models("virtual_key_models")]
#[case::user_api_key_cache_settings("user_api_key_cache_settings")]
#[case::virtual_key_token_limits("virtual_key_token_limits")]
#[case::dynamic_rate_limiter("dynamic_rate_limiter")]
#[case::dynamic_rate_limiter_borrowing("dynamic_rate_limiter_borrowing")]
#[case::messages_settings("messages_settings")]
#[case::request_timeout_float_refuses("request_timeout_float_refuses")]
#[case::request_timeout_beside_router_timeout("request_timeout_beside_router_timeout")]
#[case::request_timeout_beside_router_request_timeout(
    "request_timeout_beside_router_request_timeout"
)]
#[case::pre_call_checks("pre_call_checks")]
#[case::claude_only_providers("claude_only_providers")]
#[case::cloud_providers("cloud_providers")]
#[case::budget_features("budget_features")]
#[case::proxy_budget("proxy_budget")]
#[case::budget_alerting("budget_alerting")]
#[case::budget_throttle("budget_throttle")]
#[case::budget_status_code("budget_status_code")]
#[case::virtual_key_budgets("virtual_key_budgets")]
#[case::virtual_key_budgets_reserved("virtual_key_budgets_reserved")]
#[case::virtual_key_budgets_alerting("virtual_key_budgets_alerting")]
#[case::virtual_key_budgets_without_database("virtual_key_budgets_without_database")]
#[case::end_user("end_user")]
#[case::end_user_token_database("end_user_token_database")]
#[tokio::test(flavor = "multi_thread")]
async fn admit_answers_each_request_as_its_scenario_says(#[case] name: &str) {
    litellm_gateway::install_crypto_provider();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/admit/{name}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let scenario: Scenario =
        serde_yaml_ng::from_value(litellm_yaml::from_str(&text).unwrap()).unwrap();
    let held = match scenario.keys.is_empty() && scenario.rows.is_empty() {
        true => None,
        false => Some(listed(&scenario.keys, &scenario.rows).await),
    };
    let config = match (&held, scenario.database) {
        (Some(held), true) => with_database(scenario.config.clone(), held.url()),
        _ => scenario.config.clone(),
    };
    let config = Config::from_value(config).unwrap();
    let resources =
        litellm_gateway::Resources::over(&config, Arc::new(|_: &str| None::<String>)).unwrap();
    let state = litellm_gateway::AppState::over(&config, &Environment::default(), &resources)
        .await
        .unwrap();
    let state = match &held {
        None => state,
        Some(held) => litellm_gateway::AppState {
            auth_store: Some(Arc::new(AuthStore::new(
                keys::memory_cache(),
                held.database(),
            ))),
            ..state
        },
    };
    let admit = Arc::new(litellm_gateway::Admit::new(
        &config,
        &state,
        stored_config(scenario.stored, scenario.held),
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

#[tokio::test(flavor = "multi_thread")]
async fn a_cost_map_installed_after_boot_decides_the_next_request() {
    litellm_gateway::install_crypto_provider();
    let config = Config::from_yaml(
        "model_list:
  - model_name: latest
    litellm_params:
      model: anthropic/claude-installed-latest
      api_key: provider-key
general_settings:
  master_key: gateway-key
",
    )
    .unwrap();
    let resources =
        litellm_gateway::Resources::over(&config, Arc::new(|_: &str| None::<String>)).unwrap();
    let state = litellm_gateway::AppState::over(&config, &Environment::default(), &resources)
        .await
        .unwrap();
    let admit = Arc::new(litellm_gateway::Admit::new(
        &config,
        &state,
        StoredConfig::Unused,
    ));
    let over = || {
        let admit = admit.clone();
        async move {
            let request = Request::post("/v1/messages")
                .header("authorization", "Bearer gateway-key")
                .body(Body::from(
                    r#"{"model": "latest", "max_tokens": 16, "messages": []}"#,
                ))
                .unwrap();
            admit
                .admit(request)
                .await
                .map(drop)
                .map_err(|(_, reason)| String::from(reason.over))
        }
    };

    let before = over().await;
    state.inference.install(
        CostMap::read(
            br#"{
                "claude-installed-9": {"litellm_provider": "anthropic", "mode": "chat"},
                "claude-installed-latest": {"alias_of": "claude-installed-9"}
            }"#,
            None,
            IntegrityLimits::UNCHECKED,
        )
        .unwrap(),
    );
    let after = over().await;
    state.close().await;

    assert_eq!(
        (before, after),
        (
            Ok(()),
            Err("model_list.litellm_params.model alias claude-installed-latest".to_owned())
        )
    );
}
