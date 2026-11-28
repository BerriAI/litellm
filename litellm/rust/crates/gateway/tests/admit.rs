//! Each file in `tests/admit/` is one config and the requests sent to the gateway built from it,
//! each with who answers it, `rust` or `python`. A request for Python names the `reason` the client
//! sees in `x-litellm-gateway`, and may name the setting or trait it goes to Python `over`.
//! A request that names `jwt` claims carries them signed by a provider whose JWKS
//! `JWT_PUBLIC_KEY_URL` names, and `jwks` says what that URL answers. `mappings` lists the keys the
//! database maps a JWT's claim to. `users_after` names what the database holds for a user once every
//! request was answered, so a file shows that a request Python answers wrote nothing.

mod jwt;
mod keys;

use std::{collections::BTreeMap, net::SocketAddr, path::Path, sync::Arc};

use axum::{
    body::Body,
    extract::{ConnectInfo, Request},
    http::Method,
};
use litellm_config::{Config, Environment};
use litellm_gateway::{StoredConfig, host::auth::AuthStore};
use litellm_license::Entitlements;
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
    /// The users the database holds, each id with the columns set on its row.
    #[serde(default)]
    users: BTreeMap<String, Columns>,
    /// What the database holds for each user once every request was answered: the columns named,
    /// or nothing for a user it does not hold.
    #[serde(default)]
    users_after: BTreeMap<String, Option<Columns>>,
    /// The other rows the database holds, each under its table's name, stored in this order.
    #[serde(default)]
    rows: Vec<BTreeMap<String, serde_json::Value>>,
    allow_missing_organization: Option<bool>,
    /// The teams the database holds, each id with the columns set on its row.
    #[serde(default)]
    teams: BTreeMap<String, Columns>,
    /// The projects the database holds, each id with the columns set on its row.
    #[serde(default)]
    projects: BTreeMap<String, Columns>,
    /// The key each JWT claim maps to for every issuer: the claim's name, then each value it
    /// takes with the token of its key.
    #[serde(default)]
    mappings: BTreeMap<String, BTreeMap<String, String>>,
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
    /// Whether the deployment's license is an enterprise one, which JWT authentication needs.
    #[serde(default)]
    enterprise: bool,
    /// What the identity provider answers at its JWKS URL.
    #[serde(default)]
    jwks: Jwks,
    requests: Vec<Sent>,
}

/// `served` when the provider publishes its signing keys, `not_found` when it answers 404 where
/// they should be, and `down` when nothing listens there.
#[derive(Clone, Copy, Default, Deserialize)]
enum Jwks {
    #[default]
    #[serde(rename = "served")]
    Served,
    #[serde(rename = "not_found")]
    NotFound,
    #[serde(rename = "down")]
    Down,
}

async fn jwks_url(jwks: Jwks) -> String {
    match jwks {
        Jwks::Served => jwt::serve_jwks().await.url,
        Jwks::NotFound => jwt::serve_no_jwks().await.url,
        Jwks::Down => jwt::unreachable_jwks().await,
    }
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

/// The columns a scenario sets on a row, each by its name.
type Columns = serde_json::Map<String, serde_json::Value>;

/// The database holding each team, user, project and key of the scenario, in that order so a key
/// can point at a project and a project at a team, a key as a plain row with its columns set on
/// it, each claim mapped to its key, and each of its other rows.
async fn listed(scenario: &Scenario) -> keys::Held {
    let held = keys::Held::start().await;
    if scenario.allow_missing_organization == Some(true) {
        held.execute(
            r#"ALTER TABLE "LiteLLM_VerificationToken" DROP CONSTRAINT "LiteLLM_VerificationToken_organization_id_fkey""#,
        ).await;
    }
    for (team_id, columns) in &scenario.teams {
        let row = std::iter::once(("team_id".to_owned(), serde_json::json!(team_id)))
            .chain(columns.clone())
            .collect();
        held.insert_named("LiteLLM_TeamTable", &serde_json::Value::Object(row))
            .await;
    }
    for (user_id, columns) in &scenario.users {
        let row = std::iter::once(("user_id".to_owned(), serde_json::json!(user_id)))
            .chain(columns.clone())
            .collect();
        held.insert_named("LiteLLM_UserTable", &serde_json::Value::Object(row))
            .await;
    }
    for (project_id, columns) in &scenario.projects {
        let row = [
            ("project_id", project_id.as_str()),
            ("created_by", "test"),
            ("updated_by", "test"),
        ]
        .into_iter()
        .map(|(column, value)| (column.to_owned(), serde_json::json!(value)))
        .chain(columns.clone())
        .collect();
        held.insert_named("LiteLLM_ProjectTable", &serde_json::Value::Object(row))
            .await;
    }
    for (table, row) in scenario.rows.iter().flatten() {
        held.insert_named(table, row).await;
    }
    for (token, columns) in &scenario.keys {
        held.key(token, serde_json::Value::Object(columns.clone()))
            .await;
    }
    for (claim, mapped) in &scenario.mappings {
        for (value, token) in mapped {
            held.mapping("", claim, value, token).await;
        }
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
    unavailable: Option<AuthTable>,
    /// The client's TCP peer, which the IP allowlist reads.
    peer: Option<SocketAddr>,
    /// Claims sent as a signed JWT bearer token.
    jwt: Option<Jwt>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Jwt {
    claims: serde_json::Map<String, serde_json::Value>,
    /// Seconds from now until `exp`, negative for a token that has expired.
    #[serde(default = "an_hour")]
    expires_in: i64,
}

fn an_hour() -> i64 {
    3600
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AuthTable {
    Organization,
    Membership,
}

impl AuthTable {
    fn unavailable(self) -> &'static str {
        match self {
            Self::Organization => {
                r#"ALTER TABLE "LiteLLM_OrganizationTable" RENAME COLUMN "model_spend" TO "unavailable_model_spend""#
            }
            Self::Membership => {
                r#"ALTER TABLE "LiteLLM_TeamMembership" RENAME COLUMN "total_spend" TO "unavailable_total_spend""#
            }
        }
    }
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
        let bearer = self.jwt.as_ref().map(|jwt| {
            (
                "authorization".to_owned(),
                format!("Bearer {}", jwt::issue(&jwt.claims, jwt.expires_in)),
            )
        });
        self.headers
            .clone()
            .into_iter()
            .chain(bearer)
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
#[case::virtual_key_team("virtual_key_team")]
#[case::virtual_key_owner("virtual_key_owner")]
#[case::virtual_key_project("virtual_key_project")]
#[case::virtual_key_org("virtual_key_org")]
#[case::virtual_key_membership("virtual_key_membership")]
#[case::jwt("jwt")]
#[case::jwt_team_claim("jwt_team_claim")]
#[case::jwt_without_enterprise("jwt_without_enterprise")]
#[case::jwt_beside_keys("jwt_beside_keys")]
#[case::jwt_customer_claims("jwt_customer_claims")]
#[case::jwt_scope_based_access("jwt_scope_based_access")]
#[case::jwt_provider_unusable("jwt_provider_unusable")]
#[case::jwt_provider_down("jwt_provider_down")]
#[case::jwt_key_mapping("jwt_key_mapping")]
#[case::jwt_key_mapping_without_database("jwt_key_mapping_without_database")]
#[case::jwt_team_header("jwt_team_header")]
#[case::jwt_team_by_model("jwt_team_by_model")]
#[case::jwt_team_alias("jwt_team_alias")]
#[case::jwt_team_id_upsert("jwt_team_id_upsert")]
#[case::jwt_user_claim("jwt_user_claim")]
#[case::jwt_user_id_upsert("jwt_user_id_upsert")]
#[case::jwt_user_id_upsert_with_defaults("jwt_user_id_upsert_with_defaults")]
#[tokio::test(flavor = "multi_thread")]
async fn admit_answers_each_request_as_its_scenario_says(#[case] name: &str) {
    litellm_gateway::install_crypto_provider();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/admit/{name}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let scenario: Scenario =
        serde_yaml_ng::from_value(litellm_yaml::from_str(&text).unwrap()).unwrap();
    let nothing_held = scenario.keys.is_empty()
        && scenario.teams.is_empty()
        && scenario.users.is_empty()
        && scenario.projects.is_empty()
        && scenario.rows.is_empty()
        && scenario.allow_missing_organization != Some(true);
    let held = match nothing_held {
        true => None,
        false => Some(listed(&scenario).await),
    };
    let config = match (&held, scenario.database) {
        (Some(held), true) => with_database(scenario.config.clone(), held.url()),
        _ => scenario.config.clone(),
    };
    let config = Config::from_value(config).unwrap();
    let jwks_url = jwks_url(scenario.jwks).await;
    let process = move |name: &str| (name == "JWT_PUBLIC_KEY_URL").then(|| jwks_url.clone());
    let resources = litellm_gateway::Resources::over(&config, Arc::new(process)).unwrap();
    let state = litellm_gateway::AppState::over(&config, &Environment::default(), &resources)
        .await
        .unwrap();
    let state = match scenario.enterprise {
        false => state,
        true => litellm_gateway::AppState {
            entitlements: Arc::new(Entitlements::ServerVerified),
            ..state
        },
    };
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

    let mut answered = Vec::with_capacity(scenario.requests.len());
    for sent in &scenario.requests {
        if let Some(table) = sent.unavailable {
            held.as_ref().unwrap().execute(table.unavailable()).await;
        }
        let (who, reason, over) = match admit.clone().admit(sent.request()).await {
            Ok(_) => (Who::Rust, None, None),
            Err((_, reason)) => (
                Who::Python,
                Some(<&str>::from(reason.why).to_owned()),
                Some(String::from(reason.over)),
            ),
        };
        answered.push((
            sent.name.as_str(),
            who,
            reason,
            over.filter(|_| sent.over.is_some()),
        ));
    }
    state.close().await;
    let users = match &held {
        Some(held) => held.users().await,
        None => Vec::new(),
    };
    let users_after: BTreeMap<_, _> = scenario
        .users_after
        .iter()
        .map(|(user_id, columns)| {
            let row = users.iter().find(|row| row["user_id"] == user_id.as_str());
            let named = columns.iter().flat_map(Columns::keys);
            let held = row.map(|row| {
                named
                    .map(|column| (column.clone(), row[column].clone()))
                    .collect::<Columns>()
            });
            (user_id.clone(), held)
        })
        .collect();

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
    assert_eq!((answered, users_after), (expected, scenario.users_after));
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
