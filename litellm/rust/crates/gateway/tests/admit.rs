//! Each file in `tests/admit/` is one config and the requests sent to the gateway built from it,
//! each with who answers it, `rust` or `python`. A request for Python names the `reason` the client
//! sees in `x-litellm-gateway`, and may name the setting or trait it goes to Python `over`.

mod keys;

use std::{collections::BTreeMap, path::Path, sync::Arc};

use axum::{body::Body, extract::Request, http::Method};
use litellm_config::{Config, Environment};
use litellm_error::Source;
use litellm_gateway::{
    EndUserLookup,
    host::auth::{KeyRows, authenticator},
};
use litellm_repositories::{EndUserRecord, KeyView};
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
    /// The end users the database holds a row for, each with the columns it fills.
    end_users: Option<BTreeMap<String, Row>>,
    requests: Vec<Sent>,
}

/// What one `LiteLLM_EndUserTable` row holds beyond the end user's id.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    #[serde(default)]
    spend: f64,
    #[serde(default)]
    blocked: bool,
    budget_id: Option<String>,
}

/// Each end user with the row the scenario gives it, and no row for any other.
struct Rows(BTreeMap<String, Row>);

impl EndUserLookup for Rows {
    fn find<'a>(
        &'a self,
        end_user_id: &'a str,
    ) -> futures_util::future::BoxFuture<'a, Result<Option<EndUserRecord>, Source>> {
        Box::pin(async move {
            Ok(self.0.get(end_user_id).map(|row| EndUserRecord {
                user_id: end_user_id.to_owned(),
                alias: None,
                spend: row.spend,
                allowed_model_region: None,
                default_model: None,
                models: vec![],
                budget_id: row.budget_id.clone(),
                litellm_budget_table: None,
                object_permission_id: None,
                object_permission: None,
                blocked: row.blocked,
            }))
        })
    }
}

/// The database holding each key of the scenario, a plain row with the model list and the token
/// limit it gives.
fn listed(
    keys: &BTreeMap<String, Vec<String>>,
    tpm_limits: &BTreeMap<String, i64>,
) -> Arc<dyn KeyRows> {
    Arc::new(
        keys.iter()
            .map(|(token, models)| {
                let row = KeyView {
                    models: models.clone(),
                    tpm_limit: tpm_limits.get(token).copied(),
                    ..keys::plain(token)
                };
                (row.token.clone(), row)
            })
            .collect::<BTreeMap<String, KeyView>>(),
    )
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
#[case::unread_fallbacks("unread_fallbacks")]
#[case::drop_params_set("drop_params_set")]
#[case::database("database")]
#[case::end_user_default_budget("end_user_default_budget")]
#[case::store_model_in_db("store_model_in_db")]
#[case::router_settings_by_group("router_settings_by_group")]
#[case::groups_and_fallbacks("groups_and_fallbacks")]
#[case::model_aliases("model_aliases")]
#[case::router_setting_for_every_request("router_setting_for_every_request")]
#[case::callers("callers")]
#[case::virtual_key_models("virtual_key_models")]
#[case::virtual_key_token_limits("virtual_key_token_limits")]
#[case::default_key_limits("default_key_limits")]
#[case::rate_limit_policy("rate_limit_policy")]
#[tokio::test]
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
    let state = match scenario.keys.is_empty() {
        true => state,
        false => litellm_gateway::AppState {
            keys: Some(authenticator(
                &config,
                listed(&scenario.keys, &scenario.tpm_limits),
                None,
            )),
            ..state
        },
    };
    let state = match scenario.end_users {
        Some(end_users) => litellm_gateway::AppState {
            end_users: Some(Arc::new(Rows(end_users))),
            ..state
        },
        None => state,
    };
    let admit = Arc::new(litellm_gateway::Admit::new(
        &config,
        &Environment::default(),
        &state,
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
