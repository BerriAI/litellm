//! The facts `enable_pre_call_checks` decides by (`litellm/router_rust/steps/precall.py`): each
//! candidate's context window as `get_router_model_info` merges it, its own
//! `model_info.max_input_tokens` over the cost map's, and the tokens of the prompt the route
//! hands over, as [`prompt_estimate`](super::prompt_estimate) estimates them. router-core decides
//! with them, and a route that hands over no prompt is held to no window.
//!
//! A deployment's window depends on the deployment and the cost map alone, so [`Windows`] reads
//! each one once at boot. A call and `admit` read that one table, and so hold to the same window.
//!
//! Region and params are decided in router-core (`precall/region.rs`, `precall/params.rs`), and
//! this host reports neither, as [`ALLOWED_MODEL_REGION`] and [`REQUESTED_PARAMS`] say why.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures_util::future;
use litellm_gateway_inference::routing::Prompt;
use litellm_model_catalog::Catalog;
use litellm_router_core::{
    Deployment,
    model_info::{RouterModelInfoQuery, router_model_info},
    precall::{DeploymentFacts, PreCallFacts},
};
use serde_json::Value;
use tokio::sync::OnceCell;

use super::{model_info::NativeModelInfoHost, prompt_estimate::Countable};

/// The context window of each deployment the router was built with.
#[derive(Default)]
pub struct Windows {
    by_deployment: HashMap<String, i64>,
    /// The model group of each deployment no window was read for.
    windowless: HashSet<String>,
}

impl Windows {
    pub(crate) async fn read(catalog: &Catalog, deployments: &[Arc<Deployment>]) -> Self {
        let read = future::join_all(
            deployments
                .iter()
                .map(async |deployment| (deployment, window(catalog, deployment).await)),
        )
        .await;
        Self {
            by_deployment: read
                .iter()
                .filter_map(|(deployment, window)| Some((deployment.id.clone(), (*window)?)))
                .collect(),
            windowless: read
                .iter()
                .filter(|(_, window)| window.is_none())
                .map(|(deployment, _)| deployment.model_name.clone())
                .collect(),
        }
    }

    fn of(&self, deployment: &Deployment) -> Option<i64> {
        self.by_deployment.get(&deployment.id).copied()
    }

    /// Whether a window was read for every deployment of `group`.
    pub(crate) fn cover(&self, group: &str) -> bool {
        !self.windowless.contains(group)
    }
}

/// `_deployment_max_input_tokens`: the merged context window, or `None` when there is none or the
/// deployment cannot be resolved.
async fn window(catalog: &Catalog, deployment: &Deployment) -> Option<i64> {
    let base_model = |named: Option<&Value>| named.cloned().unwrap_or(Value::Null);
    let query = RouterModelInfoQuery {
        deployment_id: Value::String(deployment.id.clone()),
        model_info_base_model: base_model(deployment.model_info.raw.get("base_model")),
        params_base_model: base_model(deployment.raw_params.get("base_model")),
        params_model: deployment.params.model.clone(),
        extra_base_model: base_model(deployment.params.extra.get("base_model")),
        received_model_name: deployment.model_name.clone(),
    };
    let host = NativeModelInfoHost {
        catalog,
        deployment,
    };
    match router_model_info(&host, &query).await {
        Ok(merged) => merged?.info.get("max_input_tokens")?.as_i64(),
        Err(error) => {
            tracing::debug!(%error, id = %deployment.id, "skipping the deployment's context window");
            None
        }
    }
}

/// The estimate runs off the event loop, since it tokenizes the whole prompt.
async fn tokens(prompt: &Prompt<'_>) -> Option<i64> {
    let countable = Countable::of(prompt)?;
    tokio::task::spawn_blocking(move || countable.tokens())
        .await
        .ok()
        .flatten()
}

/// No region. Only the request's end user sets `allowed_model_region`, and the gateway reads no
/// end user yet, so the region stage is not wired here. `admit` sends a key that carries a region
/// to Python.
const ALLOWED_MODEL_REGION: Option<Value> = None;

/// No requested param. `response_format` is the one param the check filters on, and `admit`
/// sends a body that names it to Python, so no native call sends one.
const REQUESTED_PARAMS: Vec<String> = Vec::new();

/// The facts for one call. Its prompt is the same on every retry and fallback, so it is counted
/// once into `counted`, and only when a candidate has a window to hold it to.
pub(super) async fn facts(
    windows: &Windows,
    prompt: Option<Prompt<'_>>,
    counted: &OnceCell<Option<i64>>,
    candidates: &[Arc<Deployment>],
) -> PreCallFacts {
    let Some(prompt) = prompt else {
        return PreCallFacts::default();
    };
    let has_window = candidates
        .iter()
        .any(|deployment| windows.of(deployment).is_some());
    PreCallFacts {
        has_input: true,
        countable: true,
        input_tokens: if has_window {
            *counted.get_or_init(|| tokens(&prompt)).await
        } else {
            None
        },
        allowed_model_region: ALLOWED_MODEL_REGION,
        requested_params: REQUESTED_PARAMS,
        deployments: candidates
            .iter()
            .map(|deployment| {
                let facts = DeploymentFacts {
                    max_input_tokens: windows.of(deployment),
                    ..DeploymentFacts::default()
                };
                (deployment.id.clone(), facts)
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use litellm_router_core::DeploymentSpec;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const COST_MAP: &[u8] = br#"{"listed-model": {"litellm_provider": "anthropic", "mode": "chat", "max_input_tokens": 5000}}"#;

    #[rstest]
    #[case::its_own_over_the_cost_maps("anthropic/listed-model", json!({"max_input_tokens": 100}), Some(100))]
    #[case::the_cost_maps("anthropic/listed-model", json!({}), Some(5000))]
    #[case::its_own_for_a_model_the_cost_map_does_not_list(
        "anthropic/in-house-9",
        json!({"max_input_tokens": 100}),
        Some(100)
    )]
    #[case::none_for_a_model_the_cost_map_does_not_list("anthropic/in-house-9", json!({}), None)]
    #[tokio::test]
    async fn a_deployments_window_is(
        #[case] model: &str,
        #[case] model_info: Value,
        #[case] expected: Option<i64>,
    ) {
        let catalog = Catalog::parse(COST_MAP).unwrap().value;
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "claude",
            "litellm_params": {"model": model},
            "model_info": model_info,
        }))
        .unwrap();
        let deployment = Deployment::from_spec(spec).unwrap();

        assert_eq!(window(&catalog, &deployment).await, expected);
    }
}
