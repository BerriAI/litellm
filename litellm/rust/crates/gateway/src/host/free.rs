//! Which model groups are free, as the proxy's `is_model_cost_zero`
//! (`litellm/proxy/auth/auth_checks.py`) decides it, so a call to one is checked against no
//! budget. A name is read as the router serves it, through its `model_group_alias` once.
//!
//! A group is free only when its input and output rates, widened over its deployments as
//! `_set_model_group_info` widens them, are both set and neither is above zero, at least one of its
//! deployments declares a token rate of its own in its `model_info` or `litellm_params`
//! (`_is_cost_explicitly_configured`), and none bills reserved PTU capacity at a flat rate
//! (`_has_ptu_flat_cost`). A rate the cost map alone sets to zero is no declared rate, so such a
//! group is paid. An unknown name, or a group whose rates are missing, is paid.
//!
//! `_has_explicit_positive_cost` is covered by the widened rates, except for its per-query price,
//! which only a `parallel_ai` deployment bills and which no native route serves.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures_util::future;
use litellm_model_catalog::Catalog;
use litellm_router_core::{
    Deployment,
    registry::{DeploymentFilter, NameFilter, Registry},
};
use serde_json::{Map, Value};

use super::precall::merged_info;

const TOKEN_RATES: [&str; 2] = ["input_cost_per_token", "output_cost_per_token"];

/// The model groups a call to which is checked against no budget, and the aliases that name them.
#[derive(Debug, Default)]
pub struct FreeModels {
    free: HashSet<String>,
    aliases: HashMap<String, String>,
}

/// What one deployment tells of its group's price.
struct Priced {
    rates: [Option<f64>; 2],
    declares_token_rate: bool,
    flat_capacity: bool,
}

impl FreeModels {
    pub(crate) async fn read(catalog: &Catalog, registry: &Registry) -> Self {
        let deployments = registry.deployments(&DeploymentFilter::default());
        let priced = future::join_all(deployments.iter().map(async |deployment| {
            (
                deployment.model_name.clone(),
                priced(deployment, merged_info(catalog, deployment).await),
            )
        }))
        .await;
        let groups = priced.into_iter().fold(
            HashMap::<String, Vec<Priced>>::new(),
            |mut groups, (group, priced)| {
                groups.entry(group).or_default().push(priced);
                groups
            },
        );
        let aliases = registry
            .names(&NameFilter {
                include_aliases: true,
            })
            .into_iter()
            .filter_map(|name| {
                let target = registry.alias_of(&name)?.to_owned();
                Some((name, target))
            })
            .collect();
        Self {
            free: groups
                .into_iter()
                .filter(|(_, deployments)| free(deployments))
                .map(|(group, _)| group)
                .collect(),
            aliases,
        }
    }

    /// Whether a call that names `model` skips every budget check.
    pub fn is_free(&self, model: &str) -> bool {
        let group = self.aliases.get(model).map_or(model, String::as_str);
        self.free.contains(group)
    }
}

/// A rate as the deployment's cost map row under its id holds it: its `litellm_params` over its
/// `model_info` over the cost map's row for its model.
fn priced(deployment: &Arc<Deployment>, merged: Option<Map<String, Value>>) -> Priced {
    let declared = |rate: &str| {
        deployment
            .raw_params
            .get(rate)
            .or_else(|| deployment.model_info.raw.get(rate))
            .filter(|value| !value.is_null())
    };
    let rate = |rate: &str| {
        declared(rate)
            .or_else(|| merged.as_ref()?.get(rate))
            .and_then(Value::as_f64)
    };
    let set = |field: &str| {
        deployment
            .model_info
            .raw
            .get(field)
            .is_some_and(|value| !value.is_null())
    };
    Priced {
        rates: TOKEN_RATES.map(rate),
        declares_token_rate: TOKEN_RATES.iter().any(|rate| declared(rate).is_some()),
        flat_capacity: set("ptu_count") && set("cost_per_ptu_per_hour"),
    }
}

fn free(deployments: &[Priced]) -> bool {
    let unpriced = |index: usize| {
        let rates = deployments
            .iter()
            .filter_map(|priced| priced.rates[index])
            .collect::<Vec<_>>();
        rates.is_empty() || rates.iter().any(|rate| *rate > 0.0)
    };
    !unpriced(0)
        && !unpriced(1)
        && deployments.iter().any(|priced| priced.declares_token_rate)
        && !deployments.iter().any(|priced| priced.flat_capacity)
}

#[cfg(test)]
impl FreeModels {
    /// The free groups of the router `config` describes, over [`super::test_prices::catalog`].
    pub(crate) async fn of(config: &str) -> Self {
        let config = litellm_config::Config::from_yaml(config).unwrap();
        let router = crate::boot::unhosted(&config);
        Self::read(&super::test_prices::catalog(), &router.registry()).await
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const PRICED: &str = "model_list:
  - model_name: declared-free
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0
  - model_name: free-by-params
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
      input_cost_per_token: 0
      output_cost_per_token: 0
  - model_name: free-by-the-cost-map
    litellm_params:
      model: anthropic/free-model
      api_key: provider-key
  - model_name: free-input-only
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
  - model_name: one-rate-declared
    litellm_params:
      model: anthropic/free-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
  - model_name: unmapped
    litellm_params:
      model: anthropic/no-such-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
  - model_name: mixed
    litellm_params:
      model: anthropic/free-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0
  - model_name: mixed
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
  - model_name: reserved-capacity
    litellm_params:
      model: anthropic/free-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0
      ptu_count: 1
      cost_per_ptu_per_hour: 2.0
router_settings:
  model_group_alias:
    free-alias: declared-free
    paid-alias: mixed
";

    #[rstest]
    #[case::both_rates_declared_zero_in_model_info("declared-free", true)]
    #[case::both_rates_declared_zero_in_litellm_params("free-by-params", true)]
    #[case::zero_only_in_the_cost_map("free-by-the-cost-map", false)]
    #[case::zero_input_with_a_paid_output("free-input-only", false)]
    #[case::one_rate_declared_and_the_other_zero_in_the_cost_map("one-rate-declared", true)]
    #[case::no_output_rate_anywhere("unmapped", false)]
    #[case::a_paid_deployment_in_the_group("mixed", false)]
    #[case::reserved_capacity_billed_flat("reserved-capacity", false)]
    #[case::an_alias_of_a_free_group("free-alias", true)]
    #[case::an_alias_of_a_paid_group("paid-alias", false)]
    #[case::an_unknown_model("other", false)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_is_free_only_when_its_group_declares_both_rates_at_zero(
        #[case] model: &str,
        #[case] free: bool,
    ) {
        assert_eq!(FreeModels::of(PRICED).await.is_free(model), free);
    }
}
