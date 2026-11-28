//! Prices a Messages reservation against the live cost map before routing.

use std::sync::Arc;

use litellm_cost::{
    bound,
    bound::{
        Bound, BoundModel, BoundQuery, CostShape, DEFAULT_MAX_OUTPUT_TOKENS, FlatSource, GroupRates,
    },
    candidates::pricing_rows,
    money::UnitPrice,
};
use litellm_gateway_inference::routing::Prompt;
use litellm_model_catalog::{CostLookup, CostMap, LoweredRow, PricingContext};
use litellm_python_compat::number::float_of;
use litellm_router_core::{Router, registry::DeploymentFilter};
use serde_json::{Map, Value};

use crate::{host::precall::merged_info, prompt_estimate::Countable};

pub(super) async fn estimate(
    router: &Router,
    cost_map: &CostMap,
    body: &Map<String, Value>,
    team_id: Option<&str>,
) -> Option<(Bound, Option<i64>)> {
    let model = body.get("model")?.as_str()?;
    let registry = router.registry();
    let group = registry.alias_of(model).unwrap_or(model);
    let catalog = cost_map.catalog();
    let context = PricingContext::default();
    let lookup = CostLookup::new(catalog, &context);
    let mut source = pricing_rows(&lookup, [], []);
    let mut rates = GroupRates::default();
    let mut priced = Vec::new();
    let mut tiers = Vec::new();
    let mut deployments = registry
        .deployments(&DeploymentFilter {
            model_name: Some(group),
            ..DeploymentFilter::default()
        })
        .into_iter()
        .filter(|deployment| {
            deployment
                .model_info
                .team_id
                .as_deref()
                .is_none_or(|id| Some(id) == team_id)
        })
        .collect::<Vec<_>>();
    if deployments
        .iter()
        .any(|deployment| deployment.model_info.team_id.is_some())
    {
        deployments.retain(|deployment| deployment.model_info.team_id.is_some());
    }
    for deployment in deployments {
        let Some(fields) = merged_info(catalog, &deployment).await else {
            continue;
        };
        let rate = |field| {
            fields
                .get(field)
                .and_then(|value| float_of(value).ok())
                .and_then(UnitPrice::new)
        };
        let limit = |field| fields.get(field).and_then(Value::as_i64);
        rates.input = rates
            .input
            .into_iter()
            .chain(rate("input_cost_per_token"))
            .reduce(|a, b| if a.get() >= b.get() { a } else { b });
        rates.output = rates
            .output
            .into_iter()
            .chain(rate("output_cost_per_token"))
            .reduce(|a, b| if a.get() >= b.get() { a } else { b });
        rates.max_input_tokens = rates.max_input_tokens.max(limit("max_input_tokens"));
        rates.max_output_tokens = rates.max_output_tokens.max(limit("max_output_tokens"));
        let has_tiers = fields
            .get("tiered_pricing")
            .and_then(Value::as_array)
            .is_some_and(|tiers| !tiers.is_empty());
        let row = LoweredRow::lower(Arc::new(fields), deployment.provider());
        let at;
        (source, at) = source.with_per_call(&deployment.id, row);
        if has_tiers {
            tiers.push(at.clone());
        }
        priced.push(at);
    }
    let candidates = std::iter::once(None)
        .chain(tiers.into_iter().map(Some))
        .map(|tiers| CostShape {
            flat: FlatSource::Group(rates),
            tiers,
            deployments: priced.clone(),
        })
        .collect();
    let counted_input = match Countable::of(&Prompt::Messages(body)) {
        Some(prompt) => tokio::task::spawn_blocking(move || prompt.tokens())
            .await
            .ok()
            .flatten(),
        None => None,
    };
    let requested_output = body
        .get("max_tokens")
        .and_then(Value::as_i64)
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    Some((
        bound(
            &source,
            &BoundQuery {
                models: vec![BoundModel {
                    counted_input,
                    requested_output,
                    output_multiplier: 1,
                    candidates,
                }],
                request: Default::default(),
                policy: Default::default(),
            },
        ),
        counted_input,
    ))
}
