//! What a budget hold costs on `/v1/messages`, as the proxy's `_bound` prices it: the model
//! group's input and output rates and token windows, widened over its deployments as
//! `_set_model_group_info` widens them, held for the input tokens the Anthropic tokenizer counts
//! and the `max_tokens` the body asks for. `litellm_cost::bound` prices it.
//!
//! The gateway holds natively only where it reads what Python reads: a group named by its own
//! name, served by Anthropic deployments whose model the cost map lists under Anthropic, with no
//! tiered pricing, base model, dropped params or output count, and a body that sets no output cap
//! but `max_tokens` and counts without error. [`HoldPrices::holds`] tells `admit` whether a call
//! is one, so any other call with a budget goes to Python.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use futures_util::future;
use litellm_cost::{
    adjust::{Discounts, Margins},
    bound::{BoundModel, BoundPolicy, BoundQuery, CostShape, FlatSource, GroupRates, RequestCaps},
    money::UnitPrice,
    query::Candidate,
    row::{RowRef, RowSource},
};
use litellm_model_catalog::{Catalog, LoweredRow};
use litellm_router_core::Deployment;
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::{Map, Value};

use super::precall::merged_info;

const ANTHROPIC: &str = "anthropic";
/// The body fields the proxy reads an output cap or an output count from, beside `max_tokens`.
const UNREAD_BODY_FIELDS: [&str; 12] = [
    "max_completion_tokens",
    "max_output_tokens",
    "maxTokens",
    "inferenceConfig",
    "generationConfig",
    "generation_config",
    "contents",
    "cachedContent",
    "cached_content",
    "n",
    "best_of",
    "additional_drop_params",
];
/// The deployment params that change which output cap reaches the provider or how many outputs
/// it returns.
const UNREAD_PARAMS: [&str; 4] = ["additional_drop_params", "n", "best_of", "base_model"];

static ANTHROPIC_TOKENIZER: LazyLock<Option<TokenCounter>> = LazyLock::new(|| {
    TokenCounter::from_json(include_str!(
        "../../../../../litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    ))
    .inspect_err(|error| tracing::warn!(%error, "the Anthropic tokenizer did not load"))
    .ok()
});

/// What a call holds against each of its budgets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub reservation_cost: f64,
    pub input_cost: f64,
    pub input_tokens: i64,
}

/// The group rates of each model group the gateway holds for natively.
#[derive(Default)]
pub struct HoldPrices {
    groups: HashMap<String, GroupRates>,
}

impl HoldPrices {
    pub(crate) async fn read(catalog: &Catalog, deployments: &[Arc<Deployment>]) -> Self {
        let read = future::join_all(deployments.iter().map(async |deployment| {
            let info = merged_info(catalog, deployment).await;
            (deployment, info.filter(|info| held(deployment, info)))
        }))
        .await;
        let by_group = read.into_iter().fold(
            HashMap::<&str, Option<GroupRates>>::new(),
            |mut groups, (deployment, info)| {
                let widened = match (groups.remove(deployment.model_name.as_str()), info) {
                    (None, Some(info)) => Some(rates(&info)),
                    (Some(Some(group)), Some(info)) => Some(widen(group, &info)),
                    (_, None) | (Some(None), _) => None,
                };
                groups.insert(&deployment.model_name, widened);
                groups
            },
        );
        Self {
            groups: by_group
                .into_iter()
                .filter(|(group, _)| reads_anthropic_tokens(catalog, group))
                .filter_map(|(group, rates)| Some((group.to_owned(), rates?)))
                .collect(),
        }
    }

    /// Whether a call for `model` with `body` is held natively.
    pub(crate) fn holds(&self, model: &str, body: &Map<String, Value>) -> bool {
        self.hold(model, body).is_ok()
    }

    /// `_bound` for a `/v1/messages` call. `Ok(None)` is a call the proxy holds nothing for, as
    /// it holds nothing when no rate prices the call or the hold is not positive.
    pub(crate) fn hold(
        &self,
        model: &str,
        body: &Map<String, Value>,
    ) -> Result<Option<Hold>, Unheld> {
        let rates = self.groups.get(model).ok_or(Unheld)?;
        let requested_output = requested_output(body).ok_or(Unheld)?;
        let input_tokens = counted(body).ok_or(Unheld)?;
        let query = BoundQuery {
            models: vec![BoundModel {
                counted_input: Some(input_tokens),
                requested_output,
                output_multiplier: 1,
                candidates: vec![CostShape {
                    flat: FlatSource::Group(*rates),
                    tiers: None,
                    deployments: Vec::new(),
                }],
            }],
            request: RequestCaps::default(),
            policy: BoundPolicy::Legacy,
        };
        let bound = litellm_cost::bound(&GroupOnly, &query);
        Ok(bound
            .max
            .filter(|cost| *cost > 0.0)
            .map(|reservation_cost| Hold {
                reservation_cost,
                input_cost: bound.input_floor.unwrap_or(0.0),
                input_tokens,
            }))
    }
}

#[cfg(test)]
impl HoldPrices {
    /// The prices of the router `config` describes, over [`test_prices::catalog`].
    pub(crate) async fn of(config: &str) -> Self {
        use litellm_router_core::registry::DeploymentFilter;

        let config = litellm_config::Config::from_yaml(config).unwrap();
        let router = crate::boot::unhosted(&config);
        let deployments = router.registry().deployments(&DeploymentFilter::default());
        Self::read(&test_prices::catalog(), &deployments).await
    }
}

#[cfg(test)]
pub(crate) mod test_prices {
    use litellm_model_catalog::Catalog;
    use serde_json::{Value, json};

    pub(crate) const INPUT: f64 = 1e-07;
    pub(crate) const OUTPUT: f64 = 5e-07;
    pub(crate) const MAX_OUTPUT: i64 = 64_000;
    pub(crate) const WIDER_INPUT: f64 = 2e-07;
    pub(crate) const WIDER_MAX_OUTPUT: i64 = 128_000;

    fn row(provider: &str, input: f64, output: f64, max_output: i64) -> Value {
        json!({
            "litellm_provider": provider,
            "mode": "chat",
            "input_cost_per_token": input,
            "output_cost_per_token": output,
            "max_input_tokens": 200_000,
            "max_output_tokens": max_output,
        })
    }

    /// `held-model` at [`INPUT`] and [`OUTPUT`], and `wider-model` with a higher input rate and
    /// window but a lower output rate.
    pub(crate) fn catalog() -> Catalog {
        let rows = json!({
            "held-model": row("anthropic", INPUT, OUTPUT, MAX_OUTPUT),
            "wider-model": row("anthropic", WIDER_INPUT, OUTPUT / 2.0, WIDER_MAX_OUTPUT),
            "openai-model": row("openai", INPUT, OUTPUT, MAX_OUTPUT),
        });
        Catalog::parse(&serde_json::to_vec(&rows).unwrap())
            .unwrap()
            .value
    }
}

/// A call the gateway does not hold natively, which Python holds instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unheld;

/// A deployment whose output cap, output count and pricing Python reads as the gateway does.
fn held(deployment: &Deployment, info: &Map<String, Value>) -> bool {
    deployment.provider() == ANTHROPIC
        && !UNREAD_PARAMS
            .iter()
            .any(|param| deployment.raw_params.contains_key(*param))
        && info
            .get("tiered_pricing")
            .is_none_or(|tiers| tiers.as_array().is_none_or(Vec::is_empty))
}

/// `uses_anthropic_tokenizer`: the cost map lists `group` under Anthropic, and not as a Claude 3.
fn reads_anthropic_tokens(catalog: &Catalog, group: &str) -> bool {
    !group.contains("claude-3")
        && catalog
            .get(group)
            .and_then(|fields| fields.get("litellm_provider"))
            .and_then(Value::as_str)
            == Some(ANTHROPIC)
}

fn price(info: &Map<String, Value>, field: &str) -> Option<UnitPrice> {
    UnitPrice::new(info.get(field)?.as_f64()?)
}

fn tokens(info: &Map<String, Value>, field: &str) -> Option<i64> {
    info.get(field)?.as_i64()
}

fn rates(info: &Map<String, Value>) -> GroupRates {
    GroupRates {
        input: price(info, "input_cost_per_token"),
        output: price(info, "output_cost_per_token"),
        max_input_tokens: tokens(info, "max_input_tokens"),
        max_output_tokens: tokens(info, "max_output_tokens"),
    }
}

/// `_set_model_group_info`'s widening: each window and rate is the largest a deployment has.
fn widen(group: GroupRates, info: &Map<String, Value>) -> GroupRates {
    let next = rates(info);
    let wider = |kept: Option<UnitPrice>, next: Option<UnitPrice>| match (kept, next) {
        (Some(kept), Some(next)) if next.get() > kept.get() => Some(next),
        (None, next) => next,
        (kept, _) => kept,
    };
    GroupRates {
        input: wider(group.input, next.input),
        output: wider(group.output, next.output),
        max_input_tokens: group.max_input_tokens.max(next.max_input_tokens),
        max_output_tokens: group.max_output_tokens.max(next.max_output_tokens),
    }
}

/// `_requested_output_tokens` on the Messages surface, where every deployment honors
/// `max_tokens`: the cap the body sets, else the default. `None` for a body that sets an output
/// cap or an output count the gateway does not read.
fn requested_output(body: &Map<String, Value>) -> Option<i64> {
    if UNREAD_BODY_FIELDS
        .iter()
        .any(|field| body.contains_key(*field))
    {
        return None;
    }
    match body.get("max_tokens") {
        None | Some(Value::Null) => Some(litellm_cost::bound::DEFAULT_MAX_OUTPUT_TOKENS),
        Some(Value::Number(cap)) => cap.as_i64().filter(|cap| *cap >= 0),
        Some(_) => None,
    }
}

/// The input tokens the Anthropic tokenizer counts in the body, as Python's native count does.
fn counted(body: &Map<String, Value>) -> Option<i64> {
    let counter = ANTHROPIC_TOKENIZER.as_ref()?;
    let raw = serde_json::to_vec(body).ok()?;
    let request = CountableRequest::parse(&raw).ok()?;
    let count = counter.count_request(&request).ok()?;
    i64::try_from(count.input_tokens).ok()
}

/// A row source with no rows, since a group shape reads only its own rates.
struct GroupOnly;

static NO_DISCOUNTS: LazyLock<Discounts> = LazyLock::new(Discounts::default);
static NO_MARGINS: LazyLock<Margins> = LazyLock::new(Margins::default);

impl RowSource for GroupOnly {
    type Row = LoweredRow;

    fn resolve(&self, _: &Candidate) -> Option<RowRef> {
        None
    }

    fn row(&self, _: &RowRef) -> Option<&LoweredRow> {
        None
    }

    fn discounts(&self) -> &Discounts {
        &NO_DISCOUNTS
    }

    fn margins(&self) -> &Margins {
        &NO_MARGINS
    }
}

#[cfg(test)]
mod tests {
    use litellm_cost::bound::DEFAULT_MAX_OUTPUT_TOKENS;
    use rstest::rstest;
    use serde_json::json;

    use super::{test_prices::*, *};

    const CONFIG: &str = "model_list:
  - model_name: held-model
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
  - model_name: wider-model
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
  - model_name: wider-model
    litellm_params:
      model: anthropic/wider-model
      api_key: provider-key
  - model_name: renamed
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
  - model_name: openai-model
    litellm_params:
      model: openai/openai-model
      api_key: provider-key
  - model_name: dropping-model
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
      additional_drop_params: [max_tokens]
";

    async fn prices() -> HoldPrices {
        HoldPrices::of(CONFIG).await
    }

    fn body(fields: Value) -> Map<String, Value> {
        let Value::Object(fields) = fields else {
            unreachable!()
        };
        fields
            .into_iter()
            .chain([(
                "messages".to_owned(),
                json!([{"role": "user", "content": "Say hi"}]),
            )])
            .collect()
    }

    /// The Anthropic tokenizer counts nine input tokens in every body [`body`] builds.
    const COUNTED: i64 = 9;

    #[rstest]
    #[case::its_cap(json!({"model": "held-model", "max_tokens": 16}), INPUT, 16.0 * OUTPUT)]
    #[case::the_window_over_a_larger_cap(
        json!({"model": "held-model", "max_tokens": 100_000}),
        INPUT,
        MAX_OUTPUT as f64 * OUTPUT
    )]
    #[case::the_default_with_no_cap(
        json!({"model": "held-model"}),
        INPUT,
        DEFAULT_MAX_OUTPUT_TOKENS as f64 * OUTPUT
    )]
    #[case::each_rate_and_window_widest_over_the_groups_deployments(
        json!({"model": "wider-model", "max_tokens": 100_000}),
        WIDER_INPUT,
        100_000.0 * OUTPUT
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_messages_call_holds_its_counted_input_and_capped_output(
        #[case] sent: Value,
        #[case] input_rate: f64,
        #[case] output_cost: f64,
    ) {
        let model = sent["model"].as_str().unwrap().to_owned();
        let input_cost = COUNTED as f64 * input_rate;

        assert_eq!(
            prices().await.hold(&model, &body(sent)),
            Ok(Some(Hold {
                reservation_cost: input_cost + output_cost,
                input_cost,
                input_tokens: COUNTED,
            }))
        );
    }

    #[rstest]
    #[case::a_group_named_unlike_its_model(json!({"model": "renamed", "max_tokens": 16}))]
    #[case::a_model_counted_with_tiktoken(json!({"model": "openai-model", "max_tokens": 16}))]
    #[case::a_deployment_that_drops_params(json!({"model": "dropping-model", "max_tokens": 16}))]
    #[case::an_unknown_model(json!({"model": "other", "max_tokens": 16}))]
    #[case::another_output_cap(json!({"model": "held-model", "max_tokens": 16, "max_completion_tokens": 8}))]
    #[case::an_output_count(json!({"model": "held-model", "n": 2}))]
    #[case::a_cap_written_as_text(json!({"model": "held-model", "max_tokens": "16"}))]
    #[case::a_negative_cap(json!({"model": "held-model", "max_tokens": -1}))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_the_gateway_does_not_price_as_python_does_is_not_held(#[case] sent: Value) {
        let model = sent["model"].as_str().unwrap().to_owned();

        assert_eq!(prices().await.hold(&model, &body(sent)), Err(Unheld));
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_body_the_tokenizer_cannot_count_is_not_held() {
        let sent = Map::from_iter([
            ("model".to_owned(), json!("held-model")),
            ("messages".to_owned(), json!("not a list")),
        ]);

        assert_eq!(prices().await.hold("held-model", &sent), Err(Unheld));
    }
}
