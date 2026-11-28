//! What the gateway prices a Messages call with: the installed cost map, read the way the Python
//! host's cost map reads it, and each deployment's pricing identity, as Python's
//! `_pricing_identity` sends it.
//!
//! How one `/v1/messages` call's cost reaches its spend log row:
//!
//! 1. Boot: `Gateway::new` hands the messages route a [`CostMapPricing`] over the
//!    gateway's cost map, so a call prices over whichever map is installed when it is priced.
//! 2. Each attempt: `MessagesAttempt::run` (`messages.rs`) sets the call's
//!    `MessagesSettings.pricing` to [`priced_as`], or `None` to leave it unpriced.
//! 3. The route (`inference-messages` `handler.rs`, `execute` and `execution_facts`) settles a
//!    `Cost`: the provider's own figure, a cache hit, a message body it prices now through
//!    `Pricing::price` (this module's `price_call` over the cost map), or a `StreamMeter` for a
//!    stream. Anything it cannot price stays `Deferred`.
//! 4. `Recorder::result_ready` (`record/attempt.rs`) keeps that cost in the attempt's
//!    `AttemptRecord`, and only the served attempt's record reaches the call's record.
//! 5. A stream's meter settles when the stream ends after its final `message_delta`
//!    (`inference-messages` `pricing.rs`, `metered_stream`). The meter is shared, so the cost
//!    already in the attempt's record sees the price.
//! 6. The spend log callback reads `Field::Cost`, which `cost` in the gateway's
//!    `callbacks/fields.rs` turns into `response_cost`: the receipt total, the settled meter's
//!    total, the provider's amount, 0 for a cache hit, and nothing for `Deferred` or a meter that
//!    never settled.
//!
//! The route's side is described in full in `inference-messages/AGENTS.md` (Pricing) and
//! `host/AGENTS.md` (`ExecutionFacts` and `Cost`).

use std::collections::HashSet;
use std::sync::Arc;

use arc_swap::{ArcSwap, ArcSwapOption};
use litellm_auth::VertexParams;
use litellm_cost::receipt::Receipt;
use litellm_inference::Error as InferenceError;
use litellm_inference_messages::{PricedCall, Pricing, PricingIdentity, price_call};
use litellm_llms::{bedrock_mantle::messages::transformation as mantle, vertex_ai::common_utils};
use litellm_model_catalog::wire::EntryWire;
use litellm_model_catalog::{Catalog, CostLookup, CostMap, KnownNames, PricingContext};
use litellm_router_core::Deployment;
use litellm_secrets::source::SecretSource;
use serde_json::{Map, Value};

use crate::Error;

const RENAMED_MODELS: [&str; 6] = [
    "gpt-35-turbo",
    "gpt-35-turbo-instruct",
    "azure/gpt-41",
    "azure/gpt-41-mini",
    "azure/gpt-41-nano",
    "ada",
];

/// Fields of Python's `CustomPricingLiteLLMParams` that the cost map does not read as prices.
const CUSTOM_PRICING_KEYS_OUTSIDE_PRICES: [&str; 4] = [
    "input_cost_per_token_cache_hit",
    "output_cost_per_audio_per_second",
    "output_vector_size",
    "search_queries_cost_per_query",
];

/// Prices a call over the cost map installed when it is priced, with no discount or margin config.
pub(crate) struct CostMapPricing {
    cost_map: Arc<ArcSwap<CostMap>>,
    /// The cost map last priced over and how it is read, read again once another is installed.
    read: ArcSwapOption<Read>,
}

struct Read {
    cost_map: Arc<CostMap>,
    context: PricingContext,
}

impl CostMapPricing {
    pub(crate) fn new(cost_map: Arc<ArcSwap<CostMap>>) -> Self {
        Self {
            cost_map,
            read: ArcSwapOption::empty(),
        }
    }

    fn read(&self) -> Arc<Read> {
        let cost_map = self.cost_map.load_full();
        if let Some(read) = self.read.load_full()
            && Arc::ptr_eq(&read.cost_map, &cost_map)
        {
            return read;
        }
        let read = Arc::new(Read {
            context: context(cost_map.catalog()),
            cost_map,
        });
        self.read.store(Some(read.clone()));
        read
    }
}

/// How `catalog` is read: its own providers are the names a model path may start with.
fn context(catalog: &Catalog) -> PricingContext {
    let providers = catalog
        .served()
        .filter_map(|(_, fields)| fields.get("litellm_provider")?.as_str())
        .map(str::to_owned)
        .collect();
    PricingContext {
        rules: catalog.capability_rules().clone(),
        names: KnownNames {
            providers,
            renamed_models: RENAMED_MODELS
                .into_iter()
                .map(str::to_owned)
                .collect::<HashSet<_>>(),
        },
        ..PricingContext::default()
    }
}

impl Pricing for CostMapPricing {
    fn price(&self, call: &PricedCall) -> Option<Receipt> {
        let read = self.read();
        let lookup = CostLookup::new(read.cost_map.catalog(), &read.context);
        price_call(&lookup, call, jiff::Timestamp::now())
    }
}

/// What a call to `deployment` is priced with, its region read from `secrets` as the route reads
/// it. See [`identity`].
pub(crate) async fn priced_as(
    deployment: &Deployment,
    catalog: &Catalog,
    secrets: &dyn SecretSource,
) -> Result<Option<PricingIdentity>, Error> {
    let names: &[&str] = match regional(deployment) {
        Some(Regional::Vertex) => VertexParams::LOCATION.env,
        Some(Regional::Mantle) => &mantle::COST_REGION_SECRET_NAMES,
        None => &[],
    };
    let secrets = match names.is_empty() {
        true => None,
        false => Some(secrets.resolve(names).await.map_err(InferenceError::from)?),
    };
    Ok(identity(deployment, catalog, &|name| {
        secrets.as_ref()?.get(name)
    }))
}

/// A region no cost map row names, standing in for whatever region a deployment's secrets name.
const REGION_WITHOUT_A_ROW: &str = "region-without-a-row";

/// Whether the route prices a call to `deployment` over `catalog` in whatever region it is sent to.
/// Its region is read from its params as the route reads it, but not from its secrets, so the call
/// is priced both in the region it takes when its secrets name none and in a region with no row of
/// its own, which is where any region its secrets could name and the cost map does not list prices.
pub(crate) fn priced_wherever_served(
    pricing: &dyn Pricing,
    deployment: &Deployment,
    catalog: &Catalog,
) -> bool {
    [None, Some(REGION_WITHOUT_A_ROW)]
        .into_iter()
        .all(|secret| {
            identity(deployment, catalog, &|_| secret.map(str::to_owned)).is_some_and(|identity| {
                litellm_inference_messages::priced(
                    pricing,
                    &deployment.params.model,
                    deployment.params.custom_llm_provider.as_deref(),
                    identity,
                )
            })
        })
}

/// A deployment whose price reads where it is served.
enum Regional {
    Vertex,
    Mantle,
}

fn regional(deployment: &Deployment) -> Option<Regional> {
    match deployment.provider() {
        provider if provider.starts_with("vertex_ai") => Some(Regional::Vertex),
        "bedrock_mantle" => Some(Regional::Mantle),
        "bedrock" => deployment
            .params
            .model
            .trim_start_matches("bedrock/")
            .starts_with("mantle/")
            .then_some(Regional::Mantle),
        _ => None,
    }
}

/// Whether `fields`, read by the cost map as `wire`, set a price of their own the way Python's
/// `_CUSTOM_PRICING_KEYS` finds one. `off_peak_pricing` counts only in `model_info`, so the caller
/// checks it.
fn sets_prices(fields: &Map<String, Value>, wire: &EntryWire) -> bool {
    !wire.prices.is_empty()
        || wire.tier_rows.is_some()
        || wire.search_context.is_some()
        || wire.output_cost_per_video_clip.is_some()
        || wire.regional_endpoint_uplift.is_some()
        || wire.residency_uplift_eu.is_some()
        || wire.residency_uplift_us.is_some()
        || wire.fedramp_uplift.is_some()
        || CUSTOM_PRICING_KEYS_OUTSIDE_PRICES
            .iter()
            .any(|key| fields.get(*key).is_some_and(|value| !value.is_null()))
}

/// What a call to `deployment` is priced with. `None`, so the call goes unpriced, for one Python
/// prices another way: custom pricing in its params or `model_info`, a `pricing_provider`, or a
/// Bedrock `model_id`. A Vertex AI deployment prices in the location the call is sent to, and a
/// Bedrock Mantle one in its Mantle region, as Python's `resolve_vertex_location_for_cost` and
/// `_resolve_mantle_region_for_cost` resolve them, with `env_lookup` as the environment.
pub(crate) fn identity(
    deployment: &Deployment,
    catalog: &Catalog,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<PricingIdentity> {
    let params = &deployment.params;
    let model_info = &deployment.model_info.raw;
    let set = |value: Option<&Value>| value.is_some_and(|value| !value.is_null());
    let model_info_prices = EntryWire::read(model_info);
    let custom = sets_prices(
        &deployment.raw_params,
        &EntryWire::read(&deployment.raw_params),
    ) || sets_prices(model_info, &model_info_prices)
        || model_info_prices.off_peak.is_some();
    if custom || set(params.extra.get("model_id")) || set(model_info.get("pricing_provider")) {
        return None;
    }
    let regional = regional(deployment);
    let text = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_owned);
    Some(PricingIdentity {
        deployment_id: Some(deployment.id.clone()),
        base_model: text(params.extra.get("base_model"))
            .or_else(|| text(model_info.get("base_model"))),
        region: match regional {
            Some(Regional::Mantle) => {
                mantle::cost_region(deployment.provider(), &params.model, params, env_lookup)
            }
            _ => None,
        },
        vertex_location: match regional {
            Some(Regional::Vertex) => Some(common_utils::cost_location(
                &params.vertex,
                &params.model,
                catalog,
                env_lookup,
            )),
            _ => None,
        },
        data_residency: text(params.extra.get("data_residency")),
    })
}

#[cfg(test)]
mod tests {
    use litellm_cost::query::{Modalities, TokenUsage};
    use litellm_model_catalog::IntegrityLimits;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::deployment;

    fn call(model: &str) -> PricedCall {
        PricedCall {
            provider: "anthropic".to_owned(),
            request_model: model.to_owned(),
            response_model: Some(model.to_owned()),
            identity: PricingIdentity {
                deployment_id: Some("deployment-1".to_owned()),
                ..PricingIdentity::default()
            },
            usage: TokenUsage {
                prompt_tokens: 1000,
                input: Modalities {
                    text: 1000,
                    ..Modalities::default()
                },
                output: Modalities {
                    text: 100,
                    ..Modalities::default()
                },
                ..TokenUsage::default()
            },
            service_tier: None,
        }
    }

    /// The bundled cost map prices a Claude call at its row's own rates.
    #[rstest]
    fn a_call_is_priced_at_its_cost_map_rows_rates() {
        let cost_map = CostMap::bundled();
        let row = cost_map.catalog().get("claude-opus-5-5").unwrap();
        let rate = |key: &str| row[key].as_f64().unwrap();
        let listed = 1000.0 * rate("input_cost_per_token") + 100.0 * rate("output_cost_per_token");
        let pricing = CostMapPricing::new(Arc::new(ArcSwap::from_pointee(cost_map)));

        let receipt = pricing.price(&call("claude-opus-5-5")).unwrap();

        assert!((receipt.total - listed).abs() < 1e-12);
    }

    /// A cost map installed after a call was priced prices the next call.
    #[rstest]
    fn a_call_is_priced_over_the_cost_map_installed_last() {
        let installed = Arc::new(ArcSwap::from_pointee(CostMap::bundled()));
        let pricing = CostMapPricing::new(installed.clone());
        pricing.price(&call("claude-opus-5-5")).unwrap();
        installed.store(Arc::new(
            CostMap::read(
                br#"{"claude-opus-5-5": {"litellm_provider": "anthropic", "mode": "chat", "input_cost_per_token": 0.001, "output_cost_per_token": 0.01}}"#,
                None,
                IntegrityLimits::UNCHECKED,
            )
            .unwrap(),
        ));

        let receipt = pricing.price(&call("claude-opus-5-5")).unwrap();

        assert!((receipt.total - 2.0).abs() < 1e-12);
    }

    fn identity_of(
        params: Value,
        model_info: Value,
        env: &[(&str, &str)],
    ) -> Option<PricingIdentity> {
        let catalog = Catalog::parse(
            br#"{"vertex_ai/claude-regional": {"litellm_provider": "vertex_ai-anthropic_models", "supported_regions": ["us-east5", "europe-west1"]}}"#,
        )
        .unwrap()
        .value;
        let deployment = deployment(
            params,
            json!({"id": "deployment-1"})
                .as_object()
                .cloned()
                .map(|mut info| {
                    info.extend(model_info.as_object().cloned().unwrap());
                    Value::Object(info)
                })
                .unwrap(),
        );
        identity(&deployment, &catalog, &|name| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        })
    }

    #[rstest]
    #[case::a_cost_map_priced_deployment(json!({}), json!({}), Some(None))]
    #[case::a_base_model_in_its_params(json!({"base_model": "claude-opus-5-5"}), json!({}), Some(Some("claude-opus-5-5")))]
    #[case::a_base_model_in_its_model_info(json!({}), json!({"base_model": "claude-opus-5-5"}), Some(Some("claude-opus-5-5")))]
    #[case::custom_pricing_in_its_params(json!({"input_cost_per_token": 0.000001}), json!({}), None)]
    #[case::custom_pricing_in_its_model_info(json!({}), json!({"output_cost_per_token": 0.000002}), None)]
    #[case::off_peak_pricing_in_its_model_info(json!({}), json!({"off_peak_pricing": {}}), None)]
    #[case::off_peak_pricing_in_its_params(json!({"off_peak_pricing": {}}), json!({}), Some(None))]
    #[case::a_flex_image_output_rate(json!({"output_cost_per_image_token_flex": 0.000001}), json!({}), None)]
    #[case::a_band_and_tier_combination(json!({}), json!({"input_cost_per_token_above_512k_tokens_priority": 0.000001}), None)]
    #[case::tiered_pricing(json!({"tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 0.000001}]}), json!({}), None)]
    #[case::an_uplift_multiplier(json!({}), json!({"fedramp_uplift_multiplier": 1.2}), None)]
    #[case::a_cache_hit_rate(json!({"input_cost_per_token_cache_hit": 0.000001}), json!({}), None)]
    #[case::a_rate_the_cost_map_has_no_meter_for(json!({}), json!({"search_queries_cost_per_query": 0.01}), None)]
    #[case::a_custom_pricing_key_left_unset(json!({"input_cost_per_token": null}), json!({}), Some(None))]
    #[case::a_pricing_provider(json!({}), json!({"pricing_provider": "anthropic"}), None)]
    #[case::an_invoked_bedrock_model_id(json!({"model_id": "arn:aws:bedrock:model"}), json!({}), None)]
    fn a_deployment_is_priced_unless_python_prices_it_another_way(
        #[case] params: Value,
        #[case] model_info: Value,
        #[case] expected: Option<Option<&str>>,
    ) {
        let mut params = params.as_object().cloned().unwrap();
        params.insert("model".to_owned(), json!("anthropic/claude-opus-5-5"));

        let identity = identity_of(Value::Object(params), model_info, &[]);

        assert_eq!(
            identity.as_ref().map(|identity| (
                identity.deployment_id.as_deref(),
                identity.base_model.as_deref()
            )),
            expected.map(|base_model| (Some("deployment-1"), base_model))
        );
    }

    #[rstest]
    #[case::the_models_first_region_unconfigured(json!({"model": "vertex_ai/claude-regional"}), &[], "us-east5")]
    #[case::a_configured_region_the_model_serves(json!({"model": "vertex_ai/claude-regional", "vertex_location": "europe-west1"}), &[], "europe-west1")]
    #[case::the_other_spelling(json!({"model": "vertex_ai/claude-regional", "vertex_ai_location": "europe-west1"}), &[], "europe-west1")]
    #[case::a_configured_region_the_model_does_not_serve(json!({"model": "vertex_ai/claude-regional", "vertex_location": "asia-east1"}), &[], "us-east5")]
    #[case::the_environments_region(json!({"model": "vertex_ai/claude-regional"}), &[("VERTEX_LOCATION", "europe-west1")], "europe-west1")]
    #[case::a_model_with_no_regions_listed(json!({"model": "vertex_ai/claude-opus-5-5", "vertex_location": "asia-east1"}), &[], "asia-east1")]
    #[case::a_model_with_no_regions_unconfigured(json!({"model": "vertex_ai/claude-opus-5-5"}), &[], "us-central1")]
    fn a_vertex_ai_deployment_is_priced_in_the_location_it_is_sent_to(
        #[case] params: Value,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let identity = identity_of(params, json!({}), env).unwrap();

        assert_eq!(
            (identity.vertex_location.as_deref(), identity.region),
            (Some(expected), None)
        );
    }

    #[rstest]
    #[case::its_configured_region(json!({"model": "bedrock_mantle/claude-opus-5-5", "aws_region_name": "eu-west-1"}), &[("BEDROCK_MANTLE_REGION", "us-west-2")], "eu-west-1")]
    #[case::its_mantle_hosts_region(json!({"model": "bedrock_mantle/claude-opus-5-5", "api_base": "https://bedrock-mantle.ap-northeast-1.api.aws/v1"}), &[("BEDROCK_MANTLE_REGION", "us-west-2")], "ap-northeast-1")]
    #[case::the_environments_mantle_host(json!({"model": "bedrock_mantle/claude-opus-5-5"}), &[("BEDROCK_MANTLE_API_BASE", "https://bedrock-mantle.eu-central-1.api.aws"), ("AWS_REGION", "us-west-2")], "eu-central-1")]
    #[case::a_private_host_reads_the_environment(json!({"model": "bedrock_mantle/claude-opus-5-5", "api_base": "https://mantle.internal.test/v1"}), &[("AWS_REGION_NAME", "us-west-2")], "us-west-2")]
    #[case::the_mantle_default(json!({"model": "bedrock_mantle/claude-opus-5-5"}), &[], "us-east-1")]
    #[case::mantle_through_bedrock_configured(json!({"model": "bedrock/mantle/claude-opus-5-5", "aws_region_name": "eu-west-1"}), &[], "eu-west-1")]
    #[case::mantle_through_bedrock_from_the_environment(json!({"model": "bedrock/mantle/claude-opus-5-5"}), &[("AWS_REGION_NAME", "ap-south-1")], "ap-south-1")]
    fn a_bedrock_mantle_deployment_is_priced_in_its_region(
        #[case] params: Value,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let identity = identity_of(params, json!({}), env).unwrap();

        assert_eq!(
            (identity.region.as_deref(), identity.vertex_location),
            (Some(expected), None)
        );
    }

    /// The gateway's unpriced check passes only a deployment the route prices in whatever region it
    /// is sent to, so it never passes one the route leaves `Deferred` in the region its secrets name.
    #[rstest]
    #[case::a_mantle_row_for_the_default_region_only(json!({"model": "bedrock/mantle/claude-gateway-test"}), r#""bedrock/us-west-2/mantle/claude-gateway-test""#, (false, false))]
    #[case::a_mantle_region_in_its_params_with_its_own_row(json!({"model": "bedrock/mantle/claude-gateway-test", "aws_region_name": "us-west-2"}), r#""bedrock/us-west-2/mantle/claude-gateway-test""#, (true, true))]
    #[case::a_mantle_row_for_every_region(json!({"model": "bedrock/mantle/claude-gateway-test"}), r#""bedrock/mantle/claude-gateway-test""#, (true, true))]
    #[case::a_vertex_ai_location_in_its_params(json!({"model": "vertex_ai/claude-gateway-test", "vertex_location": "europe-west1"}), r#""vertex_ai/claude-gateway-test""#, (true, true))]
    #[case::a_vertex_ai_location_left_to_its_secrets(json!({"model": "vertex_ai/claude-gateway-test"}), r#""vertex_ai/claude-gateway-test""#, (true, true))]
    fn a_deployment_passes_the_unpriced_check_only_if_priced_where_its_secrets_send_it(
        #[case] params: Value,
        #[case] row: &str,
        #[case] expected: (bool, bool),
    ) {
        let cost_map = CostMap::read(
            format!(
                r#"{{{row}: {{"litellm_provider": "{}", "mode": "chat", "input_cost_per_token": 0.001, "output_cost_per_token": 0.01, "supported_regions": ["us-east5", "europe-west1"]}}}}"#,
                match row.contains("vertex_ai") {
                    true => "vertex_ai-anthropic_models",
                    false => "bedrock",
                }
            )
            .as_bytes(),
            None,
            IntegrityLimits::UNCHECKED,
        )
        .unwrap();
        let installed = Arc::new(ArcSwap::from_pointee(cost_map));
        let pricing = CostMapPricing::new(installed.clone());
        let cost_map = installed.load_full();
        let catalog = cost_map.catalog();
        let deployment = deployment(params, json!({"id": "deployment-1"}));
        let secrets = [
            ("AWS_REGION_NAME", "eu-west-1"),
            ("VERTEX_LOCATION", "europe-west1"),
        ];
        let routed = identity(&deployment, catalog, &|name| {
            secrets
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        })
        .is_some_and(|identity| {
            litellm_inference_messages::priced(
                &pricing,
                &deployment.params.model,
                deployment.params.custom_llm_provider.as_deref(),
                identity,
            )
        });

        assert_eq!(
            (
                priced_wherever_served(&pricing, &deployment, catalog),
                routed
            ),
            expected
        );
    }

    #[rstest]
    #[case::anthropic("anthropic/claude-opus-5-5")]
    #[case::bedrock_invoke("bedrock/us.anthropic.claude-opus-5-5")]
    fn a_deployment_served_in_one_place_names_no_region(#[case] model: &str) {
        let identity = identity_of(json!({"model": model}), json!({}), &[]).unwrap();

        assert_eq!((identity.region, identity.vertex_location), (None, None));
    }
}
