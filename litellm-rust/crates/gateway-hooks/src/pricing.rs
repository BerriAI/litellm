//! Per-deployment prices, built once at startup from the model cost map and any prices the
//! config sets on a deployment, with the proxy's key names: `input_cost_per_token`, its
//! `_above_<N>k_tokens` thresholds and its `_priority` and `_flex` tiers.

use std::{collections::HashMap, sync::Arc};

use litellm_cost::{
    PricingPlan, PromptConvention, Rate, Rates, Request, ServiceTier as CostTier, ThresholdPolicy,
    ThresholdRates, TierRates,
};
use litellm_model_catalog::Catalog;
use serde_json::{Map, Value};

use crate::usage::{ServiceTier, Usage};

/// Prices by deployment id.
#[derive(Debug, Default)]
pub struct Prices {
    plans: HashMap<Arc<str>, PricingPlan>,
}

impl Prices {
    /// Adds a deployment's prices from its cost map fields. A deployment whose fields hold no
    /// usable prices is left out, and its cost is unknown.
    pub fn insert(&mut self, id: Arc<str>, fields: &Map<String, Value>) {
        let parsed = ParsedRates::parse(fields);
        if parsed.standard == Rates::EMPTY && parsed.thresholds.is_empty() {
            return;
        }
        match parsed.compile() {
            Ok(plan) => {
                self.plans.insert(id, plan);
            }
            Err(error) => tracing::warn!(deployment = %id, ?error, "ignoring invalid prices"),
        }
    }

    pub fn len(&self) -> usize {
        self.plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// The cost in dollars of `usage` on a deployment, or `None` when its prices are unknown.
    pub fn cost(&self, deployment: &str, usage: &Usage) -> Option<f64> {
        let plan = self.plans.get(deployment)?;
        let one_hour = usage.cache_write_1h_tokens.min(usage.cache_write_tokens);
        let split = (one_hour > 0).then_some(());
        let request = Request {
            usage: litellm_cost::Usage {
                prompt_tokens: usage.prompt_tokens(),
                completion_tokens: usage.output_tokens,
                cache_read_tokens: usage.cache_read_tokens,
                cache_write_tokens: usage.cache_write_tokens,
                cache_write_5m_tokens: split.map(|()| usage.cache_write_tokens - one_hour),
                cache_write_1h_tokens: split.map(|()| one_hour),
                prompt_convention: PromptConvention::IncludesCache,
            },
            service_tier: match usage.service_tier {
                ServiceTier::Priority => CostTier::Priority,
                ServiceTier::Flex => CostTier::Flex,
                ServiceTier::Standard | ServiceTier::Batch => CostTier::Standard,
            },
            threshold_policy: ThresholdPolicy::Exclusive,
            region_multiplier: None,
            billed_at_utc_minute: None,
        };
        match plan.calculate(&request) {
            Ok(cost) => Some(cost.total()),
            Err(error) => {
                tracing::debug!(deployment, ?error, "no cost for response");
                None
            }
        }
    }
}

/// A deployment's cost map fields: the cost map entry for its model, overlaid by the prices its
/// config sets. The entry is found by `base_model`, then the model name, then the name without
/// its provider route.
pub fn deployment_fields(
    catalog: Option<&Catalog>,
    model: &str,
    base_model: Option<&str>,
    overrides: &[&Map<String, Value>],
) -> Map<String, Value> {
    let mut fields = catalog
        .and_then(|catalog| {
            let candidates = base_model.into_iter().chain([
                model,
                model.split_once('/').map_or(model, |(_, rest)| rest),
                model.rsplit_once('/').map_or(model, |(_, rest)| rest),
            ]);
            candidates
                .filter(|name| !name.is_empty())
                .find_map(|name| catalog.lookup(name))
        })
        .map(|found| found.entry.fields().clone())
        .unwrap_or_default();
    for layer in overrides {
        for (name, value) in layer.iter() {
            if price_key(name).is_some() {
                fields.insert(name.clone(), value.clone());
            }
        }
    }
    fields
}

#[derive(Clone, Copy)]
enum Slot {
    Input,
    Output,
    CacheRead,
    CacheWrite,
    CacheWrite1h,
}

/// Longest first, so the hour-long cache write is not read as a threshold of the plain one.
const SLOTS: [(&str, Slot); 5] = [
    (
        "cache_creation_input_token_cost_above_1hr",
        Slot::CacheWrite1h,
    ),
    ("cache_creation_input_token_cost", Slot::CacheWrite),
    ("cache_read_input_token_cost", Slot::CacheRead),
    ("input_cost_per_token", Slot::Input),
    ("output_cost_per_token", Slot::Output),
];

struct PriceKey {
    slot: Slot,
    above_tokens: Option<u64>,
    tier: CostTier,
}

fn price_key(name: &str) -> Option<PriceKey> {
    let (prefix, slot) = SLOTS.iter().find(|(prefix, _)| name.starts_with(prefix))?;
    let mut rest = &name[prefix.len()..];
    let mut above_tokens = None;
    if let Some(threshold) = rest.strip_prefix("_above_") {
        let (count, after) = threshold.split_once("k_tokens")?;
        let thousands: f64 = count.parse().ok()?;
        above_tokens = Some((thousands * 1000.0) as u64);
        rest = after;
    }
    let tier = match rest {
        "" => CostTier::Standard,
        "_priority" => CostTier::Priority,
        "_flex" => CostTier::Flex,
        _ => return None,
    };
    Some(PriceKey {
        slot: *slot,
        above_tokens,
        tier,
    })
}

/// A price, from a number or, as the proxy's `float()` reads config values, a numeric string.
fn rate(value: &Value) -> Rate {
    match value {
        Value::Number(number) => number.as_f64().map_or(Rate::Null, Rate::Value),
        Value::String(text) => text.trim().parse().map_or(Rate::Null, Rate::Value),
        _ => Rate::Null,
    }
}

fn set(rates: &mut Rates, slot: Slot, value: Rate) {
    *match slot {
        Slot::Input => &mut rates.input,
        Slot::Output => &mut rates.output,
        Slot::CacheRead => &mut rates.cache_read,
        Slot::CacheWrite => &mut rates.cache_write,
        Slot::CacheWrite1h => &mut rates.cache_write_1h,
    } = value;
}

fn tier_rates(tiers: &mut Vec<TierRates>, tier: CostTier) -> &mut Rates {
    let index = match tiers.iter().position(|entry| entry.tier == tier) {
        Some(index) => index,
        None => {
            tiers.push(TierRates {
                tier,
                rates: Rates::EMPTY,
            });
            tiers.len() - 1
        }
    };
    &mut tiers[index].rates
}

struct ParsedThreshold {
    above_tokens: u64,
    standard: Rates,
    tiers: Vec<TierRates>,
}

struct ParsedRates {
    standard: Rates,
    tiers: Vec<TierRates>,
    thresholds: Vec<ParsedThreshold>,
}

impl ParsedRates {
    fn parse(fields: &Map<String, Value>) -> Self {
        let mut parsed = Self {
            standard: Rates::EMPTY,
            tiers: Vec::new(),
            thresholds: Vec::new(),
        };
        for (name, value) in fields {
            let Some(key) = price_key(name) else { continue };
            let (standard, tiers) = match key.above_tokens {
                None => (&mut parsed.standard, &mut parsed.tiers),
                Some(above_tokens) => {
                    let index = match parsed
                        .thresholds
                        .iter()
                        .position(|entry| entry.above_tokens == above_tokens)
                    {
                        Some(index) => index,
                        None => {
                            parsed.thresholds.push(ParsedThreshold {
                                above_tokens,
                                standard: Rates::EMPTY,
                                tiers: Vec::new(),
                            });
                            parsed.thresholds.len() - 1
                        }
                    };
                    let entry = &mut parsed.thresholds[index];
                    (&mut entry.standard, &mut entry.tiers)
                }
            };
            let rates = match key.tier {
                CostTier::Standard => standard,
                tier => tier_rates(tiers, tier),
            };
            set(rates, key.slot, rate(value));
        }
        // The proxy applies a threshold only when it prices input above it.
        parsed.thresholds.retain(|entry| {
            matches!(entry.standard.input, Rate::Value(_))
                || entry
                    .tiers
                    .iter()
                    .any(|tier| matches!(tier.rates.input, Rate::Value(_)))
        });
        parsed
    }

    fn compile(&self) -> Result<PricingPlan, litellm_cost::PricingError> {
        let thresholds: Vec<_> = self
            .thresholds
            .iter()
            .map(|entry| ThresholdRates {
                above_prompt_tokens: entry.above_tokens,
                standard: entry.standard,
                tiers: &entry.tiers,
            })
            .collect();
        litellm_cost::compile(&litellm_cost::Pricing {
            standard: self.standard,
            tiers: &self.tiers,
            thresholds: &thresholds,
            off_peak: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn prices(fields: Value) -> Prices {
        let mut prices = Prices::default();
        prices.insert("d".into(), fields.as_object().unwrap());
        prices
    }

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            ..Usage::default()
        }
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() < 1e-12
    }

    #[rstest]
    fn standard_rates() {
        let prices = prices(json!({
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 0.3e-6,
            "cache_creation_input_token_cost": 3.75e-6,
            "cache_creation_input_token_cost_above_1hr": 6e-6,
        }));
        let usage = Usage {
            cache_read_tokens: 100,
            cache_write_tokens: 30,
            cache_write_1h_tokens: 10,
            ..usage(1000, 200)
        };
        let expected =
            1000.0 * 3e-6 + 200.0 * 15e-6 + 100.0 * 0.3e-6 + 20.0 * 3.75e-6 + 10.0 * 6e-6;
        assert!(close(prices.cost("d", &usage).unwrap(), expected));
    }

    #[rstest]
    #[case::below(200_000, 1e-6)]
    #[case::above(200_001, 2e-6)]
    fn thresholds_count_the_whole_prompt(#[case] input: u64, #[case] rate: f64) {
        let prices = prices(json!({
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 0.0,
            "input_cost_per_token_above_200k_tokens": 2e-6,
        }));
        assert!(close(
            prices.cost("d", &usage(input, 0)).unwrap(),
            input as f64 * rate
        ));
    }

    #[rstest]
    fn priority_tier_rates() {
        let prices = prices(json!({
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 2e-6,
            "input_cost_per_token_priority": 4e-6,
        }));
        let usage = Usage {
            service_tier: ServiceTier::Priority,
            ..usage(10, 10)
        };
        assert!(close(
            prices.cost("d", &usage).unwrap(),
            10.0 * 4e-6 + 10.0 * 2e-6
        ));
    }

    #[rstest]
    fn unknown_deployments_and_unpriced_entries_have_no_cost() {
        let prices = prices(json!({"max_tokens": 10}));
        assert!(prices.is_empty());
        assert_eq!(prices.cost("d", &usage(1, 1)), None);
    }

    #[rstest]
    fn config_prices_override_the_cost_map() {
        let catalog = Catalog::parse(
            br#"{"claude-x": {"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6, "max_tokens": 5}}"#,
            Default::default(),
        )
        .unwrap();
        let config = json!({"input_cost_per_token": 9e-6, "api_key": "secret"});
        let fields = deployment_fields(
            Some(&catalog),
            "anthropic/claude-x",
            None,
            &[config.as_object().unwrap()],
        );
        assert_eq!(fields["input_cost_per_token"], json!(9e-6));
        assert_eq!(fields["output_cost_per_token"], json!(2e-6));
        assert!(!fields.contains_key("api_key"));
    }
}
