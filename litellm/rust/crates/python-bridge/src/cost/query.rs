//! The pricing queries Python asks `NativeModelCostMap`: `price`, `bound`, `rates` and `adjust`. Each takes the raw
//! facts Python read off its own objects as a document and hands them to litellm-cost over one call's view of the
//! live table, then answers with a Python value. A call the core does not price answers `None`, and Python prices it
//! the old way

use std::sync::Arc;

use litellm_callbacks_legacy_python::NativeReceipt;
use litellm_cost::adjust::CostConfig;
use litellm_cost::adjust::{AdjustmentKey, Discounts, MarginRule, Margins, adjust as adjusted};
use litellm_cost::bound::{
    BoundModel, BoundPolicy, BoundQuery, CostShape, FlatSource, GroupRates, RequestCaps,
};
use litellm_cost::candidates::{
    PricingRows, PricingTable, TokenCall, priced, priced_ref, pricing_rows,
};
use litellm_cost::price::{InputUnit, Meter, OutputUnit, TokenKind};
use litellm_cost::rates::{Pricing, RatesQuery};
use litellm_cost::receipt::Receipt;
use litellm_host_python::{from_py_argument, json_from_py};
use litellm_inference_messages::PricedCall;
use litellm_model_catalog::{CatalogView, CostLookup, LoweredRow};
use litellm_python_compat::number::float_of;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;
use serde_json::Value;

use crate::model_cost::{CostSnapshot, row_fields};

#[derive(Deserialize)]
struct BoundDocument {
    models: Vec<BoundModelFacts>,
    request: RequestCaps,
}

#[derive(Deserialize)]
struct BoundModelFacts {
    #[serde(default)]
    counted_input: Option<i64>,
    requested_output: i64,
    output_multiplier: i64,
    shapes: Vec<ShapeFacts>,
}

/// One cost shape: the flat rates of a model info row or the router's group rates, beside a tier table and the
/// group's deployment prices. A row is the index of a mapping passed beside the document
#[derive(Deserialize)]
struct ShapeFacts {
    flat: FlatFacts,
    #[serde(default)]
    tiers: Option<usize>,
    #[serde(default)]
    deployments: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
enum FlatFacts {
    Row { row: usize },
    Group(GroupRates),
}

/// A row's flat token rates, the raw key facts the auth checks read and whether it bills anything
#[pyclass(
    frozen,
    get_all,
    name = "NativeRateCard",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeRateCard {
    /// `priced`, `explicitly_free` or `unpriced`
    pricing: &'static str,
    input_cost_per_token: Option<f64>,
    output_cost_per_token: Option<f64>,
    token_rate_key: bool,
    positive_cost_per_key: bool,
    cost_per_key: bool,
    ptu_flat_cost: bool,
}

/// `cost_discount_config` and `cost_margin_config` read the way `_apply_cost_discount` and `_apply_cost_margin` read
/// them. A discount main could not multiply is left out
pub(crate) fn adjustment_config(
    discount: &Bound<'_, PyAny>,
    margin: &Bound<'_, PyAny>,
) -> PyResult<(Discounts, Margins)> {
    let entries = |config: &Bound<'_, PyAny>| -> PyResult<Vec<(String, Value)>> {
        Ok(match json_from_py(config)? {
            Value::Object(entries) => entries.into_iter().collect(),
            _ => Vec::new(),
        })
    };
    let discounts = entries(discount)?
        .into_iter()
        .filter_map(|(key, fraction)| Some((key, float_of(&fraction).ok()?)));
    let margins = entries(margin)?
        .into_iter()
        .map(|(key, rule)| (key, margin_rule(&rule)));
    Ok((Discounts::new(discounts), Margins::new(margins)))
}

fn margin_rule(rule: &Value) -> MarginRule {
    match rule {
        Value::Number(_) | Value::Bool(_) => {
            float_of(rule).map_or(MarginRule::Ignored, MarginRule::Percent)
        }
        Value::Object(split) => MarginRule::Split {
            percentage: split
                .get("percentage")
                .and_then(|value| float_of(value).ok()),
            fixed_amount: split
                .get("fixed_amount")
                .and_then(|value| float_of(value).ok()),
        },
        _ => MarginRule::Ignored,
    }
}

/// Prices a token call, `None` when the core leaves it to Python
pub(crate) fn price<V: CatalogView + ?Sized>(
    py: Python<'_>,
    lookup: &CostLookup<'_, V>,
    document: &Bound<'_, PyAny>,
    row: Option<&Bound<'_, PyMapping>>,
) -> PyResult<Option<NativeReceipt>> {
    let call: TokenCall = from_py_argument(document)?;
    let row = row.map(row_fields).transpose()?;
    let per_call =
        row.map(|fields| move |provider: &str| LoweredRow::lower(Arc::new(fields), provider));
    priced(lookup, call, per_call)
        .map(|receipt| NativeReceipt::new(py, &receipt))
        .transpose()
}

/// The Messages route's pricing over the snapshot taken when the call was set up. Each call reads that snapshot
/// through one `CostLookup`, and `litellm_inference_messages::price_call` reads the call the way `completion_cost`
/// reads a native `/v1/messages` result on main
pub(crate) struct RoutePricing {
    snapshot: Arc<CostSnapshot>,
}

impl RoutePricing {
    pub(crate) fn new(snapshot: Arc<CostSnapshot>) -> Self {
        Self { snapshot }
    }
}

impl litellm_inference_messages::Pricing for RoutePricing {
    fn price(&self, call: &PricedCall) -> Option<Receipt> {
        self.snapshot.lookup(|lookup| {
            litellm_inference_messages::price_call(lookup, call, jiff::Timestamp::now())
        })
    }
}

/// The legacy budget hold and input floor, `(None, None)` when no shape prices. `rows` are the model info mappings
/// the shapes read, as main read them, so they are priced for this call alone
pub(crate) fn bound<V: CatalogView + ?Sized>(
    lookup: &CostLookup<'_, V>,
    document: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<(Option<f64>, Option<f64>)> {
    let document: BoundDocument = from_py_argument(document)?;
    let (source, refs) = rows.iter().enumerate().try_fold(
        (pricing_rows(lookup, [], []), Vec::new()),
        |(source, refs), (index, row)| -> PyResult<_> {
            let fields = row_fields(row)?;
            let provider = fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let row = LoweredRow::lower(Arc::new(fields), &provider);
            let (source, at) = source.with_per_call(&format!("bound-{index}"), row);
            Ok((source, refs.into_iter().chain([at]).collect::<Vec<_>>()))
        },
    )?;
    let at = |index: usize| refs.get(index).cloned();
    let query = BoundQuery {
        models: document
            .models
            .into_iter()
            .map(|model| BoundModel {
                counted_input: model.counted_input,
                requested_output: model.requested_output,
                output_multiplier: model.output_multiplier,
                candidates: model
                    .shapes
                    .into_iter()
                    .filter_map(|shape| {
                        Some(CostShape {
                            flat: match shape.flat {
                                FlatFacts::Row { row } => FlatSource::Row { row: at(row)? },
                                FlatFacts::Group(rates) => FlatSource::Group(rates),
                            },
                            tiers: shape.tiers.and_then(at),
                            deployments: shape.deployments.into_iter().filter_map(at).collect(),
                        })
                    })
                    .collect(),
            })
            .collect(),
        request: document.request,
        policy: BoundPolicy::Legacy,
    };
    let held = litellm_cost::bound(&source, &query);
    Ok((held.max, held.input_floor))
}

/// The rate card of the row under exactly `key`, `None` when the table has no such row
pub(crate) fn rates<V: CatalogView + ?Sized>(
    lookup: &CostLookup<'_, V>,
    key: &str,
) -> Option<NativeRateCard> {
    let row = lookup.row_at(key)?;
    let at = priced_ref(&row);
    let rows: PricingRows<LoweredRow> = pricing_rows(lookup, [], [row]);
    let card = litellm_cost::rates(
        &rows,
        &RatesQuery {
            row: at,
            prompt_tokens: None,
        },
    )?;
    let declared = card.declared;
    let rate = |meter: Meter| card.rate(&meter).map(|rate| rate.get());
    Some(NativeRateCard {
        pricing: match card.pricing {
            Pricing::Priced => "priced",
            Pricing::ExplicitlyFree => "explicitly_free",
            Pricing::Unpriced => "unpriced",
        },
        input_cost_per_token: rate(Meter::Input(InputUnit::Token(TokenKind::Text))),
        output_cost_per_token: rate(Meter::Output(OutputUnit::Token(TokenKind::Text))),
        token_rate_key: declared.token_rate_key,
        positive_cost_per_key: declared.positive_cost_per_key,
        cost_per_key: declared.cost_per_key,
        ptu_flat_cost: declared.ptu_flat_cost,
    })
}

/// `_apply_cost_discount` then `_apply_cost_margin`: the adjusted cost, the discount percent and amount, and the
/// margin percent, fixed amount and total
pub(crate) fn adjust(
    config: &CostConfig,
    cost: f64,
    provider: Option<&str>,
    model: Option<&str>,
    region: Option<&str>,
) -> (f64, f64, f64, f64, f64, f64) {
    let applied = adjusted(
        &config.discounts,
        &config.margins,
        cost,
        &AdjustmentKey {
            provider,
            model,
            region,
        },
    );
    (
        applied.cost,
        applied.discount.percent,
        applied.discount.amount,
        applied.margin.percent,
        applied.margin.fixed,
        applied.margin.total,
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::percent(json!(0.1), MarginRule::Percent(0.1))]
    #[case::int_percent(json!(1), MarginRule::Percent(1.0))]
    #[case::split(
        json!({"percentage": 0.08, "fixed_amount": 0.0005}),
        MarginRule::Split { percentage: Some(0.08), fixed_amount: Some(0.0005) }
    )]
    #[case::fixed_only(
        json!({"fixed_amount": 0.25}),
        MarginRule::Split { percentage: None, fixed_amount: Some(0.25) }
    )]
    #[case::a_word_adds_nothing(json!("ten percent"), MarginRule::Ignored)]
    fn a_margin_rule_reads_as_apply_cost_margin_reads_it(
        #[case] rule: Value,
        #[case] expected: MarginRule,
    ) {
        assert_eq!(margin_rule(&rule), expected);
    }

    #[rstest]
    fn the_adjustment_config_keeps_the_order_python_wrote_and_skips_a_discount_it_cannot_multiply()
    {
        Python::initialize();
        let (discounts, margins) = Python::attach(|py| {
            let discount = py
                .eval(
                    pyo3::ffi::c_str!(
                        "{'anthropic/claude-*': 0.2, 'anthropic': 0.05, 'openai': None}"
                    ),
                    None,
                    None,
                )
                .unwrap();
            let margin = py
                .eval(pyo3::ffi::c_str!("{'global': 0.1}"), None, None)
                .unwrap();
            adjustment_config(&discount, &margin).unwrap()
        });
        let key = |model| AdjustmentKey {
            provider: Some("anthropic"),
            model: Some(model),
            region: None,
        };

        assert_eq!(discounts.resolve(&key("claude-opus-5")), Some(0.2));
        assert_eq!(discounts.resolve(&key("other")), Some(0.05));
        assert_eq!(
            discounts.resolve(&AdjustmentKey {
                provider: Some("openai"),
                ..key("gpt")
            }),
            None
        );
        assert_eq!(margins.apply(10.0, Some("openai")).1.total, 1.0);
    }
}
