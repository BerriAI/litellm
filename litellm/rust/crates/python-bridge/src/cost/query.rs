//! The pricing queries Python asks `NativeModelCostMap`: `price`, `bound`, `rates` and `adjust`. Each takes the raw
//! facts Python read off its own objects as a document, interprets them the way main's Python did (which names to
//! try, the served tier, the residency), builds the pricing core's typed query over a snapshot of the live table and
//! answers with a typed result. A call the core does not price answers `None`, and Python prices it the old way

use std::sync::Arc;

use litellm_callbacks_legacy_python::NativeReceipt;
use litellm_cost::adjust::{AdjustmentKey, Discounts, MarginRule, Margins, adjust as adjusted};
use litellm_cost::bound::{
    BoundModel, BoundPolicy, BoundQuery, CostShape, FlatSource, GroupRates, RequestCaps,
};
use litellm_cost::money::UnitPrice;
use litellm_cost::price::{InputUnit, Meter, OutputUnit, TokenKind};
use litellm_cost::query::{
    AdjustmentPolicy, Basis, BreakdownRow, Candidate, CharacterCounts, DataResidency, Elapsed,
    FeeBilling, FlatRates, Header, PriceQuery, PriceSource, Provider, RawCounts, ReportedCost,
    RouterFee, TokenRules, TokenUsage, TokensQuery,
};
use litellm_cost::rates::{Pricing, RatesQuery};
use litellm_cost::receipt::{PriceOutcome, Receipt};
use litellm_host_python::{from_py_argument, json_from_py};
use litellm_inference_messages::PricedCall;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CallIdentity, Deferral, PricingRows, RowRef, Tried, priced_ref};
use litellm_python_compat::number::float_of;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;
use serde_json::Value;

use super::service_tier_named;
use crate::model_cost::{CostSnapshot, row_fields};

/// The key a row the caller passed for one call is priced under
const PER_CALL_ROW: &str = "requested_model_info_override";

/// The key of the row the caller looked up for the stored breakdown
const BREAKDOWN_ROW: &str = "breakdown";

/// The key of the Azure AI Model Router row whose fee the call bills
const ROUTER_FEE_ROW: &str = "router-fee";

/// The rows to try: the candidates main's `completion_cost` loop walks for a finished call, or the one model a
/// caller of `cost_per_token` names
#[derive(Deserialize)]
#[serde(tag = "by", rename_all = "snake_case")]
enum Named {
    Call(CallFacts),
    Model {
        model: String,
        provider: String,
        #[serde(default)]
        region: Option<String>,
        /// The provider the passed row is lowered under, when it is not `provider`
        #[serde(default)]
        row_provider: Option<String>,
    },
}

/// What `completion_cost` knows about the call before it picks a row
#[derive(Default, Deserialize)]
#[serde(default)]
struct CallFacts {
    provider: String,
    request_model: Option<String>,
    response_model: Option<String>,
    base_model: Option<String>,
    custom_pricing: bool,
    deployment_id: Option<String>,
    region: Option<String>,
    invoked_model_id: Option<String>,
    provider_response_model: Option<String>,
    hidden_model: Option<String>,
}

#[derive(Deserialize)]
struct PriceDocument {
    named: Named,
    usage: TokenUsage,
    #[serde(default)]
    requested_model: Option<Box<str>>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    data_residency: Option<String>,
    #[serde(default)]
    vertex_location: Option<Box<str>>,
    at: jiff::Timestamp,
    /// The discount and margin key, absent where main applies neither (`cost_per_token`)
    #[serde(default)]
    adjust: Option<AdjustFacts>,
    #[serde(default)]
    rules: TokenRules,
    #[serde(default)]
    elapsed: Option<Elapsed>,
    #[serde(default)]
    characters: Option<CharacterCounts>,
    #[serde(default)]
    embedding: bool,
    #[serde(default)]
    raw: Option<RawCounts>,
    #[serde(default)]
    reported: Option<ReportedCost>,
    #[serde(default)]
    custom: Option<CustomPricing>,
    #[serde(default)]
    decisions: bool,
    #[serde(default)]
    breakdown: BreakdownFacts,
    #[serde(default)]
    tool_fees: f64,
    #[serde(default)]
    router_fee: Option<RouterFeeFacts>,
}

/// Which row the stored breakdown reads. `row` is the `breakdown_row` passed beside the document
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BreakdownFacts {
    #[default]
    Priced,
    Named,
    Row,
    Absent,
}

/// The Azure AI Model Router fee, billed off the `fee_row` passed beside the document
#[derive(Deserialize)]
struct RouterFeeFacts {
    billed: FeeBilling,
}

/// The rows Python looked up for one token call beside the document
pub(crate) struct CallRows {
    pub(crate) row: Option<Fields>,
    pub(crate) breakdown: Option<Fields>,
    pub(crate) fee: Option<Fields>,
}

/// `custom_cost_per_token` or `custom_cost_per_second`, which price the call without a row
#[derive(Deserialize)]
struct CustomPricing {
    #[serde(default)]
    per_token: Option<FlatRates>,
    #[serde(default)]
    per_second: Option<UnitPrice>,
}

#[derive(Deserialize)]
struct AdjustFacts {
    #[serde(default)]
    provider: Option<Box<str>>,
    #[serde(default)]
    model: Option<Box<str>>,
    #[serde(default)]
    region: Option<Box<str>>,
}

#[derive(Deserialize)]
struct BoundDocument {
    models: Vec<BoundModelFacts>,
    request: RequestCaps,
}

#[derive(Deserialize)]
struct BoundModelFacts {
    #[serde(default)]
    counted_input: Option<i64>,
    shapes: Vec<ShapeFacts>,
}

/// One cost shape: the flat rates of a model info row or the router's group rates, beside a tier table. A row is the
/// index of a mapping passed beside the document
#[derive(Deserialize)]
struct ShapeFacts {
    flat: FlatFacts,
    #[serde(default)]
    tiers: Option<usize>,
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
pub(crate) fn price(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    row: Option<&Bound<'_, PyMapping>>,
    breakdown_row: Option<&Bound<'_, PyMapping>>,
    fee_row: Option<&Bound<'_, PyMapping>>,
) -> PyResult<Option<NativeReceipt>> {
    let document: PriceDocument = from_py_argument(document)?;
    for length in [
        document.usage.input_audio_seconds,
        document.usage.input_video_seconds,
    ] {
        super::seconds(length)?;
    }
    let rows = CallRows {
        row: row.map(row_fields).transpose()?,
        breakdown: breakdown_row.map(row_fields).transpose()?,
        fee: fee_row.map(row_fields).transpose()?,
    };
    priced(snapshot, document, rows)
        .map(|receipt| NativeReceipt::new(py, &receipt))
        .transpose()
}

/// The Messages route's pricing over the snapshot taken when the call was set up. It reads the call the way
/// `completion_cost` reads a native `/v1/messages` result on main: the route's provider and model, the model the
/// response names and the deployment facts the proxy sent, discount and margin applied, no row of its own
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
        let identity = &call.identity;
        priced(
            &self.snapshot,
            PriceDocument {
                named: Named::Call(CallFacts {
                    provider: call.provider.clone(),
                    request_model: Some(call.request_model.clone()),
                    response_model: call.response_model.clone(),
                    base_model: identity.base_model.clone(),
                    deployment_id: identity.deployment_id.clone(),
                    region: identity.region.clone(),
                    ..CallFacts::default()
                }),
                usage: call.usage.clone(),
                requested_model: Some(call.request_model.as_str().into()),
                service_tier: call.service_tier.clone(),
                data_residency: identity.data_residency.clone(),
                vertex_location: identity.vertex_location.as_deref().map(Into::into),
                at: jiff::Timestamp::now(),
                adjust: Some(AdjustFacts {
                    provider: Some(call.provider.as_str().into()),
                    model: None,
                    region: identity.region.as_deref().map(Into::into),
                }),
                rules: TokenRules::Branch,
                elapsed: None,
                characters: None,
                embedding: false,
                raw: None,
                reported: None,
                custom: None,
                decisions: false,
                breakdown: BreakdownFacts::Priced,
                tool_fees: 0.0,
                router_fee: None,
            },
            CallRows {
                row: None,
                breakdown: None,
                fee: None,
            },
        )
    }
}

/// The receipt of a token call, `None` when the core leaves it to Python
fn priced(snapshot: &CostSnapshot, document: PriceDocument, given: CallRows) -> Option<Receipt> {
    let CallRows {
        row,
        breakdown: breakdown_row,
        fee: fee_row,
    } = given;
    let (candidates, named) = match (&document.custom, &document.named) {
        (
            Some(_),
            Named::Model {
                model, provider, ..
            },
        ) => (
            vec![Candidate {
                model: model.as_str().into(),
                provider: provider.parse().ok()?,
            }],
            Vec::new(),
        ),
        _ => candidates(snapshot, &document.named, row.is_some())?,
    };
    let rows = snapshot.pricing_rows(named, []);
    let row_provider = match &document.named {
        Named::Model {
            row_provider: Some(provider),
            ..
        } => Some(provider.clone()),
        Named::Model { .. } | Named::Call(_) => None,
    };
    let custom = document.custom.as_ref().and_then(|custom| match custom {
        CustomPricing {
            per_token: Some(rates),
            ..
        } => Some(PriceSource::FlatPerToken(*rates)),
        CustomPricing {
            per_second: Some(rate),
            ..
        } => Some(PriceSource::PerSecond { rate: *rate }),
        CustomPricing { .. } => None,
    });
    let (rows, source) = match (custom, row, candidates.first()) {
        (Some(custom), _, _) => (rows, custom),
        (None, Some(row), Some(first)) => {
            let provider = row_provider.unwrap_or_else(|| first.provider.to_string());
            let (rows, at) = rows.with_per_call(PER_CALL_ROW, row, &provider);
            (rows, PriceSource::Row { row: at })
        }
        _ => (rows, PriceSource::Catalog),
    };
    let lowered_under = candidates
        .first()
        .map(|first| first.provider.to_string())
        .unwrap_or_default();
    let (rows, breakdown) = match (document.breakdown, breakdown_row) {
        (BreakdownFacts::Priced, _) => (rows, BreakdownRow::Priced),
        (BreakdownFacts::Named, _) => (rows, BreakdownRow::Named),
        (BreakdownFacts::Absent, _) | (BreakdownFacts::Row, None) => (rows, BreakdownRow::Absent),
        (BreakdownFacts::Row, Some(fields)) => {
            let (rows, at) = rows.with_per_call(BREAKDOWN_ROW, fields, &lowered_under);
            (rows, BreakdownRow::Row { row: at })
        }
    };
    let (rows, router_fee) = match (document.router_fee, fee_row) {
        (Some(RouterFeeFacts { billed }), Some(fields)) => {
            let (rows, at) = rows.with_per_call(ROUTER_FEE_ROW, fields, "azure_ai");
            (rows, Some(RouterFee { row: at, billed }))
        }
        (_, _) => (rows, None),
    };
    let query = PriceQuery::Tokens(Box::new(TokensQuery {
        header: Header {
            candidates,
            source,
            basis: Basis {
                service_tier: service_tier_named(document.service_tier.as_deref()),
                data_residency: data_residency(document.data_residency.as_deref()),
                vertex_location: document.vertex_location,
                at: document.at,
            },
            adjustments: document.adjust.map_or(AdjustmentPolicy::None, |key| {
                AdjustmentPolicy::Apply {
                    provider: key.provider,
                    model: key.model,
                    region: key.region,
                }
            }),
            reported: document.reported,
        },
        usage: document.usage,
        requested_model: document.requested_model,
        rules: document.rules,
        elapsed: document.elapsed,
        characters: document.characters,
        embedding: document.embedding,
        raw: document.raw.unwrap_or_default(),
        decisions: document.decisions,
        breakdown,
        tool_fees: document.tool_fees,
        router_fee,
    }));
    match litellm_cost::price(&rows, &query) {
        PriceOutcome::Priced(receipt) => Some(*receipt),
        PriceOutcome::Unpriced(_) => None,
    }
}

/// The candidates in main's order, and each one the catalog resolved beside its row
type CallCandidates = (Vec<Candidate>, Vec<(Candidate, RowRef)>);

/// The candidates in main's order and the rows the catalog resolved them to. `None` when main prices the call by a
/// path the catalog does not resolve, no row comes before it and the caller passed no row of its own
fn candidates(
    snapshot: &CostSnapshot,
    named: &Named,
    per_call_row: bool,
) -> Option<CallCandidates> {
    match named {
        Named::Call(call) => {
            let provider: Provider = call.provider.parse().ok()?;
            let found = snapshot.candidates(&CallIdentity {
                provider: call.provider.clone(),
                request_model: call.request_model.clone(),
                response_model: call.response_model.clone(),
                base_model: call.base_model.clone(),
                custom_pricing: call.custom_pricing,
                deployment_id: call.deployment_id.clone(),
                region: call.region.clone(),
                invoked_model_id: call.invoked_model_id.clone(),
                provider_response_model: call.provider_response_model.clone(),
                hidden_model: call.hidden_model.clone(),
            });
            let named: Vec<(Candidate, RowRef)> = found
                .rows
                .into_iter()
                .map(|Tried { model, row }| {
                    let candidate = Candidate {
                        model: model.into(),
                        provider: provider.clone(),
                    };
                    (candidate, row)
                })
                .collect();
            (!named.is_empty()).then(|| {
                let candidates = named
                    .iter()
                    .map(|(candidate, _)| candidate.clone())
                    .collect();
                (candidates, named)
            })
        }
        Named::Model {
            model,
            provider,
            region,
            ..
        } => {
            let row = match snapshot.priced_row(model, provider, region.as_deref()) {
                Ok(row) => row,
                Err(Deferral::Provider) => None,
                Err(_) if per_call_row => None,
                Err(_) => return None,
            };
            let candidate = Candidate {
                model: model.as_str().into(),
                provider: provider.parse().ok()?,
            };
            Some((
                vec![candidate.clone()],
                row.map(|row| (candidate, row)).into_iter().collect(),
            ))
        }
    }
}

/// `data_residency` lowercased, as the uplift helpers compare it
pub(super) fn data_residency(written: Option<&str>) -> Option<DataResidency> {
    match written?.to_lowercase().as_str() {
        "eu" => Some(DataResidency::Eu),
        "us" => Some(DataResidency::Us),
        "fedramp" => Some(DataResidency::Fedramp),
        _ => None,
    }
}

/// The legacy budget hold and input floor, `(None, None)` when no shape prices. `rows` are the model info mappings
/// the shapes read, as main read them, so they are priced for this call alone
pub(crate) fn bound(
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<(Option<f64>, Option<f64>)> {
    let document: BoundDocument = from_py_argument(document)?;
    let (source, refs) = rows.iter().enumerate().try_fold(
        (snapshot.pricing_rows([], []), Vec::new()),
        |(source, refs), (index, row)| -> PyResult<_> {
            let fields = row_fields(row)?;
            let provider = fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let (source, at) = source.with_per_call(&format!("bound-{index}"), fields, &provider);
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
pub(crate) fn rates(snapshot: &CostSnapshot, key: &str) -> Option<NativeRateCard> {
    let row = snapshot.row_at(key)?;
    let at = priced_ref(&row);
    let rows: PricingRows = snapshot.pricing_rows([], [row]);
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
    snapshot: &CostSnapshot,
    cost: f64,
    provider: Option<&str>,
    model: Option<&str>,
    region: Option<&str>,
) -> (f64, f64, f64, f64, f64, f64) {
    let applied = adjusted(
        snapshot.discounts(),
        snapshot.margins(),
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
    #[case::eu(Some("EU"), Some(DataResidency::Eu))]
    #[case::us(Some("us"), Some(DataResidency::Us))]
    #[case::fedramp(Some("FedRAMP"), Some(DataResidency::Fedramp))]
    #[case::unknown(Some("apac"), None)]
    #[case::none(None, None)]
    fn data_residency_is_read_case_blind(
        #[case] written: Option<&str>,
        #[case] expected: Option<DataResidency>,
    ) {
        assert_eq!(data_residency(written), expected);
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
