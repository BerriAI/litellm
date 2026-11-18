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
use litellm_cost::receipt::{PriceOutcome, Receipt, Unpriced};
use litellm_cost::reported::{PricedSides, ReportedQuery};
use litellm_cost::tools::{ToolUse, Tools, ToolsQuery};
use litellm_host_python::{from_py_argument, json_from_py};
use litellm_inference_messages::PricedCall;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    AfterRows, CallIdentity, Deferral, HiddenParams, NameHost, PricingRows, TableOnly, Tried,
    TriedRow, priced_ref,
};
use litellm_python_compat::number::float_of;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde::Deserialize;
use serde_json::Value;

use super::service_tier_named;
use crate::model_cost::{CostSnapshot, PythonNames, row_fields};

/// The key a row the caller passed for one call is priced under
const PER_CALL_ROW: &str = "requested_model_info_override";

/// The key of the row the caller looked up for the stored breakdown
const BREAKDOWN_ROW: &str = "breakdown";

/// The key of the Azure AI Model Router row whose fee the call bills
const ROUTER_FEE_ROW: &str = "router-fee";

/// The key of a candidate's row `get_model_info` built rather than found in the table
const CANDIDATE_ROW: &str = "candidate";

/// The rows to try: the candidates main's `completion_cost` loop walks for a finished call, or the one model a
/// caller of `cost_per_token` names
#[derive(Deserialize)]
#[serde(tag = "by", rename_all = "snake_case")]
enum Named {
    Call(Box<CallFacts>),
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
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct CallFacts {
    provider: Option<String>,
    request_model: Option<String>,
    response_model: Option<String>,
    base_model: Option<String>,
    custom_pricing: bool,
    deployment_id: Option<String>,
    region: Option<String>,
    invoked_model_id: Option<String>,
    hidden: Option<HiddenFacts>,
    loop_provider: Option<String>,
    loop_region: Option<String>,
}

/// The `_hidden_params` strings that pick a candidate
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
struct HiddenFacts {
    model: Option<String>,
    litellm_model_name: Option<String>,
    provider_response_model: Option<String>,
    region: Option<String>,
}

impl From<CallFacts> for CallIdentity {
    fn from(call: CallFacts) -> Self {
        Self {
            provider: call.provider,
            request_model: call.request_model,
            response_model: call.response_model,
            base_model: call.base_model,
            custom_pricing: call.custom_pricing,
            deployment_id: call.deployment_id,
            region: call.region,
            invoked_model_id: call.invoked_model_id,
            hidden: call.hidden.map(|hidden| HiddenParams {
                model: hidden.model,
                litellm_model_name: hidden.litellm_model_name,
                provider_response_model: hidden.provider_response_model,
                region: hidden.region,
            }),
            loop_provider: call.loop_provider,
            loop_region: call.loop_region,
        }
    }
}

/// What Python found for one candidate of a call: the built-in tools it used and the Model Router fee it bills, each
/// row named by its index among the candidate's own rows, which start at `rows_from` among the rows passed beside
/// the document, or that finding them raised, as main moved on to the next candidate when they did
#[derive(Clone, Default, Deserialize)]
#[serde(default)]
struct CandidateFacts {
    candidate: usize,
    tools: Option<ToolUse<usize>>,
    router_fee: Option<CandidateFee>,
    failed: bool,
    rows_from: usize,
}

/// One usage a call bills on its own: one service tier's share of a Responses WebSocket session, each priced as main
/// priced it, by its own walk of the call's candidates
#[derive(Deserialize)]
struct UsagePart {
    usage: TokenUsage,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    raw: Option<RawCounts>,
    #[serde(default)]
    candidates: Vec<CandidateFacts>,
}

/// The usage one walk prices and what Python found for each candidate
struct PartView<'a> {
    usage: &'a TokenUsage,
    service_tier: Option<&'a str>,
    raw: Option<RawCounts>,
    candidates: &'a [CandidateFacts],
}

#[derive(Clone, Copy, Deserialize)]
struct CandidateFee {
    billed: FeeBilling,
    row: usize,
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
    /// The built-in tools the call used, each row named by its index among the tool rows passed beside it
    #[serde(default)]
    tools: Option<ToolUse<usize>>,
    #[serde(default)]
    router_fee: Option<RouterFeeFacts>,
    /// What Python found for each candidate of a call, by its place among the candidates
    #[serde(default)]
    candidates: Vec<CandidateFacts>,
    /// The usages a call bills each on its own, in place of `usage`
    #[serde(default)]
    parts: Vec<UsagePart>,
}

impl PriceDocument {
    fn whole(&self) -> PartView<'_> {
        PartView {
            usage: &self.usage,
            service_tier: self.service_tier.as_deref(),
            raw: self.raw,
            candidates: &self.candidates,
        }
    }
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
    pub(crate) tools: Vec<Fields>,
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
    input_cost_per_character: Option<f64>,
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
    tool_rows: &[Bound<'_, PyMapping>],
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
        tools: tool_rows.iter().map(row_fields).collect::<PyResult<_>>()?,
    };
    let host = PythonNames { py };
    let receipt = match &document.named {
        Named::Call(call) => priced_call(
            snapshot,
            &(**call).clone().into(),
            &document,
            &document.whole(),
            &rows.tools,
            &host,
        ),
        Named::Model { .. } => priced(snapshot, document, rows, &host),
    };
    receipt
        .map(|receipt| NativeReceipt::new(py, &receipt))
        .transpose()
}

/// The candidates of the call in `document` as `completion_cost` reads them
pub(crate) struct CallCandidateNames {
    /// The name `select_model_name_for_cost_calc` picks with the provider the call names
    pub(crate) selected: Option<String>,
    /// The name `completion_cost` selects, with the provider it overrides the call's with
    pub(crate) call_selected: Option<String>,
    /// The names main's loop walks, each with the provider it pairs it with
    pub(crate) names: Vec<(Option<String>, Option<String>)>,
    /// The Bedrock row every Bedrock candidate prices with, the row a `register_model` price landed on and whether
    /// main merges that registration over the row
    pub(crate) bedrock: Option<(String, Option<String>, bool)>,
}

/// The names `completion_cost` walks for the call in `document`, the name `select_model_name_for_cost_calc` picks and
/// the Bedrock row every Bedrock candidate prices with
pub(crate) fn cost_candidates(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
) -> PyResult<CallCandidateNames> {
    let call: CallIdentity = from_py_argument::<CallFacts>(document)?.into();
    let host = PythonNames { py };
    let found = snapshot.cost_candidates(&call, &host);
    Ok(CallCandidateNames {
        selected: snapshot.selected_cost_model(&call, &host),
        call_selected: found.selected,
        names: found
            .names
            .into_iter()
            .map(|candidate| (candidate.model, candidate.provider))
            .collect(),
        bedrock: snapshot
            .bedrock_pricing(&call, &host)
            .map(|pricing| (pricing.key, pricing.registration, pricing.merged)),
    })
}

/// The candidates main's loop prices the call in `document` with, in its order: each candidate's place among the
/// names, the model name and provider it is priced as and the provider the loop holds. The flag is whether the walk
/// stops at a candidate only Python prices
pub(crate) fn cost_rows(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
) -> PyResult<CostRows> {
    let call: CallFacts = from_py_argument(document)?;
    let found = snapshot.candidates(&call.into(), &PythonNames { py });
    Ok((
        found
            .rows
            .into_iter()
            .map(|tried| {
                (
                    tried.candidate,
                    tried.model,
                    tried.provider,
                    tried.loop_provider,
                )
            })
            .collect(),
        matches!(found.then, AfterRows::Deferred(_)),
    ))
}

/// Each candidate Model Catalog prices a call with (its place, model, provider and the provider main's loop holds),
/// and whether the walk stops at a candidate only Python prices
pub(crate) type CostRows = (Vec<(usize, String, String, String)>, bool);

/// The receipt of each usage part of the call in `document`, each walked and priced on its own, `None` when any part
/// is left to Python
pub(crate) fn price_parts(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<Option<Vec<NativeReceipt>>> {
    let document: PriceDocument = from_py_argument(document)?;
    let Named::Call(call) = &document.named else {
        return Ok(None);
    };
    let call: CallIdentity = (**call).clone().into();
    let given: Vec<Fields> = rows.iter().map(row_fields).collect::<PyResult<_>>()?;
    let host = PythonNames { py };
    let receipts: Option<Vec<Receipt>> = document
        .parts
        .iter()
        .map(|part| {
            let view = PartView {
                usage: &part.usage,
                service_tier: part.service_tier.as_deref(),
                raw: part.raw,
                candidates: &part.candidates,
            };
            priced_call(snapshot, &call, &document, &view, &given, &host)
        })
        .collect();
    receipts
        .map(|receipts| {
            receipts
                .iter()
                .map(|receipt| NativeReceipt::new(py, receipt))
                .collect()
        })
        .transpose()
}

/// The receipt of a call's first candidate that prices, as main's loop walks them. `None` when a candidate only
/// Python prices comes first, or none prices, so Python walks the candidates itself
fn priced_call(
    snapshot: &CostSnapshot,
    call: &CallIdentity,
    document: &PriceDocument,
    part: &PartView<'_>,
    given: &[Fields],
    host: &dyn NameHost,
) -> Option<Receipt> {
    let found = snapshot.candidates(call, host);
    let walked = found.rows.iter().find_map(|tried| {
        let facts = part
            .candidates
            .iter()
            .find(|facts| facts.candidate == tried.candidate);
        if facts.is_some_and(|facts| facts.failed) {
            return None;
        }
        match price_tried(snapshot, document, part, tried, facts, given) {
            PriceOutcome::Priced(receipt) => Some(Some(*receipt)),
            PriceOutcome::Unpriced(Unpriced::NotPorted(_) | Unpriced::CatalogDeferred) => {
                Some(None)
            }
            PriceOutcome::Unpriced(_) => None,
        }
    });
    walked.flatten()
}

/// One candidate of a call priced on its own row, with its own tools, fee and breakdown row
fn price_tried(
    snapshot: &CostSnapshot,
    document: &PriceDocument,
    part: &PartView<'_>,
    tried: &Tried,
    facts: Option<&CandidateFacts>,
    given: &[Fields],
) -> PriceOutcome {
    let Ok(provider) = tried.provider.parse::<Provider>();
    let candidate = Candidate {
        model: tried.model.as_str().into(),
        provider,
    };
    let breakdown_ref = match (document.breakdown, &tried.breakdown) {
        (BreakdownFacts::Row, Some(TriedRow::Table(row))) => Some(row.clone()),
        _ => None,
    };
    let rows = snapshot.pricing_rows([], breakdown_ref.clone());
    let rows = match &tried.row {
        TriedRow::Table(row) => {
            snapshot.pricing_rows([(candidate.clone(), row.clone())], breakdown_ref.clone())
        }
        TriedRow::Fields { fields, .. } => rows.with_named_per_call(
            candidate.clone(),
            CANDIDATE_ROW,
            (**fields).clone(),
            &tried.provider,
        ),
    };
    let (rows, breakdown) = match (document.breakdown, &tried.breakdown) {
        (BreakdownFacts::Priced, _) => (rows, BreakdownRow::Priced),
        (BreakdownFacts::Named, _) => (rows, BreakdownRow::Named),
        (BreakdownFacts::Absent, _) | (BreakdownFacts::Row, None) => (rows, BreakdownRow::Absent),
        (BreakdownFacts::Row, Some(TriedRow::Table(row))) => (
            rows,
            BreakdownRow::Row {
                row: priced_ref(row),
            },
        ),
        (BreakdownFacts::Row, Some(TriedRow::Fields { fields, .. })) => {
            let (rows, at) = rows.with_per_call(BREAKDOWN_ROW, (**fields).clone(), &tried.provider);
            (rows, BreakdownRow::Row { row: at })
        }
    };
    let rows_from = facts.map_or(0, |facts| facts.rows_from);
    let fee = facts.and_then(|facts| facts.router_fee);
    let (rows, router_fee) =
        match fee.and_then(|fee| Some((fee.billed, given.get(rows_from + fee.row)?.clone()))) {
            Some((billed, fields)) => {
                let (rows, at) = rows.with_per_call(ROUTER_FEE_ROW, fields, "azure_ai");
                (rows, Some(RouterFee { row: at, billed }))
            }
            None => (rows, None),
        };
    let (rows, tools) = with_tool_rows(rows, given.to_vec());
    let query = PriceQuery::Tokens(Box::new(TokensQuery {
        header: Header {
            candidates: vec![candidate],
            source: PriceSource::Catalog,
            basis: Basis {
                service_tier: service_tier_named(part.service_tier),
                data_residency: data_residency(document.data_residency.as_deref()),
                vertex_location: document.vertex_location.clone(),
                at: document.at,
            },
            adjustments: document
                .adjust
                .as_ref()
                .map_or(AdjustmentPolicy::None, |key| AdjustmentPolicy::Apply {
                    provider: Some(tried.loop_provider.as_str().into()),
                    model: Some(
                        key.model
                            .clone()
                            .unwrap_or_else(|| tried.model.as_str().into()),
                    ),
                    region: key.region.clone(),
                }),
            reported: None,
        },
        usage: part.usage.clone(),
        requested_model: document.requested_model.clone(),
        rules: document.rules,
        elapsed: document.elapsed,
        characters: document.characters,
        embedding: document.embedding,
        raw: part.raw.unwrap_or_default(),
        decisions: document.decisions,
        breakdown,
        tools: facts
            .and_then(|facts| facts.tools.clone())
            .unwrap_or_default()
            .rows(&|index: usize| tools.get(rows_from + index).cloned()),
        router_fee,
    }));
    litellm_cost::price(&rows, &query)
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
        let document = PriceDocument {
            named: Named::Call(Box::default()),
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
            tools: None,
            router_fee: None,
            candidates: Vec::new(),
            parts: Vec::new(),
        };
        priced_call(
            &self.snapshot,
            &CallIdentity {
                provider: Some(call.provider.clone()),
                request_model: Some(call.request_model.clone()),
                response_model: call.response_model.clone(),
                base_model: identity.base_model.clone(),
                deployment_id: identity.deployment_id.clone(),
                region: identity.region.clone(),
                loop_provider: Some(call.provider.clone()),
                loop_region: identity.region.clone(),
                ..CallIdentity::default()
            },
            &document,
            &document.whole(),
            &[],
            &TableOnly,
        )
    }
}

/// The receipt of a token call a caller names one model for, `None` when the core leaves it to Python
fn priced(
    snapshot: &CostSnapshot,
    document: PriceDocument,
    given: CallRows,
    host: &dyn NameHost,
) -> Option<Receipt> {
    let CallRows {
        row,
        breakdown: breakdown_row,
        fee: fee_row,
        tools: tool_rows,
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
            None,
        ),
        _ => candidates(snapshot, &document.named, row.is_some(), host)?,
    };
    let rows = match named {
        Some((candidate, TriedRow::Table(row))) => snapshot.pricing_rows([(candidate, row)], []),
        Some((candidate, TriedRow::Fields { fields, .. })) => {
            snapshot.pricing_rows([], []).with_named_per_call(
                candidate.clone(),
                CANDIDATE_ROW,
                (*fields).clone(),
                &candidate.provider.to_string(),
            )
        }
        None => snapshot.pricing_rows([], []),
    };
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
    let (rows, tools) = with_tool_rows(rows, tool_rows);
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
        tools: document
            .tools
            .unwrap_or_default()
            .rows(&|index: usize| tools.get(index).cloned()),
        router_fee,
    }));
    match litellm_cost::price(&rows, &query) {
        PriceOutcome::Priced(receipt) => Some(*receipt),
        PriceOutcome::Unpriced(_) => None,
    }
}

/// `rows` added for this call alone, each lowered under the provider it names, and where each landed
fn with_tool_rows(
    rows: PricingRows,
    tool_rows: Vec<Fields>,
) -> (PricingRows, Vec<litellm_cost::row::RowRef>) {
    tool_rows
        .into_iter()
        .enumerate()
        .fold((rows, Vec::new()), |(rows, refs), (index, fields)| {
            let provider = fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let (rows, at) = rows.with_per_call(&format!("tool-{index}"), fields, &provider);
            (rows, refs.into_iter().chain([at]).collect())
        })
}

/// Prices built-in tool fees alone: every fee a call's tools bill, or one tool's calculator. `None` where main's
/// calculator returned `None`. `rows` are the rows the document names by index
pub(crate) fn price_tools(
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
    rows: &[Bound<'_, PyMapping>],
) -> PyResult<Option<f64>> {
    let tools: Tools<usize> = from_py_argument(document)?;
    let fields = rows.iter().map(row_fields).collect::<PyResult<_>>()?;
    let (source, refs) = with_tool_rows(snapshot.pricing_rows([], []), fields);
    let query = PriceQuery::Tools(Box::new(ToolsQuery {
        header: Header {
            candidates: Vec::new(),
            source: PriceSource::Rows { rows: Vec::new() },
            basis: Basis {
                service_tier: Default::default(),
                data_residency: None,
                vertex_location: None,
                at: jiff::Timestamp::now(),
            },
            adjustments: AdjustmentPolicy::None,
            reported: None,
        },
        tools: tools.rows(&|index: usize| refs.get(index).cloned()),
    }));
    Ok(match litellm_cost::price(&source, &query) {
        PriceOutcome::Priced(receipt) => Some(receipt.tools.get()),
        PriceOutcome::Unpriced(_) => None,
    })
}

/// A reported cost and what the catalog would charge the same usage on each side
#[derive(Deserialize)]
struct ReportedDocument {
    reported: ReportedCost,
    #[serde(default)]
    split_by: Option<PricedSides>,
    #[serde(default)]
    adjust: Option<AdjustFacts>,
}

/// Bills a cost a provider or the gateway in front of it reported, by the policy of the source that stated it
pub(crate) fn price_reported(
    py: Python<'_>,
    snapshot: &CostSnapshot,
    document: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let document: ReportedDocument = from_py_argument(document)?;
    let query = PriceQuery::Reported(Box::new(ReportedQuery {
        header: Header {
            candidates: Vec::new(),
            source: PriceSource::Rows { rows: Vec::new() },
            basis: Basis {
                service_tier: Default::default(),
                data_residency: None,
                vertex_location: None,
                at: jiff::Timestamp::now(),
            },
            adjustments: document.adjust.map_or(AdjustmentPolicy::None, |key| {
                AdjustmentPolicy::Apply {
                    provider: key.provider,
                    model: key.model,
                    region: key.region,
                }
            }),
            reported: None,
        },
        reported: document.reported,
        split_by: document.split_by,
    }));
    super::kinds::outcome(
        py,
        litellm_cost::price(&snapshot.pricing_rows([], []), &query),
    )
}

/// The one candidate a caller names, and the row the catalog resolved it to
type CallCandidates = (Vec<Candidate>, Option<(Candidate, TriedRow)>);

/// The candidate a `cost_per_token` caller names and the row the catalog resolved it to. `None` when main prices it
/// by a path the catalog does not resolve and the caller passed no row of its own
fn candidates(
    snapshot: &CostSnapshot,
    named: &Named,
    per_call_row: bool,
    host: &dyn NameHost,
) -> Option<CallCandidates> {
    let Named::Model {
        model,
        provider,
        region,
        ..
    } = named
    else {
        return None;
    };
    let priced = match snapshot.priced_row(model, provider, region.as_deref(), host) {
        Ok(priced) => priced,
        Err(Deferral::Provider) => None,
        Err(_) if per_call_row => None,
        Err(_) => return None,
    };
    let candidate = |model: &str| -> Option<Candidate> {
        Some(Candidate {
            model: model.into(),
            provider: provider.parse().ok()?,
        })
    };
    Some(match priced {
        Some(priced) => {
            let named = candidate(&priced.model)?;
            (vec![named.clone()], Some((named, priced.row)))
        }
        None => (vec![candidate(model)?], None),
    })
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
        input_cost_per_character: rate(Meter::Input(InputUnit::Character)),
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
