//! What each provider branch of main's `cost_per_token` adds on top of `generic_cost_per_token`: which uplifts the
//! total carries and how the web search tool is billed. Anthropic multiplies speed then geo
//! (`llms/anthropic/cost_calculation.py`), Bedrock multiplies geo when it looked the row up by name
//! (`llms/bedrock/cost_calculation.py`), Vertex multiplies its regional endpoint uplift, Azure AI nothing

use std::num::NonZeroU64;

use crate::call::{CallContext, InferenceGeo};
use crate::money::Usd;
use crate::price::{
    Band, Channel, InputUnit, Meter, OutputUnit, SearchContextSize, ServiceTier, Uplift, WallClock,
};
use crate::priced::Quantity;
use crate::query::{Basis, Candidate, Kind, Provider, TokenUsage, TokensQuery};
use crate::quote::Quote;
use crate::row::{LoweredRow, Provenance, ProviderEntry, RowRef};

/// How the row was reached: by the candidate's name, or handed in by the caller
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lookup {
    Catalog,
    Given { breakdown_by_name: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    Anthropic,
    /// `geo` is false when main priced a registered or caller-given row through `generic_cost_per_token`
    Bedrock {
        geo: bool,
    },
    Vertex,
    AzureAi,
}

/// `vertex_ai/cost_calculator.py` `cost_router`: the models Vertex prices by token instead of by character
const VERTEX_TOKEN_PRICED: [&str; 7] = [
    "claude",
    "llama",
    "mistral",
    "jamba",
    "codestral",
    "gemma",
    "gemini-2",
];

const OVER_128K: Band = Band::OverPromptTokens(NonZeroU64::new(128_000).unwrap());

impl Rule {
    /// `Err` names the kind the caller has to price the old way
    pub(super) fn new(
        candidate: &Candidate,
        at: &RowRef,
        row: &impl LoweredRow,
        lookup: Lookup,
        query: &TokensQuery,
    ) -> Result<Self, Kind> {
        if bills_seconds(row) {
            return Err(Kind::Tokens);
        }
        let model = &*candidate.model;
        let web_search = query.usage.web_search.is_some();
        match &candidate.provider {
            Provider::Anthropic => Ok(Self::Anthropic),
            Provider::Bedrock => Ok(Self::Bedrock {
                geo: lookup == Lookup::Catalog && at.provenance != Provenance::Registered,
            }),
            Provider::VertexAi
                if !VERTEX_TOKEN_PRICED.iter().any(|name| model.contains(name))
                    || prices_above_128k(row)
                    || (web_search && !model.to_lowercase().contains("claude")) =>
            {
                Err(Kind::Tokens)
            }
            Provider::VertexAi => Ok(Self::Vertex),
            Provider::AzureAi
                if is_model_router(model)
                    || query
                        .requested_model
                        .as_deref()
                        .is_some_and(is_model_router) =>
            {
                Err(Kind::Tokens)
            }
            Provider::AzureAi => Ok(Self::AzureAi),
            Provider::Other(_) => Err(Kind::Tokens),
        }
    }

    /// The call facts the total is priced with. `None` where main raises reading the geo multiplier
    pub(super) fn call(
        self,
        basis: &Basis,
        usage: &TokenUsage,
        entry: &ProviderEntry,
    ) -> Option<CallContext> {
        let geo = billed_geo(usage);
        let unreadable =
            |uplift: Option<Uplift>| uplift.is_some_and(|uplift| entry.rejects(&uplift));
        let geo_raises = unreadable(geo.clone().map(Uplift::InferenceGeo));
        let speed_raises = unreadable(usage.speed.map(Uplift::Speed));
        let base = CallContext {
            tier: basis.service_tier,
            channel: Channel::Online,
            residency: None,
            fedramp_endpoint: false,
            non_global_endpoint: false,
            speed: None,
            inference_geo: None,
            at: basis.at,
            overlay: None,
        };
        match self {
            Self::Anthropic if geo_raises || speed_raises => Some(base),
            Self::Anthropic => Some(CallContext {
                speed: usage.speed,
                inference_geo: geo,
                ..base
            }),
            Self::Bedrock { geo: true } if geo_raises => None,
            Self::Bedrock { geo: true } => Some(CallContext {
                inference_geo: geo,
                ..base
            }),
            Self::Bedrock { geo: false } | Self::AzureAi => Some(base),
            Self::Vertex => Some(CallContext {
                non_global_endpoint: basis.non_global_endpoint(),
                ..base
            }),
        }
    }

    /// The web search tool fee. Anthropic, Vertex and Azure AI bill each reported request at the medium context
    /// price. Bedrock bills per call at the requested context size, and a response reporting zero requests still
    /// bills one. `Err` where main would read the price off the served model's row instead
    pub(super) fn web_search_fee(self, quote: &Quote<'_>, usage: &TokenUsage) -> Result<Usd, Kind> {
        let Some(used) = usage.web_search else {
            return Ok(Usd::ZERO);
        };
        let rate = |size| {
            quote
                .rate(&Meter::WebSearch(size))
                .map(|rate| rate.unit_price)
        };
        match self {
            Self::Bedrock { .. } => {
                let size = used.context.unwrap_or(SearchContextSize::Medium);
                let calls = used.requests.max(1);
                Ok(rate(size).map_or(Usd::ZERO, |rate| rate * Quantity::Count(calls)))
            }
            Self::Anthropic | Self::Vertex | Self::AzureAi => {
                let priced = [
                    SearchContextSize::Low,
                    SearchContextSize::Medium,
                    SearchContextSize::High,
                ]
                .into_iter()
                .any(|size| rate(size).is_some());
                if !priced {
                    return Err(Kind::Tokens);
                }
                Ok(rate(SearchContextSize::Medium)
                    .map_or(Usd::ZERO, |rate| rate * Quantity::Count(used.requests)))
            }
        }
    }
}

/// `get_provider_specific_geo_multiplier` reads no geo for a global or unknown one
pub(super) fn billed_geo(usage: &TokenUsage) -> Option<InferenceGeo> {
    usage
        .inference_geo
        .clone()
        .filter(|geo| !matches!(geo.as_str(), "global" | "not_available"))
}

/// `is_azure_model_router`
fn is_model_router(model: &str) -> bool {
    let model = model.to_lowercase();
    model.contains("model-router")
        || model.contains("model_router")
        || model == "azure-model-router"
}

/// `_per_second_pricing_cost`: a row without a positive token price or tiers, in a wall-clock mode, with a
/// per-second price bills response time, not tokens
fn bills_seconds(row: &impl LoweredRow) -> bool {
    let prices = row.prices();
    let flat = prices.flat();
    let positive = |meter: Meter| {
        flat.get(&crate::price::Price::online(meter, Default::default()))
            .is_some_and(|rate| rate.get() > 0.0)
    };
    let per_second = [
        Meter::WallClockSecond,
        Meter::Input(InputUnit::Second),
        Meter::Output(OutputUnit::Second(None)),
    ];
    !positive(Meter::TEXT_INPUT)
        && !positive(Meter::TEXT_OUTPUT)
        && prices.tiers().is_none()
        && flat.billing().wall_clock != WallClock::Never
        && per_second.into_iter().any(|meter| {
            flat.get(&crate::price::Price::online(meter, Default::default()))
                .is_some()
        })
}

/// Vertex's `_handle_128k_pricing` path, for rows with `_above_128k_tokens` token prices
fn prices_above_128k(row: &impl LoweredRow) -> bool {
    row.prices().flat().iter().any(|(price, _)| {
        (price.meter == Meter::TEXT_INPUT || price.meter == Meter::TEXT_OUTPUT)
            && price.channel == Channel::Online
            && price.when.tier == ServiceTier::Standard
            && price.when.band == OVER_128K
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::query::WebSearchUse;
    use crate::testing::{
        BASE, TestRow, call, candidate, over, prices, row_ref, text_row, tokens_query,
    };

    fn search(requests: u64, context: Option<SearchContextSize>) -> TokenUsage {
        TokenUsage {
            web_search: Some(WebSearchUse { requests, context }),
            ..TokenUsage::default()
        }
    }

    #[rstest]
    #[case::anthropic_medium_per_request(
        Rule::Anthropic,
        search(3, Some(SearchContextSize::High)),
        6.0
    )]
    #[case::anthropic_zero_requests(Rule::Anthropic, search(0, None), 0.0)]
    #[case::bedrock_requested_size(Rule::Bedrock { geo: true }, search(3, Some(SearchContextSize::High)), 15.0)]
    #[case::bedrock_defaults_to_medium(Rule::Bedrock { geo: true }, search(3, None), 6.0)]
    #[case::bedrock_bills_one_call_for_zero_requests(Rule::Bedrock { geo: false }, search(0, None), 2.0)]
    #[case::no_search(Rule::Anthropic, TokenUsage::default(), 0.0)]
    fn web_search_bills_by_provider(
        #[case] rule: Rule,
        #[case] usage: TokenUsage,
        #[case] expected: f64,
    ) {
        let prices = prices(&[
            (
                Meter::WebSearch(SearchContextSize::Medium),
                ServiceTier::Standard,
                BASE,
                2.0,
            ),
            (
                Meter::WebSearch(SearchContextSize::High),
                ServiceTier::Standard,
                BASE,
                5.0,
            ),
        ]);
        let quote = prices.quote(&call(), 0);
        assert_eq!(
            rule.web_search_fee(&quote, &usage).map(Usd::get),
            Ok(expected)
        );
    }

    #[rstest]
    fn a_search_price_main_would_read_off_the_served_model_is_left_to_python() {
        let prices = prices(&[(Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 1.0)]);
        let quote = prices.quote(&call(), 0);
        assert_eq!(
            Rule::Anthropic.web_search_fee(&quote, &search(1, None)),
            Err(Kind::Tokens)
        );
        assert_eq!(
            Rule::Bedrock { geo: true }
                .web_search_fee(&quote, &search(1, None))
                .map(Usd::get),
            Ok(0.0)
        );
    }

    #[rstest]
    #[case::global("global", None)]
    #[case::not_available("not_available", None)]
    #[case::upper("US", Some("us"))]
    fn a_global_geo_bills_no_geo_uplift(#[case] geo: &str, #[case] expected: Option<&str>) {
        let usage = TokenUsage {
            inference_geo: Some(InferenceGeo::new(geo)),
            ..TokenUsage::default()
        };
        assert_eq!(
            billed_geo(&usage).as_ref().map(InferenceGeo::as_str),
            expected
        );
    }

    #[rstest]
    #[case::router("azure_ai/model-router")]
    #[case::underscore("Model_Router/gpt-5")]
    #[case::named("azure-model-router")]
    fn model_router_names_are_detected(#[case] model: &str) {
        assert!(is_model_router(model));
    }

    fn rule_for(
        provider: Provider,
        model: &str,
        row: &TestRow,
        usage: TokenUsage,
    ) -> Result<Rule, Kind> {
        let query = tokens_query(provider.clone(), model, usage);
        Rule::new(
            &candidate(provider, model),
            &row_ref(model),
            row,
            Lookup::Catalog,
            &query,
        )
    }

    #[rstest]
    #[case::vertex_claude(
        Provider::VertexAi,
        "claude-opus-4-8",
        text_row(1.0, 1.0),
        Ok(Rule::Vertex)
    )]
    #[case::vertex_gemini_2(
        Provider::VertexAi,
        "gemini-2.5-pro",
        text_row(1.0, 1.0),
        Ok(Rule::Vertex)
    )]
    #[case::vertex_128k_pricing(
        Provider::VertexAi,
        "claude-opus-4-8",
        TestRow {
            prices: prices(&[(Meter::TEXT_OUTPUT, ServiceTier::Standard, over(128_000), 1.0)]),
            ..text_row(1.0, 1.0)
        },
        Err(Kind::Tokens)
    )]
    #[case::vertex_other_band(
        Provider::VertexAi,
        "claude-opus-4-8",
        TestRow {
            prices: prices(&[(Meter::TEXT_OUTPUT, ServiceTier::Standard, over(200_000), 1.0)]),
            ..text_row(1.0, 1.0)
        },
        Ok(Rule::Vertex)
    )]
    #[case::per_second_row(
        Provider::Anthropic,
        "claude",
        TestRow {
            prices: prices(&[(Meter::WallClockSecond, ServiceTier::Standard, BASE, 1.0)]),
            ..TestRow::default()
        },
        Err(Kind::Tokens)
    )]
    #[case::per_second_beside_token_prices(
        Provider::Anthropic,
        "claude",
        TestRow {
            prices: prices(&[
                (Meter::WallClockSecond, ServiceTier::Standard, BASE, 1.0),
                (Meter::TEXT_INPUT, ServiceTier::Standard, BASE, 1.0),
            ]),
            ..TestRow::default()
        },
        Ok(Rule::Anthropic)
    )]
    fn a_row_main_prices_another_way_is_left_to_python(
        #[case] provider: Provider,
        #[case] model: &str,
        #[case] row: TestRow,
        #[case] expected: Result<Rule, Kind>,
    ) {
        assert_eq!(
            rule_for(provider, model, &row, TokenUsage::default()),
            expected
        );
    }

    #[rstest]
    #[case::claude("claude-opus-4-8", Ok(Rule::Vertex))]
    #[case::gemini("gemini-2.5-pro", Err(Kind::Tokens))]
    fn vertex_web_search_is_ported_for_claude_only(
        #[case] model: &str,
        #[case] expected: Result<Rule, Kind>,
    ) {
        let row = text_row(1.0, 1.0);
        assert_eq!(
            rule_for(Provider::VertexAi, model, &row, search(1, None)),
            expected
        );
    }

    #[rstest]
    fn an_azure_ai_call_the_client_sent_to_the_model_router_is_left_to_python() {
        let row = text_row(1.0, 1.0);
        let query = TokensQuery {
            requested_model: Some("azure_ai/model-router".into()),
            ..tokens_query(Provider::AzureAi, "gpt-5", TokenUsage::default())
        };
        let rule = Rule::new(
            &candidate(Provider::AzureAi, "gpt-5"),
            &row_ref("gpt-5"),
            &row,
            Lookup::Catalog,
            &query,
        );
        assert_eq!(rule, Err(Kind::Tokens));
    }
}
