//! Model Catalog's side of `litellm-cost`'s pricing: a `CostLookup` is a `PricingTable`, handing the core each row
//! lowered for its billed provider, and a lowered row answers as the core's `LoweredRow` with the `RowFacts` read off
//! its fields. Which rows a call reads, and how they are billed, is litellm-cost's

use std::collections::BTreeSet;
use std::sync::Arc;

use litellm_cost::adjust::CostConfig;
use litellm_cost::call::{InferenceGeo, Speed};
use litellm_cost::candidates::{PricingTable, RowProvenance, RowRef};
use litellm_cost::price::{Adjustments, Uplift};
use litellm_cost::row::{
    Declared, LoweredRow as PricedRow, ProviderEntry, RowFacts, RowMode, RowPrices,
};
use litellm_python_compat::number::{float_of, int_of};
use litellm_python_compat::truthy::json_truthy;
use serde_json::Value;

use crate::lower::BilledPrices;
use crate::view::{CatalogView, CostLookup, LoweredRow};
use crate::wire::Fields;

impl<V: CatalogView + ?Sized> PricingTable for CostLookup<'_, V> {
    type Row = LoweredRow;

    fn lowered(&self, key: &str, provider: &str) -> Option<Arc<LoweredRow>> {
        self.table.lowered(key, provider)
    }

    fn row_at(&self, key: &str) -> Option<RowRef> {
        let fields = self.table.fields(key)?;
        Some(RowRef {
            key: key.to_owned(),
            provider: fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            provenance: RowProvenance::Catalog,
        })
    }

    fn config(&self) -> &Arc<CostConfig> {
        &self.context.config
    }
}

impl PricedRow for LoweredRow {
    fn prices(&self) -> RowPrices<'_> {
        match &self.prices {
            BilledPrices::Flat(prices) => RowPrices::Flat(prices),
            BilledPrices::Tiered {
                tiers,
                empty_prompt,
            } => RowPrices::Tiered {
                tiers,
                empty_prompt,
            },
        }
    }

    fn adjustments(&self) -> &Adjustments {
        &self.adjustments
    }

    fn facts(&self) -> &RowFacts {
        &self.facts
    }
}

/// What pricing reads off a row besides its prices, as main's `get_model_info` and the auth checks read it
pub(crate) fn row_facts(fields: &Fields) -> RowFacts {
    let count = |key: &str| {
        fields
            .get(key)
            .and_then(|value| int_of(value).ok())
            .and_then(|count| u64::try_from(count).ok())
    };
    RowFacts {
        mode: match fields.get("mode").and_then(Value::as_str) {
            None | Some("chat") => RowMode::Chat,
            Some("image_generation") => RowMode::ImageGeneration,
            Some("image_edit") => RowMode::ImageEdit,
            Some(_) => RowMode::Other,
        },
        max_input_tokens: count("max_input_tokens"),
        max_output_tokens: count("max_output_tokens"),
        provider_entry: provider_entry(fields.get("provider_specific_entry")),
        declared: declared(fields),
    }
}

/// `get_provider_specific_geo_multiplier` and `get_provider_specific_speed_multiplier` read the entry with `float()`
fn provider_entry(entry: Option<&Value>) -> ProviderEntry {
    match entry.filter(|entry| json_truthy(entry)) {
        None => ProviderEntry::Readable,
        Some(Value::Object(uplifts)) => {
            let unreadable: BTreeSet<Uplift> = uplifts
                .iter()
                .filter(|(_, factor)| float_of(factor).is_err())
                .map(|(key, _)| match key.as_str() {
                    "fast" => Uplift::Speed(Speed::Fast),
                    geo => Uplift::InferenceGeo(InferenceGeo::new(geo)),
                })
                .collect();
            match unreadable.is_empty() {
                true => ProviderEntry::Readable,
                false => ProviderEntry::Unreadable(unreadable),
            }
        }
        Some(_) => ProviderEntry::NotAMapping,
    }
}

fn declared(fields: &Fields) -> Declared {
    let set = |key: &str| fields.get(key).is_some_and(|value| !value.is_null());
    let cost_per = || fields.iter().filter(|(key, _)| key.contains("cost_per"));
    Declared {
        token_rate_key: fields.contains_key("input_cost_per_token")
            || fields.contains_key("output_cost_per_token"),
        positive_cost_per_key: set("tiered_pricing")
            || cost_per().any(|(_, value)| match value {
                Value::Object(nested) => nested.values().any(is_positive_cost),
                value => is_positive_cost(value),
            }),
        cost_per_key: fields.contains_key("tiered_pricing") || cost_per().next().is_some(),
        ptu_flat_cost: set("ptu_count") && set("cost_per_ptu_per_hour"),
    }
}

/// `_is_positive_cost`: a number above zero, never a bool
fn is_positive_cost(value: &Value) -> bool {
    value.as_f64().is_some_and(|cost| cost > 0.0)
}

#[cfg(test)]
mod tests {
    use litellm_cost::adjust::{Discounts, MarginRule, Margins};
    use litellm_cost::candidates::{
        AdjustFacts, CallIdentity, Named, PricingRows, TokenCall, Tried, candidates, priced,
        pricing_rows,
    };
    use litellm_cost::query::{
        AdjustmentPolicy, Basis, Candidate, Header, Modalities, PriceQuery, PriceSource, Provider,
        TokenUsage, TokensQuery,
    };
    use litellm_cost::receipt::{PriceOutcome, Unpriced};
    use litellm_cost::row::RowSource;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::store::Store;
    use crate::test_support::{TestRows, bundled_rows, context, names};
    use crate::view::PricingContext;
    use crate::{BUNDLED_REGISTRY, Catalog};

    const MODEL: &str = "claude-opus-4-5";

    fn discounted(discount: f64) -> PricingContext {
        PricingContext {
            config: Arc::new(CostConfig {
                discounts: Discounts::new([("anthropic".to_owned(), discount)]),
                margins: Margins::new([("global".to_owned(), MarginRule::Percent(0.0))]),
            }),
            ..context(names(&["anthropic"]))
        }
    }

    fn query(source: PriceSource) -> PriceQuery {
        PriceQuery::Tokens(Box::new(TokensQuery {
            header: Header {
                candidates: vec![Candidate {
                    model: MODEL.into(),
                    provider: Provider::Anthropic,
                }],
                source,
                basis: Basis {
                    service_tier: Default::default(),
                    data_residency: None,
                    vertex_location: None,
                    at: jiff::Timestamp::UNIX_EPOCH,
                },
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some("anthropic".into()),
                    model: None,
                    region: None,
                },
                reported: None,
            },
            usage: usage(),
            requested_model: None,
        }))
    }

    fn usage() -> TokenUsage {
        TokenUsage {
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
        }
    }

    fn total(outcome: PriceOutcome) -> f64 {
        match outcome {
            PriceOutcome::Priced(receipt) => receipt.total,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn rows(lookup: &CostLookup<'_, Store<TestRows>>) -> PricingRows<LoweredRow> {
        let found = candidates(
            lookup,
            &CallIdentity {
                provider: "anthropic".to_owned(),
                request_model: Some(MODEL.to_owned()),
                response_model: Some(MODEL.to_owned()),
                ..CallIdentity::default()
            },
        );
        pricing_rows(
            lookup,
            found.rows.into_iter().map(|Tried { model, row }| {
                (
                    Candidate {
                        model: model.into(),
                        provider: Provider::Anthropic,
                    },
                    row,
                )
            }),
            [],
        )
    }

    #[rstest]
    #[case::no_discount(0.0)]
    #[case::a_tenth_off(0.1)]
    fn a_candidate_row_prices_at_its_catalog_rates_after_the_context_discount(
        #[case] discount: f64,
    ) {
        let table = Store::new(bundled_rows());
        let context = discounted(discount);
        let row = table.fields(MODEL).unwrap();
        let rate = |key: &str| row[key].as_f64().unwrap();
        let listed = 1000.0 * rate("input_cost_per_token") + 100.0 * rate("output_cost_per_token");
        let rows = rows(&CostLookup::new(&table, &context));

        let priced = total(litellm_cost::price(&rows, &query(PriceSource::Catalog)));

        assert!(
            rows.resolve(&Candidate {
                model: "anthropic/claude-opus-4-5".into(),
                provider: Provider::Anthropic,
            })
            .is_some()
        );
        assert!(
            (priced - listed * (1.0 - discount)).abs() < 1e-12,
            "{priced} vs {listed}"
        );
    }

    #[rstest]
    fn a_per_call_row_prices_instead_of_the_catalog_row() {
        let table = Store::new(bundled_rows());
        let context = discounted(0.0);
        let (rows, at) = rows(&CostLookup::new(&table, &context)).with_per_call(
            "requested_model_info_override",
            LoweredRow::lower(
                Arc::new(fields(
                    json!({"input_cost_per_token": 1.0, "output_cost_per_token": 2.0}),
                )),
                "anthropic",
            ),
        );

        let priced = total(litellm_cost::price(
            &rows,
            &query(PriceSource::Row {
                row: at,
                breakdown_by_name: false,
            }),
        ));

        assert_eq!(priced, 1200.0);
    }

    #[rstest]
    fn a_query_reaches_only_the_rows_its_caller_fetched() {
        let table = Store::new(bundled_rows());
        let context = discounted(0.0);
        let rows = pricing_rows(&CostLookup::new(&table, &context), [], []);

        assert!(matches!(
            litellm_cost::price(&rows, &query(PriceSource::Catalog)),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }

    /// The Rust host prices a `/v1/messages` Claude call from the document it parsed, with no Python, through
    /// `litellm_cost::candidates::priced`, the entry point the Python host prices through. It bills the row the walk
    /// picks at the rates that row declares
    #[rstest]
    #[case::anthropic("anthropic", "claude-opus-5-5", "claude-opus-5-5")]
    #[case::bedrock_invoke_profile("bedrock", "us.anthropic.claude-opus-5-5", "claude-opus-5-5")]
    #[case::azure_ai("azure_ai", "claude-opus-5-5", "claude-opus-5-5")]
    fn the_rust_host_prices_a_claude_call_from_its_catalog_at_the_rows_own_rates(
        #[case] provider: &str,
        #[case] request_model: &str,
        #[case] response_model: &str,
    ) {
        let catalog = Catalog::parse(BUNDLED_REGISTRY).unwrap().value;
        let context = PricingContext {
            rules: catalog.capability_rules().clone(),
            names: names(&["anthropic", "bedrock", "azure_ai"]),
            config: Arc::new(CostConfig::default()),
        };
        let lookup = CostLookup::new(&catalog, &context);
        let call = CallIdentity {
            provider: provider.to_owned(),
            request_model: Some(request_model.to_owned()),
            response_model: Some(response_model.to_owned()),
            ..CallIdentity::default()
        };
        let found = candidates(&lookup, &call);
        let billed = &found.rows.first().expect("the catalog prices the call").row;
        let row = catalog.get(&billed.key).unwrap();
        let rate = |key: &str| row[key].as_f64().unwrap();
        let listed = 1000.0 * rate("input_cost_per_token") + 100.0 * rate("output_cost_per_token");

        let receipt = priced(
            &lookup,
            TokenCall {
                named: Named::Call(call),
                usage: usage(),
                requested_model: Some(request_model.into()),
                service_tier: None,
                data_residency: None,
                vertex_location: None,
                at: jiff::Timestamp::UNIX_EPOCH,
                adjust: Some(AdjustFacts {
                    provider: Some(provider.into()),
                    ..AdjustFacts::default()
                }),
            },
            None::<fn(&str) -> LoweredRow>,
        )
        .expect("the call prices");

        assert!(listed > 0.0);
        assert!(
            (receipt.total - listed).abs() < 1e-12,
            "{} vs {listed}",
            receipt.total
        );
    }

    fn fields(row: Value) -> Fields {
        let Value::Object(fields) = row else {
            panic!("a row is an object")
        };
        fields
    }

    #[rstest]
    #[case::absent(json!({}), ProviderEntry::Readable)]
    #[case::empty(json!({"provider_specific_entry": {}}), ProviderEntry::Readable)]
    #[case::numbers(json!({"provider_specific_entry": {"us": 1.1, "fast": "6"}}), ProviderEntry::Readable)]
    #[case::a_list(json!({"provider_specific_entry": ["us"]}), ProviderEntry::NotAMapping)]
    #[case::a_word(
        json!({"provider_specific_entry": {"US": "high", "fast": null, "eu": 1.2}}),
        ProviderEntry::Unreadable(BTreeSet::from([
            Uplift::InferenceGeo(InferenceGeo::new("us")),
            Uplift::Speed(Speed::Fast),
        ]))
    )]
    fn provider_entry_marks_the_values_float_rejects(
        #[case] row: Value,
        #[case] expected: ProviderEntry,
    ) {
        assert_eq!(row_facts(&fields(row)).provider_entry, expected);
    }

    #[rstest]
    #[case::sparse_registration(json!({"id": "dep"}), Declared::default())]
    #[case::explicit_zero(
        json!({"input_cost_per_token": 0.0, "output_cost_per_token": null}),
        Declared { token_rate_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::bool_is_not_a_price(
        json!({"input_cost_per_second": true}),
        Declared { cost_per_key: true, ..Declared::default() }
    )]
    #[case::nested_positive(
        json!({"search_context_cost_per_query": {"search_context_size_low": 0.01}}),
        Declared { positive_cost_per_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::tiers(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        Declared { positive_cost_per_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::ptu(
        json!({"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "input_cost_per_token": 0}),
        Declared { token_rate_key: true, positive_cost_per_key: true, cost_per_key: true, ptu_flat_cost: true }
    )]
    fn declared_reads_the_raw_keys_the_auth_checks_read(
        #[case] row: Value,
        #[case] expected: Declared,
    ) {
        assert_eq!(row_facts(&fields(row)).declared, expected);
    }

    #[rstest]
    #[case::chat(json!({"mode": "chat", "max_input_tokens": 1000, "max_output_tokens": "64"}), RowMode::Chat, Some(1000), Some(64))]
    #[case::no_mode(json!({"max_input_tokens": -1}), RowMode::Chat, None, None)]
    #[case::image(json!({"mode": "image_generation"}), RowMode::ImageGeneration, None, None)]
    #[case::embedding(json!({"mode": "embedding", "max_input_tokens": 8191.9}), RowMode::Other, Some(8191), None)]
    fn mode_and_limits_read_like_get_model_info(
        #[case] row: Value,
        #[case] mode: RowMode,
        #[case] max_input: Option<u64>,
        #[case] max_output: Option<u64>,
    ) {
        let facts = row_facts(&fields(row));

        assert_eq!(
            (facts.mode, facts.max_input_tokens, facts.max_output_tokens),
            (mode, max_input, max_output)
        );
    }
}
