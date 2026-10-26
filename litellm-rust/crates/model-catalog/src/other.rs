use std::num::NonZeroU64;

use litellm_cost::price::{InputUnit, Meter, SearchContextSize};
use litellm_cost::{OtherPriceSource, SearchTier};

use crate::wire::{EntryWire, Fields, FlatKey};

pub(crate) fn other_price_source(
    fields: &Fields,
    wire: &EntryWire,
    id: String,
) -> OtherPriceSource {
    let declared = fields.keys().filter_map(|key| {
        FlatKey::parse_other(key).and_then(|(flat, _)| {
            (flat.band == litellm_cost::price::Band::Base && flat.tier.is_none())
                .then_some(flat.meter)
        })
    });
    let search = fields.get("search_context_cost_per_query");
    let search_declared = [
        ("search_context_size_low", SearchContextSize::Low),
        ("search_context_size_medium", SearchContextSize::Medium),
        ("search_context_size_high", SearchContextSize::High),
    ]
    .into_iter()
    .filter_map(|(key, size)| {
        search
            .filter(|value| value.is_null() || value.get(key).is_some())
            .map(|_| Meter::WebSearch(size))
    });
    let search_tiers = wire
        .tier_rows
        .iter()
        .flatten()
        .filter_map(|row| {
            let range = row.max_results_range?;
            Some(SearchTier {
                range,
                meter: Meter::Input(InputUnit::Query {
                    over_results: NonZeroU64::new(range[0].saturating_sub(1)),
                }),
            })
        })
        .collect();
    OtherPriceSource {
        id,
        declared: declared.chain(search_declared).collect(),
        search_tiers,
    }
}

#[cfg(test)]
mod tests {
    use litellm_cost::price::{Feature, Price};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{Model, PriceLowering};

    #[rstest]
    #[case::configured(json!(2.5), Some(0.0025))]
    #[case::zero(json!(0), Some(0.0))]
    #[case::null(Value::Null, None)]
    fn configured_file_search_is_per_occurrence_only_in_sdk_context(
        #[case] value: Value,
        #[case] expected: Option<f64>,
    ) {
        let fields = json!({"litellm_provider":"openai", "file_search_cost_per_1k_calls":value})
            .as_object()
            .unwrap()
            .clone();
        let price = Price::online(Meter::Feature(Feature::FileSearchCall), Default::default());
        let sdk = Model::from_fields_with_policy(&fields, PriceLowering::SdkOther).unwrap();
        let generic = Model::from_fields(&fields).unwrap();
        assert_eq!(sdk.prices.get(&price).map(|rate| rate.get()), expected);
        assert_eq!(generic.prices.get(&price), None);
    }

    #[rstest]
    #[case::boolean(json!(true))]
    #[case::negative(json!(-1.0))]
    #[case::string(json!("bad"))]
    fn new_tool_rates_reject_bad_values_only_in_sdk_context(#[case] value: Value) {
        let fields = json!({"litellm_provider":"openai", "file_search_cost_per_1k_calls":value})
            .as_object()
            .unwrap()
            .clone();
        assert!(Model::from_fields_with_policy(&fields, PriceLowering::SdkOther).is_err());
        assert!(Model::from_fields(&fields).is_ok());
    }

    #[rstest]
    #[case::nested_null(json!({"search_context_size_low":null}), true)]
    #[case::parent_null(Value::Null, true)]
    #[case::absent_child(json!({}), false)]
    fn nested_search_declarations_preserve_absent_and_null(
        #[case] value: Value,
        #[case] declared: bool,
    ) {
        let fields = json!({"search_context_cost_per_query":value})
            .as_object()
            .unwrap()
            .clone();
        let source = other_price_source(
            &fields,
            &EntryWire::read_other(&fields),
            "primary".to_owned(),
        );
        assert_eq!(
            source
                .declared
                .contains(&Meter::WebSearch(SearchContextSize::Low)),
            declared
        );
    }

    #[rstest]
    fn scalar_search_remains_rejected_in_sdk_context() {
        let fields = json!({"litellm_provider":"openai", "search_context_cost_per_query":2.0})
            .as_object()
            .unwrap()
            .clone();
        assert!(Model::from_fields_with_policy(&fields, PriceLowering::SdkOther).is_err());
        assert!(Model::from_fields(&fields).is_err());
    }
}
