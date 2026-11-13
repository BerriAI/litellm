use std::collections::BTreeMap;

use litellm_cost::money::{Multiplier, UnitPrice};
use serde_json::{Map, Value};

use crate::error::{Error, Repr, pointer_for_key};
use crate::wire::attributes::Attributes;
use crate::wire::blocks::{FlatPrice, OffPeakBlock, Reader, SearchContextCostPerQuery, TierRow};
use crate::wire::flat_key::FlatKey;

pub type Fields = Map<String, Value>;

/// One catalog entry as written. Every non-null key lands in exactly one field, and a known key
/// with a bad value is kept out and listed in `problems`.
#[derive(Debug, Default)]
pub struct EntryWire {
    pub attributes: Attributes,
    pub prices: Vec<FlatPrice>,
    pub capabilities: BTreeMap<String, bool>,
    pub tier_rows: Option<Vec<TierRow>>,
    pub off_peak: Option<OffPeakBlock>,
    pub search_context: Option<SearchContextCostPerQuery>,
    pub guardrail: Option<BTreeMap<String, UnitPrice>>,
    pub regional_endpoint_uplift: Option<Multiplier>,
    pub residency_uplift_eu: Option<Multiplier>,
    pub residency_uplift_us: Option<Multiplier>,
    pub fedramp_uplift: Option<Multiplier>,
    pub unknown: Fields,
    pub problems: Vec<Error>,
}

impl EntryWire {
    pub fn read(fields: &Fields) -> Self {
        let mut problems = Vec::new();
        let mut wire = Self::default();
        let mut reader = Reader {
            problems: &mut problems,
        };
        for (key, value) in fields.iter().filter(|(_, value)| !value.is_null()) {
            wire.read_key(&mut reader, key, value);
        }
        Self { problems, ..wire }
    }

    fn read_key(&mut self, reader: &mut Reader<'_>, key: &str, value: &Value) {
        let path = || pointer_for_key(key);
        match key {
            "tiered_pricing" => self.tier_rows = reader.tier_rows(value),
            "off_peak_pricing" => self.off_peak = reader.off_peak(value),
            "search_context_cost_per_query" => self.search_context = reader.value(path(), value),
            "guardrail_cost_per_unit" => self.guardrail = reader.value(path(), value),
            "regional_endpoint_uplift_multiplier" => {
                self.regional_endpoint_uplift = reader.value(path(), value)
            }
            "regional_processing_uplift_multiplier_eu" => {
                self.residency_uplift_eu = reader.value(path(), value)
            }
            "regional_processing_uplift_multiplier_us" => {
                self.residency_uplift_us = reader.value(path(), value)
            }
            "fedramp_uplift_multiplier" => self.fedramp_uplift = reader.value(path(), value),
            _ => self.read_open_key(reader, key, value),
        }
    }

    fn read_open_key(&mut self, reader: &mut Reader<'_>, key: &str, value: &Value) {
        if let Some(read) = self.attributes.read(key, value) {
            if let Err(error) = read {
                reader
                    .problems
                    .push(Error::from_key(key, Repr::Invalid(error)));
            }
            return;
        }
        if let Some(flat) = FlatKey::parse(key) {
            self.prices
                .extend(reader.price(key.to_owned(), pointer_for_key(key), flat, value));
            return;
        }
        if key.starts_with("supports_") {
            if let Some(flag) = reader.value(pointer_for_key(key), value) {
                self.capabilities.insert(key.to_owned(), flag);
            }
            return;
        }
        self.unknown.insert(key.to_owned(), value.clone());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn read(entry: Value) -> EntryWire {
        let Value::Object(fields) = entry else {
            panic!("entry must be an object")
        };
        EntryWire::read(&fields)
    }

    fn problem_keys(wire: &EntryWire) -> Vec<String> {
        wire.problems
            .iter()
            .map(|problem| problem.path().to_owned())
            .collect()
    }

    #[rstest]
    fn every_key_lands_in_exactly_one_place() {
        let wire = read(json!({
            "litellm_provider": "openai",
            "max_tokens": "8192",
            "input_cost_per_token": "0.000001",
            "tiered_pricing": [{"range": [0, 10], "output_cost_per_token": 2e-6}],
            "supports_vision": true,
            "id": "deployment-1",
            "mode": null
        }));

        assert_eq!(wire.attributes.litellm_provider.as_deref(), Some("openai"));
        assert_eq!(wire.attributes.max_tokens, Some(8192));
        assert_eq!(wire.attributes.mode, None);
        assert_eq!(
            wire.prices
                .iter()
                .map(|price| price.key.as_str())
                .collect::<Vec<_>>(),
            ["input_cost_per_token"]
        );
        assert_eq!(wire.tier_rows.unwrap()[0].prices.len(), 1);
        assert_eq!(
            wire.capabilities,
            BTreeMap::from([("supports_vision".to_owned(), true)])
        );
        assert_eq!(wire.unknown.keys().collect::<Vec<_>>(), ["id"]);
        assert!(wire.problems.is_empty());
    }

    #[rstest]
    fn every_bad_value_is_reported_by_its_key_at_once() {
        let wire = read(json!({
            "mode": "chatt",
            "max_tokens": -1,
            "input_cost_per_token": "x",
            "output_cost_per_image_0.5K": "x",
            "supports_vision": "yes",
            "regional_endpoint_uplift_multiplier": 0.5,
            "tiered_pricing": [
                {"range": [0, 10], "input_cost_per_token": 1e-6},
                {"range": [10.5, 20], "typo_cost": 1, "output_cost_per_token": -1}
            ],
            "off_peak_pricing": {"hours_utc": "25:00-01:00", "input_cost_per_token": 1e-6}
        }));

        assert_eq!(
            problem_keys(&wire)
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [
                "/input_cost_per_token",
                "/max_tokens",
                "/mode",
                "/off_peak_pricing/hours_utc",
                "/output_cost_per_image_0.5K",
                "/regional_endpoint_uplift_multiplier",
                "/supports_vision",
                "/tiered_pricing/1/output_cost_per_token",
                "/tiered_pricing/1/range",
                "/tiered_pricing/1/typo_cost",
            ]
            .map(str::to_owned)
            .into_iter()
            .collect()
        );
    }

    #[rstest]
    fn a_malformed_off_peak_window_is_reported_and_the_well_formed_ones_are_kept() {
        use litellm_cost::schedule::Weekday;

        use crate::wire::blocks::{OffPeakWindow, UtcHours};

        let window = |hours: &str| hours.parse().unwrap();
        let wire = read(json!({"off_peak_pricing": {
            "hours_utc": ["bad", "13:00-16:00"],
            "windows": [
                "00:00-00:00",
                {"hours_utc": 1630},
                {"hours_utc": ["25:00-26:00", "01:00-02:00"], "weekdays": [0, 8, "noday", true, 6]},
                {"hours_utc": "03:00-04:00", "weekdays": "sat"},
                {"weekdays": [6]},
                {"hours_utc": "05:00-06:00", "days": [1]},
                {"hours_utc": "07:00-08:00", "weekdays": ["Sun"]}
            ],
            "input_cost_per_token": 1e-6
        }}));
        let block = wire.off_peak.as_ref().unwrap();

        assert_eq!(
            block.hours_utc,
            Some(UtcHours::Multiple(vec![window("13:00-16:00")]))
        );
        assert_eq!(
            block.windows,
            Some(vec![
                OffPeakWindow {
                    hours_utc: UtcHours::Multiple(vec![window("01:00-02:00")]),
                    weekdays: Some(vec![Weekday::Saturday]),
                },
                OffPeakWindow {
                    hours_utc: UtcHours::Single(window("03:00-04:00")),
                    weekdays: Some(Vec::new()),
                },
                OffPeakWindow {
                    hours_utc: UtcHours::Single(window("05:00-06:00")),
                    weekdays: None,
                },
                OffPeakWindow {
                    hours_utc: UtcHours::Single(window("07:00-08:00")),
                    weekdays: Some(vec![Weekday::Sunday]),
                },
            ])
        );
        assert_eq!(
            problem_keys(&wire),
            [
                "/off_peak_pricing/hours_utc",
                "/off_peak_pricing/windows/0",
                "/off_peak_pricing/windows/1",
                "/off_peak_pricing/windows/2",
                "/off_peak_pricing/windows/3",
                "/off_peak_pricing/windows/4",
                "/off_peak_pricing/windows/5",
            ]
        );
    }

    #[rstest]
    #[case::hours_not_a_window(json!({"off_peak_pricing": {"hours_utc": 1630}}))]
    #[case::one_malformed_window(json!({"off_peak_pricing": {"hours_utc": "25:00-26:00"}}))]
    #[case::windows_not_a_list(json!({"off_peak_pricing": {"windows": "00:00-00:00"}}))]
    fn off_peak_hours_main_cannot_read_are_left_out(#[case] entry: Value) {
        let wire = read(entry);
        let block = wire.off_peak.as_ref().unwrap();

        assert_eq!(
            (block.hours_utc.as_ref(), block.windows.as_ref()),
            (None, None)
        );
        assert_eq!(wire.problems.len(), 1);
    }

    #[rstest]
    #[case::tiers_not_a_list(json!({"tiered_pricing": {"range": [0, 1]}}), "/tiered_pricing")]
    #[case::row_not_an_object(json!({"tiered_pricing": [3]}), "/tiered_pricing/0")]
    #[case::off_peak_not_an_object(json!({"off_peak_pricing": true}), "/off_peak_pricing")]
    #[case::unknown_search_size(
        json!({"search_context_cost_per_query": {"search_context_size_huge": 1}}),
        "/search_context_cost_per_query"
    )]
    #[case::negative_guardrail(
        json!({"guardrail_cost_per_unit": {"contentPolicyUnits": -1}}),
        "/guardrail_cost_per_unit"
    )]
    fn a_bad_block_is_reported_at_the_block(#[case] entry: Value, #[case] expected: &str) {
        assert_eq!(problem_keys(&read(entry)), [expected]);
    }
}
