use litellm_cost::money::UnitPrice;
use litellm_cost::schedule::{UtcWindow, Weekday};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use serde_with::de::DeserializeAsWrap;

use crate::error::{Error, Located, pointer_child, pointer_for_key};
use crate::wire::flat_key::FlatKey;
use crate::wire::lax::Count;

/// A flat price key with its spelling, kept so problems can name it.
#[derive(Clone, Debug, PartialEq)]
pub struct FlatPrice {
    pub key: String,
    pub path: String,
    pub flat: FlatKey,
    pub price: UnitPrice,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TierRow {
    pub path: String,
    pub range: Option<[u64; 2]>,
    pub max_results_range: Option<[u64; 2]>,
    pub prices: Vec<FlatPrice>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OffPeakBlock {
    pub hours_utc: Option<UtcHours>,
    pub windows: Option<Vec<OffPeakWindow>>,
    pub weekday_timezone: Option<String>,
    pub prices: Vec<FlatPrice>,
}

/// One UTC window or a list of them.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum UtcHours {
    Single(UtcWindow),
    Multiple(Vec<UtcWindow>),
}

/// One off-peak window entry inside `windows`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OffPeakWindow {
    pub hours_utc: UtcHours,
    pub weekdays: Option<Vec<Weekday>>,
}

/// USD cost per web search query, keyed by search context size.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SearchContextCostPerQuery {
    pub search_context_size_low: Option<UnitPrice>,
    pub search_context_size_medium: Option<UnitPrice>,
    pub search_context_size_high: Option<UnitPrice>,
}

impl UtcHours {
    pub fn windows(&self) -> &[UtcWindow] {
        match self {
            Self::Single(window) => std::slice::from_ref(window),
            Self::Multiple(windows) => windows,
        }
    }
}

pub(crate) struct Reader<'a> {
    pub(crate) problems: &'a mut Vec<Located>,
}

impl Reader<'_> {
    pub(crate) fn value<T: DeserializeOwned>(&mut self, path: String, value: &Value) -> Option<T> {
        T::deserialize(value)
            .map_err(|error| {
                self.problems
                    .push(Located::new(path, Error::Invalid(error)))
            })
            .ok()
    }

    pub(crate) fn price(
        &mut self,
        key: String,
        path: String,
        flat: FlatKey,
        value: &Value,
    ) -> Option<FlatPrice> {
        let price = self.value(path.clone(), value)?;
        Some(FlatPrice {
            key,
            path,
            flat,
            price,
        })
    }

    fn object<'v>(&mut self, path: &str, value: &'v Value) -> Option<&'v Map<String, Value>> {
        match value {
            Value::Object(fields) => Some(fields),
            other => self
                .value::<Map<String, Value>>(path.to_owned(), other)
                .and(None),
        }
    }

    fn block_price(
        &mut self,
        name: &str,
        key: String,
        path: String,
        value: &Value,
    ) -> Option<FlatPrice> {
        let Some(flat) = FlatKey::parse(name) else {
            self.problems
                .push(Located::new(path, Error::UnknownBlockKey));
            return None;
        };
        self.price(key, path, flat, value)
    }

    pub(crate) fn tier_rows(&mut self, value: &Value) -> Option<Vec<TierRow>> {
        let path = pointer_for_key("tiered_pricing");
        let rows: Vec<Value> = self.value(path.clone(), value)?;
        Some(
            rows.iter()
                .enumerate()
                .filter_map(|(index, row)| self.tier_row(path.clone(), index, row))
                .collect(),
        )
    }

    fn tier_row(&mut self, parent: String, index: usize, value: &Value) -> Option<TierRow> {
        let path = format!("{parent}/{index}");
        let display_path = format!("tiered_pricing[{index}]");
        let fields = self.object(&path, value)?;
        let mut row = TierRow {
            path: path.clone(),
            ..TierRow::default()
        };
        for (name, value) in fields.iter().filter(|(_, value)| !value.is_null()) {
            let key = format!("{display_path}.{name}");
            let child_path = pointer_child(&path, name);
            match name.as_str() {
                "range" => row.range = self.bounds(child_path, value),
                "max_results_range" => row.max_results_range = self.bounds(child_path, value),
                _ => row
                    .prices
                    .extend(self.block_price(name, key, child_path, value)),
            }
        }
        Some(row)
    }

    fn bounds(&mut self, path: String, value: &Value) -> Option<[u64; 2]> {
        DeserializeAsWrap::<[u64; 2], [Count; 2]>::deserialize(value)
            .map(DeserializeAsWrap::into_inner)
            .map_err(|error| {
                self.problems
                    .push(Located::new(path, Error::Invalid(error)))
            })
            .ok()
    }

    pub(crate) fn off_peak(&mut self, value: &Value) -> Option<OffPeakBlock> {
        let path = pointer_for_key("off_peak_pricing");
        let fields = self.object(&path, value)?;
        let mut block = OffPeakBlock::default();
        for (name, value) in fields.iter().filter(|(_, value)| !value.is_null()) {
            let key = format!("off_peak_pricing.{name}");
            let child_path = pointer_child(&path, name);
            match name.as_str() {
                "hours_utc" => block.hours_utc = self.value(child_path, value),
                "windows" => block.windows = self.value(child_path, value),
                "weekday_timezone" => block.weekday_timezone = self.value(child_path, value),
                _ => block
                    .prices
                    .extend(self.block_price(name, key, child_path, value)),
            }
        }
        Some(block)
    }
}
