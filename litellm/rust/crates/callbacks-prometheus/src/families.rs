//! The metric families `PrometheusLogger` declares with its default settings, in the order it
//! declares them, and the label names `get_labels_for_metric` gives each metric. `families.json`
//! is recorded from the Python logger, and `tests/unit/integrations/test_prometheus_rust_families.py`
//! checks it against the logger as it is.

use serde::Deserialize;

use crate::{
    store::{FamilySpec, Kind, Store, StoreError},
    tracker::SeriesLimits,
};

const FAMILIES: &str = include_str!("../families.json");

#[derive(Deserialize)]
struct Declared {
    declared: String,
    documentation: String,
    kind: String,
    labelnames: Vec<String>,
    #[serde(default)]
    buckets: Vec<f64>,
}

#[derive(Deserialize)]
struct Table {
    families: Vec<Declared>,
    metric_labels: std::collections::BTreeMap<String, Vec<String>>,
}

/// A family table that does not describe a store.
#[derive(Debug, thiserror::Error)]
pub enum FamiliesError {
    #[error("families.json: {0}")]
    Table(#[from] serde_json::Error),
    #[error("families.json: {0} is no metric kind")]
    Kind(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// A store holding every family the logger declares with its default settings, which has no
/// series cap and no excluded label, so no family is wrapped.
pub fn default_store(
    limits: SeriesLimits,
    clock: Box<dyn crate::Clock>,
) -> Result<Store, FamiliesError> {
    let table: Table = serde_json::from_str(FAMILIES)?;
    let store = Store::new(limits, clock);
    for family in table.families {
        let kind: Kind = family
            .kind
            .parse()
            .map_err(|_| FamiliesError::Kind(family.kind.clone()))?;
        store.register(FamilySpec {
            declared: family.declared,
            documentation: family.documentation,
            kind,
            labelnames: family.labelnames,
            excluded: Vec::new(),
            buckets: family.buckets,
            most_recent: false,
            wrapped: false,
            shares_overflow: kind != Kind::Gauge,
        })?;
    }
    for (metric, labels) in table.metric_labels {
        store.declare_labels(&metric, labels);
    }
    Ok(store)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        ManualClock,
        exposition::{Scrape, render},
    };

    fn store() -> Store {
        default_store(SeriesLimits::default(), Box::new(ManualClock::default()))
            .unwrap_or_else(|error| panic!("{error}"))
    }

    #[rstest]
    fn every_family_registers_once_under_the_name_python_declares() {
        let store = store();
        let table: Table = serde_json::from_str(FAMILIES).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(store.snapshots().len(), table.families.len());
        for family in &table.families {
            assert!(
                store.family(&family.declared).is_some(),
                "{}",
                family.declared
            );
        }
    }

    #[rstest]
    fn an_unlabeled_family_is_scraped_at_zero_and_a_labeled_one_by_its_help_alone() {
        let text = render(
            &store(),
            &Scrape {
                names: Vec::new(),
                created: false,
            },
        );
        assert!(text.contains(
            "# HELP litellm_total_users Total number of users in LiteLLM\n# TYPE litellm_total_users gauge\nlitellm_total_users 0.0\n"
        ));
        assert!(text.contains(
            "# TYPE litellm_requests_metric_total counter\n# HELP litellm_zero_cost_requests_total"
        ));
    }
}
