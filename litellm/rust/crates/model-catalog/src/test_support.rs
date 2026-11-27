//! A table shaped like Python's `litellm.model_cost` for resolve tests: the bundled document as Python serves it, plus
//! rows a test registers

use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::store::{Rows, Store};
use litellm_cost::adjust::CostConfig;

use crate::view::{CatalogView, KnownNames, PricingContext};
use crate::wire::Fields;
use crate::{CapabilityRules, CostMap};

/// A row object. Aliases of one model share it, as in `litellm.model_cost`
#[derive(Clone, Debug)]
pub(crate) struct TestRow(Arc<(Fields, BTreeSet<String>)>);

impl TestRow {
    pub(crate) fn new(fields: Fields, registered_fields: BTreeSet<String>) -> Self {
        Self(Arc::new((fields, registered_fields)))
    }

    pub(crate) fn fields(&self) -> Fields {
        self.0.0.clone()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TestRows(pub(crate) IndexMap<String, TestRow>);

impl Rows for TestRows {
    type Row = TestRow;
    type Stamp = ();

    fn fold(name: &str) -> String {
        name.to_lowercase()
    }

    fn stamp(&self) -> Self::Stamp {}

    fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn row(&self, key: &str) -> Option<Cow<'_, TestRow>> {
        self.0.get(key).map(Cow::Borrowed)
    }

    fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
        for (key, row) in &self.0 {
            visit(
                key,
                row.0
                    .0
                    .get("litellm_provider")
                    .and_then(|provider| provider.as_str()),
            );
        }
    }
}

/// The bundled document as Python's `litellm.model_cost`, every alias sharing its owner's row object
pub(crate) fn bundled_rows() -> TestRows {
    let cost_map = CostMap::bundled();
    let catalog = cost_map.catalog();
    let owners: IndexMap<&str, TestRow> = catalog
        .model_names()
        .map(|name| {
            let fields = catalog.get(name).cloned().unwrap_or_default();
            (name, TestRow::new(fields, BTreeSet::new()))
        })
        .collect();
    TestRows(
        catalog
            .served()
            .map(|(name, _)| {
                let owner = catalog.aliases().get(name).map_or(name, String::as_str);
                (name.to_owned(), owners[owner].clone())
            })
            .collect(),
    )
}

impl CatalogView for Store<TestRows> {
    fn contains(&self, key: &str) -> bool {
        self.rows().contains(key)
    }

    fn find_key(&self, name: &str) -> Option<String> {
        Store::find_key(self, name)
    }

    fn fields(&self, key: &str) -> Option<Arc<Fields>> {
        Some(Arc::new(self.get(key)?.fields()))
    }

    fn registered_fields(&self, key: &str) -> BTreeSet<String> {
        self.get(key).map(|row| row.0.1.clone()).unwrap_or_default()
    }
}

pub(crate) fn context(names: KnownNames) -> PricingContext {
    PricingContext {
        rules: bundled_rules(),
        names,
        config: Arc::new(CostConfig::default()),
    }
}

pub(crate) fn bundled_rules() -> CapabilityRules {
    CostMap::bundled().catalog().capability_rules().clone()
}

pub(crate) fn names(providers: &[&str]) -> KnownNames {
    KnownNames {
        providers: providers
            .iter()
            .map(|provider| (*provider).to_owned())
            .collect(),
        renamed_models: HashSet::new(),
    }
}
