//! A live table for snapshot tests: the bundled document as Python serves it, plus rows a test registers

use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use indexmap::IndexMap;

use crate::snapshot::{Generation, KnownNames, Snapshot, SnapshotInputs, TableRow};
use crate::store::{Rows, Store};
use crate::wire::Fields;
use crate::{CapabilityRules, CostMap};

/// A row object. Aliases of one model share it, as in `litellm.model_cost`
#[derive(Clone, Debug)]
pub(crate) struct TestRow(Arc<(Fields, BTreeSet<String>)>);

impl TestRow {
    pub(crate) fn new(fields: Fields, registered_fields: BTreeSet<String>) -> Self {
        Self(Arc::new((fields, registered_fields)))
    }
}

impl TableRow for TestRow {
    type Identity = usize;

    fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    fn fields(&self) -> Fields {
        self.0.0.clone()
    }

    fn registered_fields(&self) -> BTreeSet<String> {
        self.0.1.clone()
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

pub(crate) const GENERATION: Generation = Generation {
    table: 1,
    size: 1,
    writes: 0,
    discount: 0,
    margin: 0,
};

pub(crate) fn snapshot(rows: TestRows, names: KnownNames) -> Snapshot<Store<TestRows>> {
    Snapshot::new(
        GENERATION,
        SnapshotInputs {
            table: Store::new(rows),
            rules: Arc::new(bundled_rules()),
            names: Arc::new(names),
        },
    )
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
