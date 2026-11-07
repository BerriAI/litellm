use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};

use indexmap::{IndexMap, IndexSet};
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::bundled::BUNDLED_REGISTRY;
use crate::catalog::{Catalog, IntegrityLimits, Written};
use crate::error::Error;
use crate::register::Registrations;
use crate::wire::Fields;

const FALLBACK_GENERALIZATIONS: &str = "fallback_generalizations";
const ALIASES: &str = "aliases";

/// Which document is installed: the `git_blob_id` of its bytes and the ETag it was served with
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub revision: String,
    pub etag: Option<String>,
}

/// The inputs a host's live table is built from: the installed document and the registrations replayed above it.
/// It never holds the live table itself
#[derive(Clone, Debug)]
pub struct CostMap {
    base: Arc<Base>,
    overlay: Registrations,
    provenance: Provenance,
    bundled_model_count: usize,
}

#[derive(Debug)]
struct Base {
    catalog: Catalog,
    /// Each owner's names, the owner first, for owners with at least one alias
    groups: IndexMap<String, Vec<String>>,
}

/// One key of a served table, in the order Python's dict holds it
#[derive(Clone, Debug, PartialEq)]
pub struct Row<'a> {
    pub key: &'a str,
    pub shape: RowShape<'a>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RowShape<'a> {
    /// The very object served under `owner`, which this key is an alias of
    AliasOf(&'a str),
    Model(ModelRow<'a>),
    /// A top-level value that is not a model row, such as `sample_spec`, served as written
    Section(&'a Value),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelRow<'a> {
    pub fields: Cow<'a, Fields>,
    /// Every name the row is served under, the owner first. Empty for a row without aliases
    pub names: &'a [String],
    /// The fields a registration set on this row
    pub registered_fields: BTreeSet<&'a str>,
}

impl Base {
    fn new(catalog: Catalog) -> Self {
        let groups = catalog.aliases().iter().fold(
            IndexMap::<String, Vec<String>>::new(),
            |mut groups, (alias, owner)| {
                groups
                    .entry(owner.clone())
                    .or_insert_with(|| vec![owner.clone()])
                    .push(alias.clone());
                groups
            },
        );
        Self { catalog, groups }
    }

    fn contains(&self, key: &str) -> bool {
        self.catalog.get(key).is_some() || self.catalog.section(key).is_some()
    }

    /// The installed row as Python serves it, without the `aliases` list its document carried
    fn row(&self, key: &str) -> Option<Cow<'_, Fields>> {
        let fields = self
            .catalog
            .get(key)
            .or_else(|| match self.catalog.section(key) {
                Some(Value::Object(fields)) if key != FALLBACK_GENERALIZATIONS => Some(fields),
                _ => None,
            })?;
        if fields.get(ALIASES).is_none_or(Value::is_null) {
            return Some(Cow::Borrowed(fields));
        }
        Some(Cow::Owned(
            fields
                .iter()
                .filter(|(name, _)| *name != ALIASES)
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        ))
    }

    fn names(&self, key: &str) -> &[String] {
        let owner = self.catalog.aliases().get(key).map_or(key, String::as_str);
        self.groups
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

static BUNDLED: LazyLock<CostMap> = LazyLock::new(|| {
    let catalog = Catalog::parse(BUNDLED_REGISTRY)
        .expect("the bundled registry parses")
        .value;
    CostMap {
        bundled_model_count: catalog.model_count(),
        base: Arc::new(Base::new(catalog)),
        overlay: Registrations::default(),
        provenance: Provenance {
            revision: git_blob_id(BUNDLED_REGISTRY),
            etag: None,
        },
    }
});

impl CostMap {
    pub fn bundled() -> Self {
        BUNDLED.clone()
    }

    /// Installs a fetched document as the base. A document that does not parse, or that has fewer models than
    /// `limits` allow against the bundled one, changes nothing
    pub fn accept(
        &mut self,
        body: &[u8],
        etag: Option<String>,
        limits: IntegrityLimits,
    ) -> Result<&Provenance, Error> {
        let catalog = Catalog::parse(body)?.value;
        catalog.check_size(self.bundled_model_count, limits)?;
        self.base = Arc::new(Base::new(catalog));
        self.provenance = Provenance {
            revision: git_blob_id(body),
            etag,
        };
        Ok(&self.provenance)
    }

    /// Installs the bundled document as the base. It is never size checked
    pub fn accept_bundled(&mut self) -> &Provenance {
        self.base = Arc::clone(&BUNDLED.base);
        self.provenance = BUNDLED.provenance.clone();
        &self.provenance
    }

    /// Keeps `fields` for `key` across every later install, merged into the key's earlier registration
    pub fn register(&mut self, key: &str, fields: &Fields) {
        self.overlay.register(key, fields);
    }

    /// Records the row key a registration under `registered` was written to
    pub fn landed(&mut self, written: &str, registered: &str) {
        self.overlay.landed(written, registered);
    }

    pub fn registrations(&self) -> &Registrations {
        &self.overlay
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The `fallback_generalizations` block as the document wrote it
    pub fn fallback_generalizations(&self) -> Option<&Value> {
        self.base.catalog.section(FALLBACK_GENERALIZATIONS)
    }

    /// The installed document as Python serves it: every top-level key but `fallback_generalizations` in document
    /// order, then each owner's aliases
    pub fn loaded(&self) -> Vec<Row<'_>> {
        self.rows(false)
    }

    /// `loaded` with every registration applied, then the rows only a registration names
    pub fn materialize(&self) -> Vec<Row<'_>> {
        self.rows(true)
    }

    /// The installed row under `key` with the registrations that apply to it, `None` when neither has one
    pub fn registered_row(&self, key: &str) -> Option<Cow<'_, Fields>> {
        let registered = self.overlay.over(&[key]);
        let base = self.base.row(key);
        if registered.is_empty() {
            return base;
        }
        Some(Cow::Owned(merged(
            base.map(Cow::into_owned).unwrap_or_default(),
            &registered,
        )))
    }

    /// The keys `registered_row` answers for, installed keys first
    pub fn registered_keys(&self) -> Vec<&str> {
        let loaded: IndexSet<&str> = self.loaded().into_iter().map(|row| row.key).collect();
        let registered: Vec<&str> = self
            .overlay
            .entries()
            .map(|(key, _)| key)
            .filter(|key| !loaded.contains(key))
            .collect();
        loaded.into_iter().chain(registered).collect()
    }

    fn rows(&self, with_overlay: bool) -> Vec<Row<'_>> {
        let documented = self
            .base
            .catalog
            .written()
            .filter(|(key, _)| *key != FALLBACK_GENERALIZATIONS)
            .map(|(key, written)| Row {
                key,
                shape: match written {
                    Written::Entry(_) => RowShape::Model(self.model_row(key, with_overlay)),
                    Written::Section(value) => RowShape::Section(value),
                },
            });
        let aliases = self.base.groups.iter().flat_map(|(owner, names)| {
            names.iter().skip(1).map(move |alias| Row {
                key: alias,
                shape: RowShape::AliasOf(owner),
            })
        });
        let registered_only: IndexSet<&str> = if with_overlay {
            self.overlay
                .targets()
                .filter(|key| !self.base.contains(key))
                .collect()
        } else {
            IndexSet::new()
        };
        let registered_only = registered_only.into_iter().map(|key| Row {
            key,
            shape: RowShape::Model(self.model_row(key, true)),
        });
        documented.chain(aliases).chain(registered_only).collect()
    }

    fn model_row<'a>(&'a self, key: &'a str, with_overlay: bool) -> ModelRow<'a> {
        let names = self.base.names(key);
        let base = self.base.row(key).unwrap_or_default();
        let registered = if with_overlay {
            let group: Vec<&str> = match names {
                [] => vec![key],
                names => names.iter().map(String::as_str).collect(),
            };
            self.overlay.over(&group)
        } else {
            Vec::new()
        };
        if registered.is_empty() {
            return ModelRow {
                fields: base,
                names,
                registered_fields: BTreeSet::new(),
            };
        }
        ModelRow {
            registered_fields: registered
                .iter()
                .flat_map(|fields| fields.iter())
                .filter(|(_, value)| !value.is_null())
                .map(|(name, _)| name.as_str())
                .collect(),
            fields: Cow::Owned(merged(base.into_owned(), &registered)),
            names,
        }
    }
}

fn merged(base: Fields, registered: &[&Fields]) -> Fields {
    registered.iter().fold(base, |mut row, fields| {
        row.extend(
            fields
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        row
    })
}

/// `git hash-object` of the bytes, which main reports as `source_revision`
pub fn git_blob_id(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    format!("{:x}", hasher.finalize())
}
