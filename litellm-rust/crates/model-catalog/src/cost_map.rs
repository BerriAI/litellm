use std::sync::{Arc, LazyLock};

use indexmap::IndexMap;
use serde_json::Value;
use sha1::{Digest, Sha1};

use crate::bundled::BUNDLED_REGISTRY;
use crate::catalog::{Catalog, FALLBACK_GENERALIZATIONS, IntegrityLimits};
use crate::error::Error;

/// Which document is installed: the `git_blob_id` of its bytes and the ETag it was served with
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub revision: String,
    pub etag: Option<String>,
}

/// An installed cost map document. A host builds its live table from it
#[derive(Clone, Debug)]
pub struct CostMap {
    base: Arc<Base>,
    provenance: Provenance,
}

#[derive(Debug)]
struct Base {
    catalog: Catalog,
    /// Each owner's names, the owner first, for owners with at least one alias
    groups: IndexMap<String, Vec<String>>,
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
}

static BUNDLED: LazyLock<CostMap> = LazyLock::new(|| CostMap {
    base: Arc::new(Base::new(
        Catalog::parse(BUNDLED_REGISTRY)
            .expect("the bundled registry parses")
            .value,
    )),
    provenance: Provenance {
        revision: git_blob_id(BUNDLED_REGISTRY),
        etag: None,
    },
});

impl CostMap {
    pub fn bundled() -> Self {
        BUNDLED.clone()
    }

    /// Reads a fetched document. One that does not parse, or that has fewer models than `limits` allow against the
    /// bundled one, is refused. Every alias problem it has is logged
    pub fn read(body: &[u8], etag: Option<String>, limits: IntegrityLimits) -> Result<Self, Error> {
        let loaded = Catalog::parse(body)?;
        loaded
            .value
            .check_size(BUNDLED.base.catalog.model_count(), limits)?;
        for problem in loaded.errors.iter().filter(|error| error.is_alias()) {
            litellm_tracing::warn!(
                "LiteLLM: model cost map alias problem: {}",
                litellm_error::Report::new(problem)
            );
        }
        Ok(Self {
            base: Arc::new(Base::new(loaded.value)),
            provenance: Provenance {
                revision: git_blob_id(body),
                etag,
            },
        })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.base.catalog
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Every name the row under `key` is served under, the owner first. Empty for a row without aliases
    pub fn names(&self, key: &str) -> &[String] {
        let owner = self
            .catalog()
            .aliases()
            .get(key)
            .map_or(key, String::as_str);
        self.base
            .groups
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    /// Each owner with its names, the owner first, in the order the document declared the aliases
    pub fn alias_groups(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.base
            .groups
            .iter()
            .map(|(owner, names)| (owner.as_str(), names.as_slice()))
    }

    /// The `fallback_generalizations` block as the document wrote it
    pub fn fallback_generalizations(&self) -> Option<&Value> {
        self.catalog().section(FALLBACK_GENERALIZATIONS)
    }
}

/// `git hash-object` of the bytes, which main reports as `source_revision`
pub fn git_blob_id(body: &[u8]) -> String {
    let digest = Sha1::new()
        .chain_update(format!("blob {}\0", body.len()))
        .chain_update(body)
        .finalize();
    format!("{digest:x}")
}
