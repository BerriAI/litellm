mod overlay;

use std::collections::{BTreeSet, HashMap};

use indexmap::IndexMap;
pub use overlay::*;
use serde_json::Value;

use crate::resolve::{Keys, provider_matches};
use crate::wire::Fields;

/// Where one `register` call writes and which row it inherits from
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationTarget {
    /// Always the caller's key, never a case variant of it
    pub write: String,
    /// The first candidate that is an exact key whose row passes `provider_matches`
    pub inherit_from: Option<String>,
    /// Whether the row inherited from is the one the key names, so its rates stay as written
    pub names_the_row: bool,
    /// The match came from the stripped names, so the row is seeded from the capability rules under it
    pub seeded: bool,
}

/// One name a registration may inherit from, with the provider its row has to match
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationCandidate {
    pub name: String,
    pub provider: Option<String>,
    /// main's `_resolve_builtin_model_cost_entry` only takes rows that name a provider
    pub requires_provider: bool,
    /// The candidate is the row the key names rather than a stripped sibling
    pub names_the_row: bool,
    /// One of `_resolve_builtin_model_cost_entry`'s names, whose match seeds the row from the capability rules
    pub seeded: bool,
}

/// Python's inheritance candidates in main's order, `get_model_info`'s then `_resolve_builtin_model_cost_entry`'s
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationCandidates(pub Vec<RegistrationCandidate>);

/// The row a registration goes on top of. A lookup match is the row as written. A stripped-name match keeps the
/// capability seed, takes the row without its provider, and drops the rates a flat price replaces unless the row is the
/// one the key names. With no match the seed alone
pub fn registration_base(
    target: &RegistrationTarget,
    inherited_row: Option<&Fields>,
    seed: Option<&Fields>,
    registration: &Fields,
) -> Fields {
    let Some(row) = inherited_row else {
        return seed.cloned().unwrap_or_default();
    };
    if !target.seeded {
        return row.clone();
    }
    let inherited = if target.names_the_row {
        row.clone()
    } else {
        without_overridden_price_variants(row, registration)
    };
    let mut seeded = seed.cloned().unwrap_or_default();
    seeded.extend(
        inherited
            .into_iter()
            .filter(|(name, _)| name != "litellm_provider"),
    );
    seeded
}

/// main's "cache cost will default to 0" warning, for a custom price with no catalog base and no cache prices
pub fn misses_cache_prices(target: &RegistrationTarget, registration: &Fields) -> bool {
    let sets = |name: &str| registration.get(name).is_some_and(|value| !value.is_null());
    target.inherit_from.is_none()
        && !sets("cache_creation_input_token_cost")
        && !sets("cache_read_input_token_cost")
        && !sets("tiered_pricing")
        && (sets("input_cost_per_token") || sets("output_cost_per_token"))
}

/// Layer 3, and the fields every registration set since the last install
#[derive(Clone, Debug, Default)]
pub struct Registrations {
    persisted: IndexMap<String, Fields>,
    registered_fields: HashMap<String, BTreeSet<String>>,
    /// Keys a registration created, which never inherit from their own earlier write
    created: BTreeSet<String>,
}

/// B1. The caller's key is always the key written. The inheritance source is picked apart, from exact keys only, so a
/// case variant of a row is a new row (#39909)
pub fn registration_target(
    keys: &impl Keys,
    caller_key: &str,
    candidates: &RegistrationCandidates,
) -> RegistrationTarget {
    let inherited = candidates.0.iter().find(|candidate| {
        keys.contains(&candidate.name)
            && keys.row(&candidate.name).is_some_and(|row| {
                let row_provider = row.get("litellm_provider");
                if candidate.requires_provider && matches!(row_provider, None | Some(Value::Null)) {
                    return false;
                }
                let row_provider = match row_provider {
                    None | Some(Value::Null) => None,
                    Some(Value::String(provider)) => Some(provider.as_str()),
                    Some(_) => Some(""),
                };
                provider_matches(row_provider, candidate.provider.as_deref())
            })
    });
    RegistrationTarget {
        write: caller_key.to_owned(),
        inherit_from: inherited.map(|candidate| candidate.name.clone()),
        names_the_row: inherited.is_some_and(|candidate| candidate.names_the_row),
        seeded: inherited.is_some_and(|candidate| candidate.seeded),
    }
}

impl Registrations {
    /// Merges one persisted registration into layer 3 the way main merges `_runtime_registered_model_cost`
    pub fn record(&mut self, key: &str, fields: &Fields) {
        let merged = update_dictionary(self.persisted.get(key).unwrap_or(&Fields::new()), fields);
        self.persisted.insert(key.to_owned(), merged);
    }

    /// Notes the fields a registration of either kind set on `key`, read by Bedrock profile pricing
    pub fn note_fields(&mut self, key: &str, fields: &Fields) {
        self.registered_fields
            .entry(key.to_owned())
            .or_default()
            .extend(
                fields
                    .iter()
                    .filter(|(_, value)| !value.is_null())
                    .map(|(name, _)| name.clone()),
            );
    }

    pub fn note_created(&mut self, key: &str) {
        self.created.insert(key.to_owned());
    }

    pub fn created(&self, key: &str) -> bool {
        self.created.contains(key)
    }

    pub fn registered_fields(&self, key: &str) -> Option<&BTreeSet<String>> {
        self.registered_fields.get(key)
    }

    /// Layer 3 in recording order, replayed by every install
    pub fn replay(&self) -> impl Iterator<Item = (&str, &Fields)> {
        self.persisted
            .iter()
            .map(|(key, fields)| (key.as_str(), fields))
    }

    /// An install starts every row from the catalog again, so no field counts as registered until a replay sets it
    pub fn forget_fields(&mut self) {
        self.registered_fields.clear();
    }

    pub fn persisted(&self) -> &IndexMap<String, Fields> {
        &self.persisted
    }

    /// Puts back a snapshot of layer 3 and forgets every note about earlier writes, for test isolation
    pub fn restore(&mut self, persisted: IndexMap<String, Fields>) {
        self.persisted = persisted;
        self.registered_fields.clear();
        self.created.clear();
    }
}
