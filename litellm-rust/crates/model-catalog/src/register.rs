use std::collections::{BTreeSet, HashMap};

use indexmap::IndexMap;

use crate::wire::Fields;

/// Where one `register` call writes and which row it inherits from
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationTarget {
    /// Always the caller's key, never a case variant of it
    pub write: String,
    /// The first candidate that is an exact key whose row passes `provider_matches`
    pub inherit_from: Option<String>,
}

/// Python's inheritance candidates in main's order, `get_model_info`'s then `_resolve_builtin_model_cost_entry`'s
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrationCandidates(pub Vec<String>);

/// Layer 3, and the fields `register` last wrote, which litellm owns again
#[derive(Clone, Debug, Default)]
#[expect(dead_code, reason = "07 adds the functions that read registrations")]
pub struct Registrations {
    persisted: IndexMap<String, Fields>,
    registered_fields: HashMap<String, BTreeSet<String>>,
}
