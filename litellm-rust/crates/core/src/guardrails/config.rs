use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum PiiAction {
    #[default]
    Mask,
    Block,
}

/// In-process PII detection. Runs compiled regex recognizers locally with no
/// network call, unlike the Presidio guardrail which proxies to an analyzer
/// service.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalPiiConfig {
    /// Which built-in entity types to detect, and the action per type. Empty
    /// means detect every built-in recognizer and mask it.
    #[serde(default)]
    pub pii_entities_config: HashMap<String, PiiAction>,
    /// Literal terms to catch in addition to the built-in recognizers. Matched
    /// case-insensitively and reported under the `CUSTOM` entity type.
    #[serde(default)]
    pub deny_list: Vec<String>,
    /// Action applied to `deny_list` hits.
    #[serde(default)]
    pub deny_list_action: PiiAction,
    /// Literal values that are never flagged, even when a recognizer matches
    /// them (e.g. a public support email). Compared case-insensitively against
    /// the matched span text.
    #[serde(default)]
    pub allow_list: Vec<String>,
    /// Replacement for masked spans. `{entity}` expands to the entity type, so
    /// the default `<{entity}>` renders `<EMAIL_ADDRESS>`. A literal token with
    /// no placeholder (e.g. `[REDACTED]`) is used verbatim for every entity.
    #[serde(default)]
    pub mask_token: Option<String>,
}
