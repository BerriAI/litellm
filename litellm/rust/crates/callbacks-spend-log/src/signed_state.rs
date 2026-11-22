//! The response fields a provider verifies when a later turn replays them: Claude thinking blocks,
//! Responses API reasoning items and Gemini thought signatures. Cutting one makes the provider
//! refuse the follow-up, so the row keeps them whole.

use litellm_python_compat::Value;

use crate::consts::{
    REASONING_FIELDS, REDACTED_THINKING_FIELDS, SIGNED_TOOL_CALL_ID_KEYS, THINKING_FIELDS,
    THOUGHT_SIGNATURE_KEYS, THOUGHT_SIGNATURE_SEPARATOR,
};
use crate::py::{as_str, lookup, truthy};

/// The signed fields of one stored mapping, by its block type.
pub(crate) struct SignedParent {
    fields: &'static [&'static str],
}

impl SignedParent {
    pub(crate) fn of(parent: &[(Value, Value)]) -> Self {
        let fields: &'static [&'static str] = match lookup(parent, "type").and_then(as_str) {
            Some("thinking") if lookup(parent, "signature").is_some_and(truthy) => &THINKING_FIELDS,
            Some("redacted_thinking") => &REDACTED_THINKING_FIELDS,
            Some("reasoning") => &REASONING_FIELDS,
            _ => &[],
        };
        Self { fields }
    }

    /// `is_provider_signed(parent, key, value)`.
    pub(crate) fn signs(&self, key: &str, value: &Value) -> bool {
        if SIGNED_TOOL_CALL_ID_KEYS.contains(&key) {
            return as_str(value).is_some_and(|id| id.contains(THOUGHT_SIGNATURE_SEPARATOR));
        }
        THOUGHT_SIGNATURE_KEYS.contains(&key) || self.fields.contains(&key)
    }
}
