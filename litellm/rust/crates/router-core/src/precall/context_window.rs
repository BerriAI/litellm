//! The context-window stage of `_pre_call_checks`: a deployment is dropped when the prompt's
//! counted tokens exceed its `max_input_tokens`, and a group left with nothing for that reason
//! raises one `ContextWindowExceeded` naming each deployment that overflowed.

use serde_json::Value;

use super::PreCallFacts;
use crate::{error::Error, registry::Deployment};

/// A deployment the prompt does not fit.
pub(super) struct Overflow {
    model: String,
    limit: i64,
    tokens: i64,
}

impl Overflow {
    fn detail(&self) -> String {
        format!(
            "Model={}, Max Input Tokens={}, Got={}",
            self.model, self.limit, self.tokens
        )
    }
}

/// Whether the prompt overflows `deployment`. A deployment without a window, an uncounted
/// prompt, or one that may not be counted (a compaction is pending) always fits.
pub(super) fn overflow(deployment: &Deployment, facts: &PreCallFacts) -> Option<Overflow> {
    let limit = facts.of(deployment).max_input_tokens?;
    let tokens = facts.input_tokens.filter(|_| facts.countable)?;
    (tokens > limit).then(|| Overflow {
        model: named_model(deployment),
        limit,
        tokens,
    })
}

/// The error for a group every candidate was dropped from, naming each overflow back to back.
pub(super) fn exceeded<'a>(model: &str, overflows: impl Iterator<Item = &'a Overflow>) -> Error {
    Error::ContextWindowExceeded {
        model: model.to_owned(),
        detail: overflows.map(Overflow::detail).collect(),
    }
}

/// The deployment's `base_model` from its `model_info`, then its params, else its model.
fn named_model(deployment: &Deployment) -> String {
    let base_model = |section: &serde_json::Map<String, Value>| {
        section
            .get("base_model")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
    };
    base_model(&deployment.model_info.raw)
        .or_else(|| base_model(&deployment.raw_params))
        .unwrap_or_else(|| deployment.params.model.clone())
}
