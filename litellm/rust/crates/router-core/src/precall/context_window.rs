//! The context-window stage of `_pre_call_checks`: a deployment is dropped when the prompt's
//! counted tokens exceed its `max_input_tokens`, and a group left with nothing for that reason
//! raises one `ContextWindowExceeded` naming each deployment that overflowed. A count the host
//! estimated with a stand-in tokenizer drops a deployment only past twice its window, and one
//! within that band is kept for the provider to judge.

use serde_json::Value;

use super::PreCallFacts;
use crate::{error::Error, registry::Deployment};

const STAND_IN_OVERCOUNT_MARGIN: i64 = 2;

/// A deployment the prompt does not fit.
pub(super) struct Overflow {
    model: String,
    limit: i64,
    tokens: i64,
    estimated: bool,
}

impl Overflow {
    pub(super) fn certain(&self) -> bool {
        !self.estimated || self.tokens > self.limit.saturating_mul(STAND_IN_OVERCOUNT_MARGIN)
    }

    pub(super) fn detail(&self) -> String {
        format!(
            "Model={}, Max Input Tokens={}, Got={}",
            self.model, self.limit, self.tokens
        )
    }
}

/// Whether the prompt overflows `deployment`, by the count the host made for it. A deployment
/// without a window, an uncounted prompt, or one that may not be counted (a compaction is
/// pending) always fits.
pub(super) fn overflow(deployment: &Deployment, facts: &PreCallFacts) -> Option<Overflow> {
    let found = facts.of(deployment);
    let limit = found.max_input_tokens?;
    let tokens = found.input_tokens.filter(|_| facts.countable)?;
    (tokens > limit).then(|| Overflow {
        model: named_model(deployment),
        limit,
        tokens,
        estimated: found.estimated,
    })
}

/// The error for a group every candidate was dropped from, naming each overflow back to back.
pub(super) fn exceeded<'a>(model: &str, overflows: impl Iterator<Item = &'a Overflow>) -> Error {
    Error::ContextWindowExceeded {
        model: model.to_owned(),
        detail: overflows.map(Overflow::detail).collect(),
        held_for_fallback: false,
    }
}

/// The deployment's `base_model` from its `model_info`, then its params, else its model.
pub(crate) fn named_model(deployment: &Deployment) -> String {
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
