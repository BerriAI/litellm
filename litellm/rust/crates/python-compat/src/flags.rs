//! LiteLLM's own text-to-bool helpers, one function per Python helper.
//!
//! | Function | Python |
//! |---|---|
//! | [`str_to_bool`] | `litellm.secret_managers.main.str_to_bool(text)` |
//! | [`redis_str_to_bool`] | `litellm._redis._str_to_bool(text)` |
//! | [`lower_is_true`] | `text.lower() == "true"` |
//! | [`coerces_to_true`] | `litellm._redis._coerces_to_true(value)` |
//! | [`str_to_bool_or_truthy`] | `bool(str_to_bool(value) if isinstance(value, str) else value)` |
//! | [`lax_bool`] | `litellm.litellm_core_utils.core_helpers.lax_bool(value)` |

use crate::{Value, pydantic, truthy::truthy, unicode::isspace};

/// `true` or `false` in any case once stripped, otherwise `None`.
pub fn str_to_bool(text: &str) -> Option<bool> {
    let word = text.trim_matches(isspace);
    if word.eq_ignore_ascii_case("true") {
        return Some(true);
    }
    word.eq_ignore_ascii_case("false").then_some(false)
}

/// `text.lower() in ("true", "1", "yes")`, with no stripping.
pub fn redis_str_to_bool(text: &str) -> bool {
    ["true", "1", "yes"]
        .iter()
        .any(|word| text.eq_ignore_ascii_case(word))
}

/// `text.lower() == "true"`, with no stripping.
pub fn lower_is_true(text: &str) -> bool {
    text.eq_ignore_ascii_case("true")
}

/// Text read by [`redis_str_to_bool`], anything else by `bool(value)`.
pub fn coerces_to_true(value: &Value) -> bool {
    match value {
        Value::Str(text) => redis_str_to_bool(text),
        other => truthy(other),
    }
}

/// Text read by [`str_to_bool`], where `None` counts as false, anything else by `bool(value)`.
pub fn str_to_bool_or_truthy(value: &Value) -> bool {
    match value {
        Value::Str(text) => str_to_bool(text) == Some(true),
        other => truthy(other),
    }
}

/// Pydantic's lax `bool` after stripping text, `None` where pydantic refuses or the value is `None`.
pub fn lax_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Str(text) => pydantic::lax_bool(&text.trim_matches(isspace).into()),
        other => pydantic::lax_bool(other),
    }
}
