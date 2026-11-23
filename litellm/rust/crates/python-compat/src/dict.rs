//! `dict` reads and writes over [`Value::Dict`], which keeps insertion order as Python does.

use crate::Value;

const NONE: Value = Value::None;

/// The entries of a dict, or `None` for any other value.
pub fn entries(value: &Value) -> Option<&[(Value, Value)]> {
    match value {
        Value::Dict(entries) => Some(entries),
        _ => None,
    }
}

/// `value[key]` for a dict with a `str` key, or `None` when the value is not a dict or lacks the key.
pub fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    lookup(entries(value)?, key)
}

/// `d[key]` over a dict's entries, without building the dict: the item under the `str` key, or
/// `None` when it is missing.
pub fn lookup<'a>(entries: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    entries
        .iter()
        .find(|(name, _)| matches!(name, Value::Str(name) if name == key))
        .map(|(_, item)| item)
}

/// `value.get(key)` for a dict: the item, or `None` when the value is not a dict or lacks the key.
pub fn get_or_none<'a>(value: &'a Value, key: &str) -> &'a Value {
    get(value, key).unwrap_or(&NONE)
}

/// `key in value` for a dict.
pub fn has(value: &Value, key: &str) -> bool {
    get(value, key).is_some()
}

/// A dict being built: `d[key] = value` replaces an existing key in place and appends a new one,
/// so the result keeps Python's insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dict(Vec<(Value, Value)>);

impl Dict {
    /// `dict(entries)`.
    pub fn from_entries(entries: &[(Value, Value)]) -> Self {
        Self(entries.to_vec())
    }

    /// `d[key] = value` for a `str` key.
    pub fn set(&mut self, key: &str, value: Value) {
        match self
            .0
            .iter_mut()
            .find(|(name, _)| matches!(name, Value::Str(name) if name == key))
        {
            Some(slot) => slot.1 = value,
            None => self.0.push((Value::Str(key.to_owned()), value)),
        }
    }

    /// `d[key] = value` for any key.
    pub fn set_value(&mut self, key: Value, value: Value) {
        match self.0.iter_mut().find(|(name, _)| *name == key) {
            Some(slot) => slot.1 = value,
            None => self.0.push((key, value)),
        }
    }

    /// `d.update(entries)`.
    pub fn extend(&mut self, entries: &[(Value, Value)]) {
        for (key, value) in entries {
            self.set_value(key.clone(), value.clone());
        }
    }

    /// `d.get(key)` for a `str` key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0
            .iter()
            .find(|(name, _)| matches!(name, Value::Str(name) if name == key))
            .map(|(_, value)| value)
    }

    /// `d.pop(key, None)` for a `str` key.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let index = self
            .0
            .iter()
            .position(|(name, _)| matches!(name, Value::Str(name) if name == key))?;
        Some(self.0.remove(index).1)
    }

    /// The finished dict.
    pub fn into_value(self) -> Value {
        Value::Dict(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(value: &str) -> Value {
        Value::Str(value.to_owned())
    }

    #[test]
    fn assignment_keeps_insertion_order() {
        let mut dict = Dict::from_entries(&[(text("a"), text("1")), (text("b"), text("2"))]);
        dict.set("a", text("3"));
        dict.set("c", text("4"));
        assert_eq!(dict.remove("b"), Some(text("2")));
        let value = dict.into_value();
        assert_eq!(get(&value, "a"), Some(&text("3")));
        assert_eq!(
            entries(&value)
                .map(|items| items.iter().map(|(key, _)| key.clone()).collect::<Vec<_>>()),
            Some(vec![text("a"), text("c")])
        );
        assert_eq!(get_or_none(&value, "missing"), &Value::None);
    }
}
