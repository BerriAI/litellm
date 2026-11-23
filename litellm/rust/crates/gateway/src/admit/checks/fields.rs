//! Naming what a config section sets. `set!` lists the fields of a destructured section that are
//! set, each named as the config spells it, and the helpers beside it name the keys a section
//! holds beyond its typed fields.

use std::collections::BTreeMap;

use litellm_config::{AdditionalFields, Object};

pub(super) trait IsSet {
    fn is_set(&self) -> bool;
}

impl IsSet for bool {
    fn is_set(&self) -> bool {
        *self
    }
}

impl<T> IsSet for Option<T> {
    fn is_set(&self) -> bool {
        self.is_some()
    }
}

impl<T> IsSet for Box<[T]> {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> IsSet for BTreeMap<K, V> {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

impl IsSet for Object {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

/// Each of the fields that is set, each of the `on` flags that is `true`, and each `by` field its
/// check holds for, named as the config spells it. A flag written `false` is the proxy's default.
macro_rules! set {
    (
        $section:literal,
        [$($field:ident),* $(,)?]
        $(, on: [$($flag:ident),+ $(,)?])?
        $(, by: [$($checked:ident => $check:expr),+ $(,)?])?
    ) => {
        [
            $((concat!($section, stringify!($field)), IsSet::is_set($field)),)*
            $($((concat!($section, stringify!($flag)), *$flag == Some(true)),)+)?
            $($((concat!($section, stringify!($checked)), ($check)($checked)),)+)?
        ]
        .into_iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| Box::<str>::from(name))
    };
}

/// Each of `keys` that is not among `honored`.
pub(super) fn other<'a>(
    section: &'a str,
    keys: impl Iterator<Item = &'a String> + 'a,
    honored: &'a [&str],
) -> impl Iterator<Item = Box<str>> + 'a {
    keys.filter(|key| !honored.contains(&key.as_str()))
        .map(move |key| format!("{section}{key}").into())
}

pub(super) fn unknown<'a>(
    section: &'a str,
    fields: &'a AdditionalFields,
) -> impl Iterator<Item = Box<str>> + 'a {
    other(section, fields.keys(), &[])
}
