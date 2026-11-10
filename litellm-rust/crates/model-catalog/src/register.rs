use std::collections::BTreeMap;

use indexmap::IndexMap;

/// `register_model(persist_across_reloads=True)` calls, replayed above every installed document
#[derive(Clone, Debug, PartialEq)]
pub struct Registrations<V> {
    /// Each registered key's merged value, ordered by its latest registration
    entries: IndexMap<String, V>,
    /// The registered keys whose write landed on each row key, in landing order
    landed_on: IndexMap<String, Vec<String>>,
}

impl<V> Default for Registrations<V> {
    fn default() -> Self {
        Self {
            entries: IndexMap::new(),
            landed_on: IndexMap::new(),
        }
    }
}

/// One registration written to a host's table: the key it was registered under and the row key it landed on
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Landed {
    pub registered: String,
    pub written: String,
}

impl<V> Registrations<V> {
    /// Keeps `value` as the key's registration, already merged with `get(key)` by the host, and makes it the latest
    pub fn register(&mut self, key: &str, value: V) {
        self.entries.shift_remove(key);
        self.entries.insert(key.to_owned(), value);
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries.get(key)
    }

    pub fn record(&mut self, landed: &[Landed]) {
        for Landed {
            registered,
            written,
        } in landed
        {
            let names = self.landed_on.entry(written.clone()).or_default();
            if !names.contains(registered) {
                names.push(registered.clone());
            }
        }
    }

    /// Every registration, oldest first
    pub fn entries(&self) -> impl Iterator<Item = (&str, &V)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// The registrations that apply to the row under `key`, oldest first: `key`'s own and every one that landed on it
    pub fn over<'a>(&'a self, key: &str) -> Vec<&'a V> {
        let landed = self.landed_on.get(key).into_iter().flatten();
        let by_order: BTreeMap<usize, &V> = landed
            .map(String::as_str)
            .chain([key])
            .filter_map(|name| self.entries.get_full(name))
            .map(|(order, _, value)| (order, value))
            .collect();
        by_order.into_values().collect()
    }

    pub fn map<W>(&self, mut convert: impl FnMut(&V) -> W) -> Registrations<W> {
        Registrations {
            entries: self
                .entries
                .iter()
                .map(|(key, value)| (key.clone(), convert(value)))
                .collect(),
            landed_on: self.landed_on.clone(),
        }
    }
}

/// A host's live table that registrations are written into
pub trait RegistrationTable {
    type Registration;
    type Error;

    /// Writes `registration` where it lands, `None` when this table is no longer the one the host publishes
    fn place(
        &mut self,
        key: &str,
        registration: &Self::Registration,
    ) -> Result<Option<Placed>, Self::Error>;
}

/// Where a registration was written and whether a built-in row matched it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub key: String,
    pub matched_builtin: bool,
}

#[derive(Debug, PartialEq)]
pub enum Pass<'a, R> {
    /// Every entry was written. `unmatched` are those no built-in row matched
    Finished {
        landed: Vec<Landed>,
        unmatched: Vec<(&'a str, &'a R)>,
    },
    /// The table stopped being the published one before every entry was written
    Swapped { landed: Vec<Landed> },
}

impl<R> Pass<'_, R> {
    pub fn landed(&self) -> &[Landed] {
        match self {
            Pass::Finished { landed, .. } | Pass::Swapped { landed } => landed,
        }
    }
}

/// Writes `entries` into `table` in order
pub fn write_pass<'a, T: RegistrationTable>(
    table: &mut T,
    entries: &'a [(String, T::Registration)],
) -> Result<Pass<'a, T::Registration>, T::Error> {
    let mut landed = Vec::with_capacity(entries.len());
    let mut unmatched = Vec::new();
    for (key, registration) in entries {
        let Some(written) = table.place(key, registration)? else {
            return Ok(Pass::Swapped { landed });
        };
        if !written.matched_builtin {
            unmatched.push((key.as_str(), registration));
        }
        landed.push(Landed {
            registered: key.clone(),
            written: written.key,
        });
    }
    Ok(Pass::Finished { landed, unmatched })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn landed(registered: &str, written: &str) -> Landed {
        Landed {
            registered: registered.to_owned(),
            written: written.to_owned(),
        }
    }

    #[rstest]
    fn reregistering_a_key_makes_it_the_latest() {
        let mut registrations = Registrations::default();
        registrations.register("a", 1);
        registrations.register("b", 2);
        registrations.register("a", 3);
        assert_eq!(
            registrations.entries().collect::<Vec<_>>(),
            [("b", &2), ("a", &3)]
        );
    }

    #[rstest]
    fn over_applies_names_landed_on_the_row_in_registration_order() {
        let mut registrations = Registrations::default();
        registrations.register("openai/o3-pro", 1);
        registrations.register("o3-pro", 2);
        registrations.register("other", 3);
        registrations.record(&[
            landed("openai/o3-pro", "o3-pro"),
            landed("o3-pro", "o3-pro"),
        ]);
        registrations.register("openai/o3-pro", 4);
        assert_eq!(registrations.over("o3-pro"), [&2, &4]);
    }

    #[rstest]
    fn over_applies_a_keys_own_registration_that_landed_on_another_row() {
        let mut registrations = Registrations::default();
        registrations.register("openai/o3-pro", 1);
        registrations.record(&[landed("openai/o3-pro", "o3-pro")]);
        assert_eq!(
            (
                registrations.over("openai/o3-pro"),
                registrations.over("o3-pro")
            ),
            (vec![&1], vec![&1])
        );
    }

    /// A table published until `swaps_after` writes, whose rows match a built-in only when the key starts with `b`
    struct Table {
        writes: Vec<(String, i32)>,
        swaps_after: usize,
    }

    impl RegistrationTable for Table {
        type Registration = i32;
        type Error = ();

        fn place(&mut self, key: &str, registration: &i32) -> Result<Option<Placed>, ()> {
            if self.writes.len() >= self.swaps_after {
                return Ok(None);
            }
            let written = key.to_uppercase();
            self.writes.push((written.clone(), *registration));
            Ok(Some(Placed {
                matched_builtin: written.starts_with('B'),
                key: written,
            }))
        }
    }

    #[rstest]
    #[case::published_throughout(
        usize::MAX,
        Pass::Finished { landed: vec![landed("a", "A"), landed("b", "B")], unmatched: vec![("a", &1)] },
        &[("A", 1), ("B", 2)],
    )]
    #[case::swapped_after_the_first_write(
        1,
        Pass::Swapped { landed: vec![landed("a", "A")] },
        &[("A", 1)],
    )]
    fn a_pass_writes_entries_in_order_until_the_table_is_swapped(
        #[case] swaps_after: usize,
        #[case] expected: Pass<'static, i32>,
        #[case] writes: &[(&str, i32)],
    ) {
        let entries = [("a".to_owned(), 1), ("b".to_owned(), 2)];
        let mut table = Table {
            writes: Vec::new(),
            swaps_after,
        };
        assert_eq!(write_pass(&mut table, &entries), Ok(expected));
        assert_eq!(
            table.writes,
            writes
                .iter()
                .map(|(key, value)| ((*key).to_owned(), *value))
                .collect::<Vec<_>>()
        );
    }
}
