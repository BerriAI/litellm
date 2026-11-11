use serde_json::Value;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum Recognized<T> {
    Known(T),
    Unrecognized(Value),
}

/// `deserialize_with` for an optional `Recognized` field whose explicit `null` must stay
/// distinguishable from an absent key: `null` becomes `Some(Unrecognized(Null))`.
pub fn present<'de, D, T>(deserializer: D) -> Result<Option<Recognized<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    Recognized<T>: serde::Deserialize<'de>,
{
    <Recognized<T> as serde::Deserialize<'de>>::deserialize(deserializer).map(Some)
}

impl<T> Recognized<T> {
    pub fn known(&self) -> Option<&T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unrecognized(_) => None,
        }
    }

    pub fn map_known(self, f: impl FnOnce(T) -> T) -> Self {
        match self {
            Self::Known(value) => Self::Known(f(value)),
            unrecognized => unrecognized,
        }
    }
}

pub fn map_known_runs<T>(
    items: Vec<Recognized<T>>,
    mut f: impl FnMut(Vec<T>) -> Vec<T>,
) -> Vec<Recognized<T>> {
    let mut mapped = Vec::with_capacity(items.len());
    let mut run = Vec::new();
    for item in items {
        match item {
            Recognized::Known(value) => run.push(value),
            unrecognized => {
                if !run.is_empty() {
                    mapped.extend(
                        f(std::mem::take(&mut run))
                            .into_iter()
                            .map(Recognized::Known),
                    );
                }
                mapped.push(unrecognized);
            }
        }
    }
    if !run.is_empty() {
        mapped.extend(f(run).into_iter().map(Recognized::Known));
    }
    mapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[test]
    fn runs_of_known_items_are_mapped_around_unrecognized_ones() {
        let items = vec![
            Recognized::Known(1),
            Recognized::Known(2),
            Recognized::Unrecognized(json!("x")),
            Recognized::Known(3),
        ];
        let mut runs = Vec::new();
        let mapped = map_known_runs(items, |run: Vec<u64>| {
            runs.push(run.clone());
            run.into_iter()
                .filter(|item| *item != 2)
                .map(|item| item * 10)
                .collect()
        });
        assert_eq!(runs, vec![vec![1, 2], vec![3]]);
        assert_eq!(
            mapped,
            vec![
                Recognized::Known(10),
                Recognized::Unrecognized(json!("x")),
                Recognized::Known(30),
            ]
        );
        assert_eq!(
            map_known_runs(Vec::<Recognized<u64>>::new(), |run| run),
            vec![]
        );
    }

    #[rstest]
    #[case::known(json!(7), Recognized::Known(7))]
    #[case::wrong_type(json!("7"), Recognized::Unrecognized(json!("7")))]
    #[case::out_of_range(json!(-1), Recognized::Unrecognized(json!(-1)))]
    #[case::object(json!({"a": 1}), Recognized::Unrecognized(json!({"a": 1})))]
    fn value_is_known_only_when_it_parses_as_the_type(
        #[case] value: Value,
        #[case] expected: Recognized<u64>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<u64>>(value.clone()).unwrap(),
            expected
        );
        assert_eq!(serde_json::to_value(expected).unwrap(), value);
    }
}
