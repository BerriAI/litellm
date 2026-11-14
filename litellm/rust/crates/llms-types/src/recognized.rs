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

/// `deserialize_with` for an optional float Python forwards as the caller sent it: an integer
/// stays `Unrecognized` with its exact digits, as Python sends, logs and prints it, and `null`
/// stays absent as before.
pub fn exact_float<'de, D>(deserializer: D) -> Result<Option<Recognized<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(None);
    }
    if matches!(&value, Value::Number(number) if !number.is_f64()) {
        return Ok(Some(Recognized::Unrecognized(value)));
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
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

impl Recognized<bool> {
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Known(flag) => *flag,
            Self::Unrecognized(other) => python_truthy(other),
        }
    }
}

pub fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_none_or(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
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

    #[rstest]
    #[case::known_true(Recognized::Known(true), true)]
    #[case::known_false(Recognized::Known(false), false)]
    #[case::string_true(Recognized::Unrecognized(json!("true")), true)]
    #[case::string_false(Recognized::Unrecognized(json!("false")), true)]
    #[case::empty_string(Recognized::Unrecognized(json!("")), false)]
    #[case::zero(Recognized::Unrecognized(json!(0)), false)]
    #[case::negative_zero(Recognized::Unrecognized(json!(-0.0)), false)]
    #[case::one(Recognized::Unrecognized(json!(1)), true)]
    #[case::null(Recognized::Unrecognized(Value::Null), false)]
    #[case::empty_list(Recognized::Unrecognized(json!([])), false)]
    #[case::list(Recognized::Unrecognized(json!([0])), true)]
    #[case::empty_object(Recognized::Unrecognized(json!({})), false)]
    #[case::object(Recognized::Unrecognized(json!({"a": 0})), true)]
    fn flags_are_read_by_python_truthiness(#[case] flag: Recognized<bool>, #[case] expected: bool) {
        assert_eq!(flag.is_truthy(), expected);
    }

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

    #[derive(serde::Deserialize)]
    struct Sampling {
        #[serde(default, deserialize_with = "exact_float")]
        temperature: Option<Recognized<f64>>,
    }

    #[rstest]
    #[case::float(json!({"temperature": 0.5}), Some(Recognized::Known(0.5)))]
    #[case::small_int(json!({"temperature": 2}), Some(Recognized::Unrecognized(json!(2))))]
    #[case::zero(json!({"temperature": 0}), Some(Recognized::Unrecognized(json!(0))))]
    #[case::integral_float(json!({"temperature": 1.0}), Some(Recognized::Known(1.0)))]
    #[case::largest_exact_int(json!({"temperature": 9_007_199_254_740_992_i64}), Some(Recognized::Unrecognized(json!(9_007_199_254_740_992_i64))))]
    #[case::inexact_int(json!({"temperature": 1_152_921_504_606_846_977_i64}), Some(Recognized::Unrecognized(json!(1_152_921_504_606_846_977_i64))))]
    #[case::inexact_negative_int(json!({"temperature": -9_223_372_036_854_775_807_i64}), Some(Recognized::Unrecognized(json!(-9_223_372_036_854_775_807_i64))))]
    #[case::inexact_unsigned_int(json!({"temperature": 18_446_744_073_709_551_615_u64}), Some(Recognized::Unrecognized(json!(18_446_744_073_709_551_615_u64))))]
    #[case::string(json!({"temperature": "0.5"}), Some(Recognized::Unrecognized(json!("0.5"))))]
    #[case::null(json!({"temperature": null}), None)]
    #[case::absent(json!({}), None)]
    fn integers_keep_their_digits(#[case] value: Value, #[case] expected: Option<Recognized<f64>>) {
        assert_eq!(
            serde_json::from_value::<Sampling>(value)
                .unwrap()
                .temperature,
            expected
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
