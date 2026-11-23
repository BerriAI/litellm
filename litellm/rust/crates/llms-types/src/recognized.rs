use litellm_python_compat::{
    scalar::{known_str, python_number, value_str},
    truthy::json_truthy,
};
use serde_json::Value;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum Recognized<T> {
    Known(T),
    Unrecognized(Value),
}

pub fn is_one(value: &Recognized<f64>) -> bool {
    match value {
        Recognized::Known(number) => *number == 1.0,
        Recognized::Unrecognized(other) => python_number(other) == Some(1.0),
    }
}

pub fn python_str<T: serde::Serialize>(value: &Recognized<T>) -> String {
    match value {
        Recognized::Known(known) => known_str(known),
        Recognized::Unrecognized(other) => value_str(other),
    }
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

/// `deserialize_with` for an optional typed field whose explicit `null` Python forwards: `null`
/// becomes `Some(Unrecognized(Null))`, and any other value must decode as `T`.
pub fn null_or_known<'de, D, T>(deserializer: D) -> Result<Option<Recognized<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(Some(Recognized::Unrecognized(Value::Null)));
    }
    serde_json::from_value(value)
        .map(|known| Some(Recognized::Known(known)))
        .map_err(serde::de::Error::custom)
}

/// `deserialize_with` for an optional raw field whose explicit `null` Python forwards.
pub fn present_value<'de, D>(deserializer: D) -> Result<Option<Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    <Value as serde::Deserialize>::deserialize(deserializer).map(Some)
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
            Self::Unrecognized(other) => json_truthy(other),
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

    #[rstest]
    #[case::known_one(Recognized::Known(1.0), true)]
    #[case::known_half(Recognized::Known(0.5), false)]
    #[case::int_one(Recognized::Unrecognized(json!(1)), true)]
    #[case::bool_true(Recognized::Unrecognized(json!(true)), true)]
    #[case::string_one(Recognized::Unrecognized(json!("1")), false)]
    fn one_is_compared_as_python_compares_it(
        #[case] value: Recognized<f64>,
        #[case] expected: bool,
    ) {
        assert_eq!(is_one(&value), expected);
    }

    #[rstest]
    #[case::known(Recognized::Known(0.5), "0.5")]
    #[case::known_wide(Recognized::Known(1e16), "1e+16")]
    #[case::bool(Recognized::Unrecognized(json!(false)), "False")]
    #[case::null(Recognized::Unrecognized(Value::Null), "None")]
    fn values_print_as_python_prints_them(#[case] value: Recognized<f64>, #[case] expected: &str) {
        assert_eq!(python_str(&value), expected);
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
