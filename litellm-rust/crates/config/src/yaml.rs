use std::fmt;

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};
use serde_yaml_ng::{Mapping, Value};

pub fn mapping(text: &str) -> Result<Mapping, serde_yaml_ng::Error> {
    serde_yaml_ng::from_value(serde_yaml_ng::from_str::<LastWins>(text)?.0)
}

struct LastWins(Value);

impl<'de> Deserialize<'de> for LastWins {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(LastWinsVisitor).map(Self)
    }
}

// PyYAML safe_load keeps the later value of a duplicate key, serde_yaml_ng rejects it
struct LastWinsVisitor;

impl<'de> Visitor<'de> for LastWinsVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("any YAML value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_str<E>(self, value: &str) -> Result<Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Value, E> {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        LastWins::deserialize(deserializer).map(|value| value.0)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut items: A) -> Result<Value, A::Error> {
        let mut sequence = Vec::new();
        while let Some(LastWins(item)) = items.next_element()? {
            sequence.push(item);
        }
        Ok(Value::Sequence(sequence))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Value, A::Error> {
        let mut mapping = Mapping::new();
        while let Some((LastWins(key), LastWins(value))) = entries.next_entry()? {
            mapping.insert(key, value);
        }
        Ok(Value::Mapping(mapping))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn yaml(text: &str) -> Mapping {
        serde_yaml_ng::from_str(text).unwrap()
    }

    #[rstest]
    #[case::top_level(
        r#"
a: 1
b: 2
a: 3
"#,
        "{a: 3, b: 2}"
    )]
    #[case::sequence_value(
        r#"
cb: [prometheus]
cb: [s3]
"#,
        "cb: [s3]"
    )]
    #[case::nested("outer: {x: [1], y: 2, x: [3]}", "outer: {x: [3], y: 2}")]
    #[case::deeply_nested("a: {b: {c: 1, c: 2}}", "a: {b: {c: 2}}")]
    #[case::inside_sequence("list: [{k: old, k: new}]", "list: [{k: new}]")]
    #[case::mapping_replaced_by_scalar(
        r#"
a: {b: 1}
a: 2
"#,
        "a: 2"
    )]
    #[case::value_replaced_by_null(
        r#"
a: 1
a:
"#,
        "a: null"
    )]
    #[case::three_occurrences(
        r#"
a: 1
a: 2
a: 3
"#,
        "a: 3"
    )]
    #[case::non_string_keys(
        r#"
1: a
true: b
1: c
"#,
        "{1: c, true: b}"
    )]
    #[case::scalars_without_duplicates(
        r#"
i: -1
u: 18446744073709551615
f: 1.5
b: false
n: ~
s: text
"#,
        "{i: -1, u: 18446744073709551615, f: 1.5, b: false, n: ~, s: text}"
    )]
    #[case::empty_document("", "{}")]
    fn later_duplicate_keys_override_earlier_ones(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(mapping(text).unwrap(), yaml(expected));
    }

    #[rstest]
    fn duplicate_key_keeps_its_first_position() {
        let keys: Vec<_> = mapping("a: 1\nb: 2\na: 3\n").unwrap().into_keys().collect();
        assert_eq!(keys, [Value::from("a"), Value::from("b")]);
    }

    #[rstest]
    #[case::sequence(
        r#"
- a
- b
"#
    )]
    #[case::scalar("text")]
    #[case::malformed("a: [")]
    fn rejects_a_document_that_is_not_a_mapping(#[case] text: &str) {
        assert!(mapping(text).is_err());
    }
}
