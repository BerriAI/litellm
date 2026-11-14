use std::fmt;

use litellm_host::raw_json;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeStruct,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i128),
    Float(f64),
    Text(String),
    Stringified {
        text: String,
        truthy: bool,
        hashable: bool,
        integer: bool,
        type_name: String,
        hash_error: Option<String>,
    },
}

impl Scalar {
    pub fn truthy(&self) -> bool {
        match self {
            Self::Bool(flag) => *flag,
            Self::Int(number) => *number != 0,
            Self::Float(number) => *number != 0.0,
            Self::Text(text) => !text.is_empty(),
            Self::Stringified { truthy, .. } => *truthy,
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Text(_) => "str",
            Self::Stringified { type_name, .. } => type_name,
        }
    }

    pub fn hash_error(&self) -> Option<&str> {
        match self {
            Self::Stringified { hash_error, .. } => hash_error.as_deref(),
            _ => None,
        }
    }
}

fn non_finite(value: f64) -> &'static str {
    if value.is_nan() {
        "NaN"
    } else if value > 0.0 {
        "Infinity"
    } else {
        "-Infinity"
    }
}

fn ascii_literal(text: &str) -> String {
    let mut literal = String::with_capacity(text.len() + 2);
    literal.push('"');
    for character in text.chars() {
        match character {
            '"' => literal.push_str("\\\""),
            '\\' => literal.push_str("\\\\"),
            '\n' => literal.push_str("\\n"),
            '\r' => literal.push_str("\\r"),
            '\t' => literal.push_str("\\t"),
            '\u{8}' => literal.push_str("\\b"),
            '\u{c}' => literal.push_str("\\f"),
            ' '..='~' => literal.push(character),
            _ => {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    literal.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    literal.push('"');
    literal
}

struct TextOut<'a>(&'a str);

impl Serialize for TextOut<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if raw_json::is_marked(self.0) {
            serializer.serialize_str(&raw_json::encode(&ascii_literal(self.0)))
        } else {
            serializer.serialize_str(self.0)
        }
    }
}

impl Serialize for Scalar {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bool(flag) => serializer.serialize_bool(*flag),
            Self::Int(number) => match (i64::try_from(*number), u64::try_from(*number)) {
                (Ok(small), _) => serializer.serialize_i64(small),
                (_, Ok(unsigned)) => serializer.serialize_u64(unsigned),
                _ => serializer.serialize_str(&raw_json::encode(&number.to_string())),
            },
            Self::Float(number) if number.is_finite() => serializer.serialize_f64(*number),
            Self::Float(number) => serializer.serialize_str(&raw_json::encode(non_finite(*number))),
            Self::Text(text) => TextOut(text).serialize(serializer),
            Self::Stringified {
                text,
                truthy,
                hashable,
                integer,
                type_name,
                hash_error,
            } => {
                let mut members = serializer.serialize_struct("Stringified", 6)?;
                members.serialize_field("text", &TextOut(text))?;
                members.serialize_field("truthy", truthy)?;
                members.serialize_field("hashable", hashable)?;
                members.serialize_field("integer", integer)?;
                members.serialize_field("type_name", &TextOut(type_name))?;
                members.serialize_field("hash_error", &hash_error.as_deref().map(TextOut))?;
                members.end()
            }
        }
    }
}

fn marked<E: de::Error>(raw: &str) -> Result<Scalar, E> {
    match raw {
        "NaN" => Ok(Scalar::Float(f64::NAN)),
        "Infinity" => Ok(Scalar::Float(f64::INFINITY)),
        "-Infinity" => Ok(Scalar::Float(f64::NEG_INFINITY)),
        _ if raw.starts_with('"') => serde_json::from_str(raw)
            .map(Scalar::Text)
            .map_err(E::custom),
        _ => raw
            .parse()
            .map(Scalar::Int)
            .map_err(|_| E::custom(format!("{raw} is outside the i128 range"))),
    }
}

struct TextIn(String);

impl<'de> Deserialize<'de> for TextIn {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Scalar::deserialize(deserializer)? {
            Scalar::Text(text) => Ok(Self(text)),
            other => Err(de::Error::custom(format!(
                "{other:?} is not the str() text of a value"
            ))),
        }
    }
}

const STRINGIFIED: &[&str] = &[
    "text",
    "truthy",
    "hashable",
    "integer",
    "type_name",
    "hash_error",
];

struct ScalarVisitor;

impl<'de> Visitor<'de> for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a bool, an integer, a float, a string or a stringified value")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut members: A) -> Result<Scalar, A::Error> {
        let mut text = None;
        let mut truthy = None;
        let mut hashable = None;
        let mut integer = None;
        let mut type_name = None;
        let mut hash_error = None;
        while let Some(name) = members.next_key::<String>()? {
            match name.as_str() {
                "text" => text = Some(members.next_value::<TextIn>()?.0),
                "truthy" => truthy = Some(members.next_value::<bool>()?),
                "hashable" => hashable = Some(members.next_value::<bool>()?),
                "integer" => integer = Some(members.next_value::<bool>()?),
                "type_name" => type_name = Some(members.next_value::<TextIn>()?.0),
                "hash_error" => {
                    hash_error = Some(members.next_value::<Option<TextIn>>()?.map(|text| text.0));
                }
                other => return Err(de::Error::unknown_field(other, STRINGIFIED)),
            }
        }
        Ok(Scalar::Stringified {
            text: text.ok_or_else(|| de::Error::missing_field("text"))?,
            truthy: truthy.ok_or_else(|| de::Error::missing_field("truthy"))?,
            hashable: hashable.ok_or_else(|| de::Error::missing_field("hashable"))?,
            integer: integer.ok_or_else(|| de::Error::missing_field("integer"))?,
            type_name: type_name.ok_or_else(|| de::Error::missing_field("type_name"))?,
            hash_error: hash_error.ok_or_else(|| de::Error::missing_field("hash_error"))?,
        })
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Scalar, E> {
        Ok(Scalar::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Scalar, E> {
        Ok(Scalar::Int(value.into()))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Scalar, E> {
        Ok(Scalar::Int(value.into()))
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Scalar, E> {
        Ok(Scalar::Int(value))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Scalar, E> {
        Ok(Scalar::Float(value))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Scalar, E> {
        match raw_json::decode(value) {
            Some(raw) => marked(raw),
            None => Ok(Scalar::Text(value.to_owned())),
        }
    }
}

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ScalarVisitor)
    }
}

pub mod float {
    use litellm_host::raw_json;
    use serde::{Deserialize, Deserializer, Serializer, de};

    use super::{Scalar, non_finite};

    pub fn serialize<S: Serializer>(value: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_none(),
            Some(number) if number.is_finite() => serializer.serialize_some(number),
            Some(number) => serializer.serialize_some(&raw_json::encode(non_finite(*number))),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        match Option::<Scalar>::deserialize(deserializer)? {
            None => Ok(None),
            Some(Scalar::Float(number)) => Ok(Some(number)),
            Some(other) => Err(de::Error::custom(format!("{other:?} is not a float"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn stringified(
        text: &str,
        truthy: bool,
        hashable: bool,
        integer: bool,
        type_name: &str,
        hash_error: Option<&str>,
    ) -> Scalar {
        Scalar::Stringified {
            text: text.to_owned(),
            truthy,
            hashable,
            integer,
            type_name: type_name.to_owned(),
            hash_error: hash_error.map(str::to_owned),
        }
    }

    #[rstest]
    #[case::flag(json!(true), Scalar::Bool(true))]
    #[case::int(json!(123), Scalar::Int(123))]
    #[case::negative(json!(-7), Scalar::Int(-7))]
    #[case::unsigned(json!(u64::MAX), Scalar::Int(u64::MAX.into()))]
    #[case::float(json!(1.5), Scalar::Float(1.5))]
    #[case::whole_float(json!(4.0), Scalar::Float(4.0))]
    #[case::text(json!(" 20 "), Scalar::Text(" 20 ".to_owned()))]
    #[case::wide_int(json!(raw_json::encode("100000000000000000000")), Scalar::Int(10_i128.pow(20)))]
    #[case::infinity(json!(raw_json::encode("Infinity")), Scalar::Float(f64::INFINITY))]
    #[case::negative_infinity(json!(raw_json::encode("-Infinity")), Scalar::Float(f64::NEG_INFINITY))]
    #[case::marked_text(json!(raw_json::encode(r#""\uf8fflitellm-raw-json:\n\"\u00e9\ud83d\ude00""#)), Scalar::Text("\u{f8ff}litellm-raw-json:\n\"\u{e9}\u{1f600}".to_owned()))]
    #[case::stringified_list(json!({"text": "['a1']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}), stringified("['a1']", true, false, false, "list", Some("unhashable type: 'list'")))]
    #[case::stringified_empty_set(json!({"text": "set()", "truthy": false, "hashable": false, "integer": false, "type_name": "set", "hash_error": "unhashable type: 'set'"}), stringified("set()", false, false, false, "set", Some("unhashable type: 'set'")))]
    #[case::stringified_tuple(json!({"text": "('a',)", "truthy": true, "hashable": true, "integer": false, "type_name": "tuple", "hash_error": null}), stringified("('a',)", true, true, false, "tuple", None))]
    #[case::stringified_past_i128(json!({"text": "170141183460469231731687303715884105728", "truthy": true, "hashable": true, "integer": true, "type_name": "int", "hash_error": null}), stringified("170141183460469231731687303715884105728", true, true, true, "int", None))]
    #[case::stringified_marked_text(json!({"text": raw_json::encode(r#""\uf8fflitellm-raw-json:x""#), "truthy": true, "hashable": true, "integer": false, "type_name": "Marked", "hash_error": null}), stringified("\u{f8ff}litellm-raw-json:x", true, true, false, "Marked", None))]
    fn a_python_scalar_keeps_its_type_and_value(#[case] sent: Value, #[case] read: Scalar) {
        let decoded: Scalar = serde_json::from_value(sent.clone()).unwrap();
        assert_eq!(decoded, read);
        assert_eq!(serde_json::to_value(&decoded).unwrap(), sent);
    }

    #[rstest]
    fn nan_crosses_as_a_float() {
        let decoded: Scalar = serde_json::from_value(json!(raw_json::encode("NaN"))).unwrap();
        assert!(matches!(decoded, Scalar::Float(value) if value.is_nan()));
        assert_eq!(
            serde_json::to_value(&decoded).unwrap(),
            json!(raw_json::encode("NaN"))
        );
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Cost {
        #[serde(default, with = "float")]
        value: Option<f64>,
    }

    #[rstest]
    #[case::finite(json!({"value": 1.5}), Some(1.5))]
    #[case::infinity(json!({"value": raw_json::encode("Infinity")}), Some(f64::INFINITY))]
    #[case::negative_infinity(json!({"value": raw_json::encode("-Infinity")}), Some(f64::NEG_INFINITY))]
    #[case::null(json!({"value": null}), None)]
    #[case::missing(json!({}), None)]
    fn a_float_member_keeps_infinity_both_ways(#[case] sent: Value, #[case] read: Option<f64>) {
        let decoded: Cost = serde_json::from_value(sent).unwrap();
        assert_eq!(decoded, Cost { value: read });
        let again: Cost = serde_json::from_value(serde_json::to_value(&decoded).unwrap()).unwrap();
        assert_eq!(again, decoded);
    }

    #[rstest]
    fn a_nan_float_member_crosses_as_the_marker() {
        let sent = json!({"value": raw_json::encode("NaN")});
        let decoded: Cost = serde_json::from_value(sent.clone()).unwrap();
        assert!(decoded.value.is_some_and(f64::is_nan));
        assert_eq!(serde_json::to_value(&decoded).unwrap(), sent);
    }

    #[rstest]
    #[case::int(json!({"value": 2}))]
    #[case::text(json!({"value": "1.5"}))]
    fn a_float_member_takes_only_a_float(#[case] sent: Value) {
        assert!(serde_json::from_value::<Cost>(sent).is_err());
    }

    #[rstest]
    fn a_stringified_value_never_equals_the_text_it_carries() {
        let stringified: Scalar = serde_json::from_value(
            json!({"text": "prod", "truthy": true, "hashable": true, "integer": false, "type_name": "Tag", "hash_error": null}),
        )
        .unwrap();
        assert_ne!(stringified, Scalar::Text("prod".to_owned()));
    }

    #[rstest]
    #[case::true_flag(Scalar::Bool(true), true)]
    #[case::false_flag(Scalar::Bool(false), false)]
    #[case::zero(Scalar::Int(0), false)]
    #[case::wide(Scalar::Int(10_i128.pow(20)), true)]
    #[case::zero_float(Scalar::Float(-0.0), false)]
    #[case::nan(Scalar::Float(f64::NAN), true)]
    #[case::empty_text(Scalar::Text(String::new()), false)]
    #[case::text(Scalar::Text("0".to_owned()), true)]
    #[case::empty_list(
        stringified("[]", false, false, false, "list", Some("unhashable type: 'list'")),
        false
    )]
    #[case::object(stringified("", true, true, false, "object", None), true)]
    fn truthiness_is_what_python_bool_gives(#[case] value: Scalar, #[case] truthy: bool) {
        assert_eq!(value.truthy(), truthy);
    }

    #[rstest]
    #[case::flag(Scalar::Bool(false), "bool", None)]
    #[case::int(Scalar::Int(7), "int", None)]
    #[case::float(Scalar::Float(1.5), "float", None)]
    #[case::text(Scalar::Text("prod".to_owned()), "str", None)]
    #[case::bytes(stringified("b'7'", true, true, false, "bytes", None), "bytes", None)]
    #[case::tuple_of_list(
        stringified(
            "('a', ['b'])",
            true,
            false,
            false,
            "tuple",
            Some("unhashable type: 'list'")
        ),
        "tuple",
        Some("unhashable type: 'list'")
    )]
    fn the_type_name_and_hash_error_are_what_python_names_in_its_errors(
        #[case] value: Scalar,
        #[case] type_name: &str,
        #[case] hash_error: Option<&str>,
    ) {
        assert_eq!(
            (value.type_name(), value.hash_error()),
            (type_name, hash_error)
        );
    }

    #[rstest]
    #[case::list(json!(["a1"]))]
    #[case::object(json!({"a": 1}))]
    #[case::stringified_without_truthiness(json!({"text": "[]", "hashable": false, "integer": false, "type_name": "list", "hash_error": null}))]
    #[case::stringified_without_text(json!({"truthy": false, "hashable": false, "integer": false, "type_name": "list", "hash_error": null}))]
    #[case::stringified_without_hashable(json!({"text": "[]", "truthy": false, "integer": false, "type_name": "list", "hash_error": null}))]
    #[case::stringified_without_integer(json!({"text": "[]", "truthy": false, "hashable": false, "type_name": "list", "hash_error": null}))]
    #[case::stringified_without_type_name(json!({"text": "[]", "truthy": false, "hashable": false, "integer": false, "hash_error": null}))]
    #[case::stringified_without_hash_error(json!({"text": "[]", "truthy": false, "hashable": false, "integer": false, "type_name": "list"}))]
    #[case::stringified_with_a_number_as_type_name(json!({"text": "[]", "truthy": false, "hashable": false, "integer": false, "type_name": 7, "hash_error": null}))]
    #[case::stringified_with_a_number_as_text(json!({"text": 7, "truthy": true, "hashable": true, "integer": false, "type_name": "list", "hash_error": null}))]
    #[case::stringified_with_another_member(json!({"text": "[]", "truthy": false, "hashable": false, "integer": false, "type_name": "list", "hash_error": null, "type": "list"}))]
    #[case::null(Value::Null)]
    #[case::beyond_i128(json!(raw_json::encode("1000000000000000000000000000000000000000")))]
    fn a_value_that_is_not_a_scalar_does_not_decode(#[case] sent: Value) {
        assert!(serde_json::from_value::<Scalar>(sent).is_err());
    }
}
