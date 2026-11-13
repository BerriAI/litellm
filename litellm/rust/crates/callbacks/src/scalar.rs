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
    Stringified { text: String, truthy: bool },
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
            Self::Stringified { text, truthy } => {
                let mut members = serializer.serialize_struct("Stringified", 2)?;
                members.serialize_field("text", &TextOut(text))?;
                members.serialize_field("truthy", truthy)?;
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

const STRINGIFIED: &[&str] = &["text", "truthy"];

struct ScalarVisitor;

impl<'de> Visitor<'de> for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a bool, an integer, a float, a string or a stringified value")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut members: A) -> Result<Scalar, A::Error> {
        let mut text = None;
        let mut truthy = None;
        while let Some(name) = members.next_key::<String>()? {
            match name.as_str() {
                "text" => text = Some(members.next_value::<TextIn>()?.0),
                "truthy" => truthy = Some(members.next_value::<bool>()?),
                other => return Err(de::Error::unknown_field(other, STRINGIFIED)),
            }
        }
        Ok(Scalar::Stringified {
            text: text.ok_or_else(|| de::Error::missing_field("text"))?,
            truthy: truthy.ok_or_else(|| de::Error::missing_field("truthy"))?,
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
    #[case::stringified_list(json!({"text": "['a1']", "truthy": true}), Scalar::Stringified { text: "['a1']".to_owned(), truthy: true })]
    #[case::stringified_empty_set(json!({"text": "set()", "truthy": false}), Scalar::Stringified { text: "set()".to_owned(), truthy: false })]
    #[case::stringified_marked_text(json!({"text": raw_json::encode(r#""\uf8fflitellm-raw-json:x""#), "truthy": true}), Scalar::Stringified { text: "\u{f8ff}litellm-raw-json:x".to_owned(), truthy: true })]
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
        let stringified: Scalar =
            serde_json::from_value(json!({"text": "prod", "truthy": true})).unwrap();
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
    #[case::empty_list(Scalar::Stringified { text: "[]".to_owned(), truthy: false }, false)]
    #[case::object(Scalar::Stringified { text: String::new(), truthy: true }, true)]
    fn truthiness_is_what_python_bool_gives(#[case] value: Scalar, #[case] truthy: bool) {
        assert_eq!(value.truthy(), truthy);
    }

    #[rstest]
    #[case::list(json!(["a1"]))]
    #[case::object(json!({"a": 1}))]
    #[case::stringified_without_truthiness(json!({"text": "[]"}))]
    #[case::stringified_without_text(json!({"truthy": false}))]
    #[case::stringified_with_a_number_as_text(json!({"text": 7, "truthy": true}))]
    #[case::stringified_with_another_member(json!({"text": "[]", "truthy": false, "type": "list"}))]
    #[case::null(Value::Null)]
    #[case::beyond_i128(json!(raw_json::encode("1000000000000000000000000000000000000000")))]
    fn a_value_that_is_not_a_scalar_does_not_decode(#[case] sent: Value) {
        assert!(serde_json::from_value::<Scalar>(sent).is_err());
    }
}
