use serde_json::Value;

pub fn raw_scalar(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => litellm_host::raw_json::decode(text),
        _ => None,
    }
}

pub fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_i64() || number.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => match raw_scalar(value) {
            Some(raw) if raw.starts_with('"') => "str",
            Some(raw) if raw.contains(['.', 'e', 'E', 'N', 'I']) => "float",
            Some(_) => "int",
            None => "str",
        },
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn saturated(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_u64().map(|_| i64::MAX)),
        _ => {
            let raw = raw_scalar(value)?;
            let digits = raw.strip_prefix('-').unwrap_or(raw);
            (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())).then(|| {
                if raw.starts_with('-') {
                    i64::MIN
                } else {
                    i64::MAX
                }
            })
        }
    }
}

pub fn saturating_int<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
    saturated(&value)
        .ok_or_else(|| serde::de::Error::custom(format!("an int was expected, not {value}")))
}

pub fn int_as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) if number.is_i64() || number.is_u64() => number.as_f64(),
        _ => {
            let raw = raw_scalar(value)?;
            let digits = raw.strip_prefix('-').unwrap_or(raw);
            (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| raw.parse().ok())
                .flatten()
        }
    }
}

pub fn saturating_optional_int<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<i64>, D::Error> {
    let value = <Value as serde::Deserialize>::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(None);
    }
    saturated(&value)
        .map(Some)
        .ok_or_else(|| serde::de::Error::custom(format!("an int was expected, not {value}")))
}

const STRINGIFIED: &str = "\u{F8FF}litellm-stringified";
const STRINGIFIED_MEMBERS: [&str; 4] = ["text", "truthy", "hashable", "integer"];
const STRINGIFIED_EXTRAS: [&str; 2] = ["type_name", "hash_error"];

pub struct Stringified<'a> {
    pub text: &'a str,
    pub truthy: bool,
    pub hashable: bool,
    pub hash_error: Option<&'a str>,
    pub type_name: Option<&'a str>,
}

pub fn stringified(value: &Value) -> Option<Stringified<'_>> {
    let Value::Object(members) = value.as_object()?.get(STRINGIFIED)? else {
        return None;
    };
    Some(Stringified {
        text: members.get("text")?.as_str()?,
        truthy: members.get("truthy")?.as_bool()?,
        hashable: members.get("hashable")?.as_bool()?,
        hash_error: members.get("hash_error").and_then(Value::as_str),
        type_name: members.get("type_name").and_then(Value::as_str),
    })
}

pub fn scalar_member(value: Value) -> Value {
    let Value::Object(members) = value else {
        return value;
    };
    let shaped = STRINGIFIED_MEMBERS
        .iter()
        .all(|name| members.contains_key(*name))
        && members.keys().all(|name| {
            STRINGIFIED_MEMBERS.contains(&name.as_str())
                || STRINGIFIED_EXTRAS.contains(&name.as_str())
        });
    match (shaped, members.get("integer"), members.get("text")) {
        (true, Some(Value::Bool(true)), Some(Value::String(text))) => {
            Value::String(litellm_host::raw_json::encode(text))
        }
        (true, _, _) => Value::Object(serde_json::Map::from_iter([(
            STRINGIFIED.to_owned(),
            Value::Object(members),
        )])),
        (false, _, _) => Value::Object(members),
    }
}

pub fn hash_error(value: &Value) -> Option<String> {
    let found = stringified(value)?;
    if found.hashable {
        return None;
    }
    Some(found.hash_error.unwrap_or("unhashable type").to_owned())
}
