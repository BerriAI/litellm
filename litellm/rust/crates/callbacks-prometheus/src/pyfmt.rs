use serde_json::Value;

pub const OPAQUE_KEY: &str = "$py";
pub const NUMBER_KEY: &str = "$num";
pub const TRUTHY_KEY: &str = "$truthy";
pub const TYPE_KEY: &str = "$type";

pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let scientific = format!("{value:e}");
    let (sign, unsigned) = scientific
        .strip_prefix('-')
        .map_or(("", scientific.as_str()), |rest| ("-", rest));
    let (mantissa, exponent) = unsigned.split_once('e').unwrap_or((unsigned, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let decpt = exponent + 1;
    let count = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let body = if !(-3..=16).contains(&decpt) {
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() {
            String::new()
        } else {
            format!(".{rest}")
        };
        let exp = decpt - 1;
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{first}{fraction}e{exp_sign}{:02}", exp.abs())
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat(decpt.unsigned_abs() as usize))
    } else if decpt >= count {
        format!(
            "{digits}{}.0",
            "0".repeat(usize::try_from(decpt - count).unwrap_or(0))
        )
    } else {
        let (whole, fraction) = digits.split_at(usize::try_from(decpt).unwrap_or(0));
        format!("{whole}.{fraction}")
    };
    format!("{sign}{body}")
}

pub fn float_to_go_string(value: f64) -> String {
    if value == f64::INFINITY {
        return "+Inf".into();
    }
    if value == f64::NEG_INFINITY {
        return "-Inf".into();
    }
    if value.is_nan() {
        return "NaN".into();
    }
    let repr = float_repr(value);
    match repr.find('.') {
        Some(dot) if value > 0.0 && dot > 6 => {
            let mantissa = format!("{}.{}{}", &repr[..1], &repr[1..dot], &repr[dot + 1..]);
            let mantissa = mantissa.trim_end_matches(['0', '.']);
            format!("{mantissa}e+0{}", dot - 1)
        }
        _ => repr,
    }
}

fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

fn opaque(value: &Value) -> Option<&str> {
    let Value::Object(map) = value else {
        return None;
    };
    let text = map.get(OPAQUE_KEY)?.as_str()?;
    map.iter()
        .all(|(key, item)| match key.as_str() {
            OPAQUE_KEY => true,
            TRUTHY_KEY => item.is_boolean(),
            TYPE_KEY => item.is_string(),
            _ => false,
        })
        .then_some(text)
}

pub fn is_opaque(value: &Value) -> bool {
    opaque(value).is_some() || tagged_number(value).is_some()
}

pub fn stringified(text: &str, truthy: bool, type_name: &str) -> Value {
    let mut members = serde_json::Map::from_iter([
        (OPAQUE_KEY.to_owned(), Value::from(text)),
        (TYPE_KEY.to_owned(), Value::from(type_name)),
    ]);
    if !truthy {
        members.insert(TRUTHY_KEY.to_owned(), Value::Bool(false));
    }
    Value::Object(members)
}

fn tagged_number(value: &Value) -> Option<&str> {
    match value {
        Value::Object(map) if map.len() == 1 => map.get(NUMBER_KEY).and_then(Value::as_str),
        _ => None,
    }
}

fn is_int_text(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn py_repr(value: &Value) -> String {
    match value {
        Value::String(text) => str_repr(text),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => match opaque(value).or_else(|| tagged_number(value)) {
            Some(text) => text.to_owned(),
            None => format!(
                "{{{}}}",
                map.iter()
                    .map(|(key, item)| format!("{}: {}", str_repr(key), py_repr(item)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        },
        other => py_str(other),
    }
}

pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => match (number.as_i64(), number.as_u64()) {
            (Some(int), _) => int.to_string(),
            (None, Some(int)) => int.to_string(),
            _ => float_repr(number.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(text) => text.clone(),
        Value::Array(_) | Value::Object(_) => py_repr(value),
    }
}

pub fn is_py_int(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.is_i64() || number.is_u64(),
        other => tagged_number(other).is_some_and(is_int_text),
    }
}

pub fn is_py_number(value: &Value) -> bool {
    matches!(value, Value::Number(_)) || tagged_number(value).is_some()
}

pub fn py_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        other => tagged_number(other).and_then(|text| text.parse().ok()),
    }
}

pub fn py_big_int(value: &Value) -> Option<i128> {
    tagged_number(value)
        .filter(|text| is_int_text(text))
        .and_then(|text| text.parse().ok())
}

pub fn py_type_name(value: &Value) -> &str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) if is_py_int(value) => "int",
        Value::Number(_) => "float",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(map) if opaque(value).is_some() => map
            .get(TYPE_KEY)
            .and_then(Value::as_str)
            .unwrap_or("object"),
        Value::Object(_) if is_py_int(value) => "int",
        Value::Object(_) if tagged_number(value).is_some() => "float",
        Value::Object(_) => "dict",
    }
}

pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) if opaque(value).is_some() => {
            map.get(TRUTHY_KEY) != Some(&Value::Bool(false))
        }
        Value::Object(map) => !map.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case(0.0, "0.0")]
    #[case(-0.0, "-0.0")]
    #[case(1.0, "1.0")]
    #[case(0.005, "0.005")]
    #[case(0.0001, "0.0001")]
    #[case(0.00001, "1e-05")]
    #[case(1.5e-7, "1.5e-07")]
    #[case(1e15, "1000000000000000.0")]
    #[case(1e16, "1e+16")]
    #[case(1.234_567_8e16, "1.2345678e+16")]
    #[case(123.456, "123.456")]
    #[case(9_223_372_036_854_775_807.0, "9.223372036854776e+18")]
    #[case(1e300, "1e+300")]
    #[case(0.1 + 0.2, "0.30000000000000004")]
    fn floats_print_as_python_repr(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case(f64::INFINITY, "+Inf")]
    #[case(f64::NAN, "NaN")]
    #[case(1.0, "1.0")]
    #[case(600.0, "600.0")]
    #[case(1_234_567.0, "1.234567e+06")]
    #[case(10_000_000_000.0, "1e+010")]
    #[case(0.025, "0.025")]
    fn floats_print_as_the_prometheus_client_go_string(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_to_go_string(value), expected);
    }

    #[rstest]
    #[case(json!(null), "None")]
    #[case(json!(true), "True")]
    #[case(json!(42), "42")]
    #[case(json!(2.5), "2.5")]
    #[case(json!(3.0), "3.0")]
    #[case(json!("x"), "x")]
    #[case(json!(["a", 1, null]), "['a', 1, None]")]
    #[case(json!({"k": "it's"}), "{'k': \"it's\"}")]
    #[case(json!({"$py": "<obj>"}), "<obj>")]
    #[case(json!({"$num": "nan"}), "nan")]
    #[case(json!({"$num": "-inf"}), "-inf")]
    #[case(json!({"$num": "100000000000000000000"}), "100000000000000000000")]
    #[case(json!([{"$num": "inf"}]), "[inf]")]
    fn json_values_print_as_python_str(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(py_str(&value), expected);
    }

    #[rstest]
    #[case::finite(json!(2.5), Some(2.5), "float")]
    #[case::positive_infinity(json!({"$num": "inf"}), Some(f64::INFINITY), "float")]
    #[case::negative_infinity(json!({"$num": "-inf"}), Some(f64::NEG_INFINITY), "float")]
    #[case::big_int(json!({"$num": "100000000000000000000"}), Some(1e20), "int")]
    #[case::opaque(json!({"$py": "inf"}), None, "object")]
    #[case::typed_opaque(json!({"$py": "[1]", "$type": "list"}), None, "list")]
    #[case::falsy_typed_opaque(json!({"$py": "[]", "$type": "list", "$truthy": false}), None, "list")]
    #[case::text(json!("inf"), None, "str")]
    fn tagged_numbers_read_as_python_numbers(
        #[case] value: Value,
        #[case] expected: Option<f64>,
        #[case] type_name: &str,
    ) {
        assert_eq!(py_number(&value), expected);
        assert_eq!(py_type_name(&value), type_name);
    }

    #[rstest]
    fn a_tagged_nan_reads_as_nan() {
        assert!(py_number(&json!({"$num": "nan"})).is_some_and(f64::is_nan));
    }
}
