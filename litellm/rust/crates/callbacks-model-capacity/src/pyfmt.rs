use litellm_python_compat::{json, repr, truthy};
use serde_json::{Number, Value};

pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let formatted = format!("{value:e}");
        let (mantissa, exponent) = formatted.split_once('e').unwrap_or((&formatted, "0"));
        let (sign, digits) = exponent
            .strip_prefix('-')
            .map_or(("+", exponent), |digits| ("-", digits));
        return format!("{mantissa}e{sign}{digits:0>2}");
    }
    let formatted = format!("{value}");
    if formatted.contains('.') {
        formatted
    } else {
        format!("{formatted}.0")
    }
}

pub fn number(value: &Number) -> String {
    if value.is_i64() || value.is_u64() {
        return value.to_string();
    }
    value.as_f64().map_or_else(|| value.to_string(), float_repr)
}

pub fn value(value: Option<&Value>) -> String {
    match value {
        None => "None".into(),
        Some(value) => repr::to_str(&json::from_json(value.clone())),
    }
}

fn integral(number: &Number) -> Option<i128> {
    number
        .as_i64()
        .map(i128::from)
        .or_else(|| number.as_u64().map(i128::from))
}

fn numeric(value: &Value) -> Option<Result<i128, f64>> {
    match value {
        Value::Bool(flag) => Some(Ok(i128::from(*flag))),
        Value::Number(number) => {
            Some(integral(number).map_or_else(|| Err(number.as_f64().unwrap_or(f64::NAN)), Ok))
        }
        _ => None,
    }
}

fn int_equals_float(int: i128, float: f64) -> bool {
    float.is_finite() && float.fract() == 0.0 && float.abs() < 1.7e38 && float as i128 == int
}

pub fn equals(left: &Value, right: &Value) -> bool {
    match (numeric(left), numeric(right)) {
        (Some(Ok(a)), Some(Ok(b))) => a == b,
        (Some(Err(a)), Some(Err(b))) => a == b,
        (Some(Ok(int)), Some(Err(float))) | (Some(Err(float)), Some(Ok(int))) => {
            int_equals_float(int, float)
        }
        (Some(_), None) | (None, Some(_)) => false,
        (None, None) => left == right,
    }
}

pub fn numeric_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let chars: Vec<char> = trimmed.chars().collect();
    let digit = |index: Option<usize>| {
        index
            .and_then(|index| chars.get(index))
            .is_some_and(char::is_ascii_digit)
    };
    let misplaced = chars
        .iter()
        .enumerate()
        .any(|(index, ch)| *ch == '_' && !(digit(index.checked_sub(1)) && digit(Some(index + 1))));
    (!misplaced).then(|| trimmed.replace('_', ""))
}

pub fn str_repr(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{}'", text.replace('\'', "\\'"))
    }
}

pub fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

pub fn optional_int(value: Option<i64>) -> String {
    value.map_or_else(|| "None".into(), |value| value.to_string())
}

pub fn percent(value: f64, places: usize) -> String {
    let scaled = value * 100.0;
    if scaled.is_nan() {
        return "nan%".into();
    }
    if scaled.is_infinite() {
        return if scaled > 0.0 { "inf%" } else { "-inf%" }.into();
    }
    format!("{scaled:.places$}%")
}

pub fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(|value| truthy::truthy(&json::from_json(value.clone())))
}

pub fn is_latin1(text: &str) -> bool {
    text.chars().all(|character| (character as u32) < 256)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::grouped(" 1_000 ", Some("1000"))]
    #[case::plain("12.5", Some("12.5"))]
    #[case::leading("_1", None)]
    #[case::trailing("1_", None)]
    #[case::doubled("1__0", None)]
    #[case::beside_a_point("1_.5", None)]
    fn numbers_accept_underscores_only_between_digits_like_python(
        #[case] text: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(numeric_text(text).as_deref(), expected);
    }

    #[rstest]
    #[case(2.0, "2.0")]
    #[case(1.5, "1.5")]
    #[case(1_000_000.0, "1000000.0")]
    #[case(0.1, "0.1")]
    #[case(1e-5, "1e-05")]
    #[case(1e16, "1e+16")]
    #[case(-3.25, "-3.25")]
    fn floats_render_like_python_repr(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(float_repr(value), expected);
    }

    #[rstest]
    #[case::ints(json!(5), "5")]
    #[case::bool(json!(true), "True")]
    #[case::list(json!([1, "a", null, 1.5]), "[1, 'a', None, 1.5]")]
    #[case::dict(json!({"b": 1, "a": "it's"}), "{'b': 1, 'a': \"it's\"}")]
    #[case::text(json!("plain"), "plain")]
    fn values_render_like_python_str(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(super::value(Some(&value)), expected);
    }

    #[rstest]
    #[case::int_float(json!(1), json!(1.0), true)]
    #[case::bool_int(json!(true), json!(1), true)]
    #[case::text_int(json!("1"), json!(1), false)]
    #[case::null(Value::Null, Value::Null, true)]
    #[case::half(json!(0.5), json!(0), false)]
    fn keys_compare_like_python(#[case] left: Value, #[case] right: Value, #[case] expected: bool) {
        assert_eq!(equals(&left, &right), expected);
    }

    #[rstest]
    #[case::tenths(vec![0.1; 10], 1.0)]
    #[case::thirds(vec![0.3, 0.3, 0.3, 0.1], 1.0)]
    #[case::mixed(vec![0.6, 8.0 / 30.0, 0.3], 1.1666666666666665)]
    fn sums_compensate_like_python(#[case] values: Vec<f64>, #[case] expected: f64) {
        assert_eq!(sum(values).to_bits(), expected.to_bits());
    }

    #[rstest]
    #[case(0.8, 1, "80.0%")]
    #[case(0.123456, 2, "12.35%")]
    #[case(1.0 / 3.0, 1, "33.3%")]
    #[case(0.0, 2, "0.00%")]
    #[case(1.25, 1, "125.0%")]
    fn percents_render_like_python_format(
        #[case] value: f64,
        #[case] places: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(percent(value, places), expected);
    }
}
