use crate::{Value, repr::repr};

const MAX_LEVEL: usize = 6;
const MAX_TUPLE: usize = 6;
const MAX_LIST: usize = 6;
const MAX_SET: usize = 6;
const MAX_DICT: usize = 4;
const MAX_STRING: usize = 30;
const MAX_LONG: usize = 40;
const MAX_OTHER: usize = 30;
const FILL: &str = "...";

pub fn reprlib(value: &Value) -> String {
    repr1(value, MAX_LEVEL)
}

fn repr1(value: &Value, level: usize) -> String {
    match value {
        Value::Str(text) => repr_str(text),
        Value::Int(_) => shortened(repr(value), MAX_LONG),
        Value::Tuple(values) => repr_iterable(values, level, ("(", ")"), MAX_TUPLE, ","),
        Value::List(values) => repr_iterable(values, level, ("[", "]"), MAX_LIST, ""),
        Value::Set(values) if values.is_empty() => "set()".to_owned(),
        Value::Set(values) => repr_iterable(&sorted(values), level, ("{", "}"), MAX_SET, ""),
        Value::Dict(entries) => repr_dict(entries, level),
        Value::None
        | Value::Bool(_)
        | Value::Float(_)
        | Value::Complex { .. }
        | Value::Bytes(_) => shortened(repr(value), MAX_OTHER),
    }
}

fn shortened(text: String, limit: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text;
    }
    let head = (limit - FILL.len()) / 2;
    let tail = limit - FILL.len() - head;
    let mut out: String = chars[..head].iter().collect();
    out.push_str(FILL);
    out.extend(&chars[chars.len() - tail..]);
    out
}

fn repr_str(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let quoted = repr(&Value::Str(chars.iter().take(MAX_STRING).collect()));
    if quoted.chars().count() <= MAX_STRING {
        return quoted;
    }
    let head = (MAX_STRING - FILL.len()) / 2;
    let tail = MAX_STRING - FILL.len() - head;
    let kept: String = chars[..head.min(chars.len())]
        .iter()
        .chain(&chars[chars.len().saturating_sub(tail)..])
        .collect();
    let quoted: Vec<char> = repr(&Value::Str(kept)).chars().collect();
    let mut out: String = quoted[..head].iter().collect();
    out.push_str(FILL);
    out.extend(&quoted[quoted.len() - tail..]);
    out
}

fn repr_iterable(
    values: &[Value],
    level: usize,
    (left, right): (&str, &str),
    max_items: usize,
    trail: &str,
) -> String {
    if level == 0 && !values.is_empty() {
        return format!("{left}{FILL}{right}");
    }
    let mut pieces: Vec<String> = values
        .iter()
        .take(max_items)
        .map(|value| repr1(value, level - 1))
        .collect();
    if values.len() > max_items {
        pieces.push(FILL.to_owned());
    }
    let trail = if values.len() == 1 { trail } else { "" };
    format!("{left}{}{trail}{right}", pieces.join(", "))
}

fn repr_dict(entries: &[(Value, Value)], level: usize) -> String {
    if entries.is_empty() {
        return "{}".to_owned();
    }
    if level == 0 {
        return format!("{{{FILL}}}");
    }
    let mut pieces: Vec<String> = sorted_entries(entries)
        .into_iter()
        .take(MAX_DICT)
        .map(|(key, value)| format!("{}: {}", repr1(key, level - 1), repr1(value, level - 1)))
        .collect();
    if entries.len() > MAX_DICT {
        pieces.push(FILL.to_owned());
    }
    format!("{{{}}}", pieces.join(", "))
}

fn sorted_entries(entries: &[(Value, Value)]) -> Vec<&(Value, Value)> {
    let mut ordered: Vec<&(Value, Value)> = entries.iter().collect();
    if entries.iter().all(|(key, _)| matches!(key, Value::Str(_))) {
        ordered.sort_by(|(left, _), (right, _)| match (left, right) {
            (Value::Str(left), Value::Str(right)) => left.cmp(right),
            _ => std::cmp::Ordering::Equal,
        });
    }
    ordered
}

fn sorted(values: &[Value]) -> Vec<Value> {
    let mut ordered = values.to_vec();
    if values.iter().all(|value| matches!(value, Value::Str(_))) {
        ordered.sort_by(|left, right| match (left, right) {
            (Value::Str(left), Value::Str(right)) => left.cmp(right),
            _ => std::cmp::Ordering::Equal,
        });
    }
    ordered
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn list(values: impl IntoIterator<Item = Value>) -> Value {
        Value::List(values.into_iter().collect())
    }

    fn dict(entries: &[(&str, Value)]) -> Value {
        Value::Dict(
            entries
                .iter()
                .map(|(key, value)| (Value::from(*key), value.clone()))
                .collect(),
        )
    }

    fn nested(depth: usize) -> Value {
        (0..depth).fold(list([Value::from(1)]), |inner, _| list([inner]))
    }

    #[rstest]
    #[case::none(Value::None, "None")]
    #[case::bool(Value::Bool(true), "True")]
    #[case::int(Value::from(3), "3")]
    #[case::wide_int(Value::Int("1".repeat(45).parse().unwrap()), "111111111111111111...1111111111111111111")]
    #[case::float(Value::Float(0.1), "0.1")]
    #[case::long_float(Value::Float(1.0 / 3.0), "0.3333333333333333")]
    #[case::nan(Value::Float(f64::NAN), "nan")]
    #[case::short_str(Value::from("high"), "'high'")]
    #[case::long_str(
        Value::from("abcdefghijklmnopqrstuvwxyz0123456789"),
        "'abcdefghijkl...xyz0123456789'"
    )]
    #[case::quoted_str(
        Value::from("it's a fairly long string with a quote"),
        "\"it's a fairl... with a quote\""
    )]
    #[case::list(list([Value::from(1)]), "[1]")]
    #[case::long_list(list((0..8).map(Value::from)), "[0, 1, 2, 3, 4, 5, ...]")]
    #[case::one_tuple(Value::Tuple(vec![Value::from(1)]), "(1,)")]
    #[case::sorted_dict(dict(&[("b", Value::from(2)), ("a", Value::from(1))]), "{'a': 1, 'b': 2}")]
    #[case::long_dict(dict(&[("e", Value::None), ("d", Value::None), ("c", Value::None), ("b", Value::None), ("a", Value::None)]), "{'a': None, 'b': None, 'c': None, 'd': None, ...}")]
    #[case::empty_dict(dict(&[]), "{}")]
    #[case::deep(nested(6), "[[[[[[[...]]]]]]]")]
    #[case::deep_dict(dict(&[("a", dict(&[("b", dict(&[("c", dict(&[("d", dict(&[("e", dict(&[("f", dict(&[("g", Value::None)]))]))]))]))]))]))]), "{'a': {'b': {'c': {'d': {'e': {'f': {...}}}}}}}")]
    fn values_shorten_as_reprlib_shortens_them(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(reprlib(&value), expected);
    }
}
