//! `SafeConstructor` for the tags a config uses: a composed node becomes the `Value`
//! `yaml.safe_load` would return.

use litellm_python_compat::{BigInt, number};
use serde_yaml_ng::{Mapping, Number, Value};

use crate::{
    Error, Location,
    compose::{Kind, Node, Tag},
};

pub(crate) fn construct(node: Node) -> Result<Value, Error> {
    let Node { tag, kind, at, .. } = node;
    match (tag, kind) {
        (Tag::Map, Kind::Mapping(pairs)) => mapping(pairs).map(Value::Mapping),
        (Tag::Seq, Kind::Sequence(items)) => Ok(Value::Sequence(
            items.into_iter().map(construct).collect::<Result<_, _>>()?,
        )),
        (Tag::Null, Kind::Scalar(_)) => Ok(Value::Null),
        (Tag::Str, Kind::Scalar(text)) => Ok(Value::String(text)),
        (Tag::Bool, Kind::Scalar(text)) => match text.to_lowercase().as_str() {
            "yes" | "true" | "on" => Ok(Value::Bool(true)),
            "no" | "false" | "off" => Ok(Value::Bool(false)),
            _ => Err(invalid(Tag::Bool, text, at)),
        },
        (Tag::Int, Kind::Scalar(text)) => match int(&text) {
            Some(number) => integer(number).ok_or(Error::IntOutOfRange { text, at }),
            None => Err(invalid(Tag::Int, text, at)),
        },
        (Tag::Float, Kind::Scalar(text)) => match float(&text) {
            Some(number) => Ok(Value::Number(Number::from(number))),
            None => Err(invalid(Tag::Float, text, at)),
        },
        (tag @ (Tag::Null | Tag::Bool | Tag::Int | Tag::Float | Tag::Str), _) => {
            Err(Error::WrongNode {
                tag,
                expected: "scalar",
                at,
            })
        }
        (tag @ Tag::Map, _) => Err(Error::WrongNode {
            tag,
            expected: "mapping",
            at,
        }),
        (tag @ Tag::Seq, _) => Err(Error::WrongNode {
            tag,
            expected: "sequence",
            at,
        }),
        (tag @ (Tag::Merge | Tag::Other(_)), _) => Err(Error::UnsupportedTag { tag, at }),
    }
}

fn invalid(tag: Tag, text: String, at: Location) -> Error {
    Error::InvalidScalar { tag, text, at }
}

/// `construct_mapping` after `flatten_mapping`: a repeated key keeps its first position and its
/// last value, like a `dict`.
fn mapping(pairs: Vec<(Node, Node)>) -> Result<Mapping, Error> {
    flatten(pairs)?
        .into_iter()
        .try_fold(Mapping::new(), |mut mapping, (key, value)| {
            let at = key.at;
            let key = construct(key)?;
            if matches!(key, Value::Sequence(_) | Value::Mapping(_)) {
                return Err(Error::UnhashableKey { at });
            }
            mapping.insert(key, construct(value)?);
            Ok(mapping)
        })
}

/// `SafeConstructor.flatten_mapping`: the pairs of every `<<` mapping come first, an earlier
/// mapping in a `<<` list winning over a later one, then the mapping's own pairs, which win over
/// all of them.
fn flatten(pairs: Vec<(Node, Node)>) -> Result<Vec<(Node, Node)>, Error> {
    let (merges, own): (Vec<_>, Vec<_>) = pairs
        .into_iter()
        .partition(|(key, _)| key.tag == Tag::Merge);
    let merged = merges
        .into_iter()
        .map(|(_, source)| merge_source(source))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(merged.into_iter().flatten().chain(own).collect())
}

fn merge_source(source: Node) -> Result<Vec<(Node, Node)>, Error> {
    match source.kind {
        Kind::Mapping(pairs) => flatten(pairs),
        Kind::Sequence(items) => {
            let sources = items
                .into_iter()
                .map(|item| match item.kind {
                    Kind::Mapping(pairs) => flatten(pairs),
                    _ => Err(Error::MergeNotMapping { at: item.at }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(sources.into_iter().rev().flatten().collect())
        }
        Kind::Scalar(_) => Err(Error::MergeNotMapping { at: source.at }),
    }
}

fn integer(number: BigInt) -> Option<Value> {
    if let Ok(small) = i64::try_from(&number) {
        return Some(Value::Number(small.into()));
    }
    u64::try_from(&number)
        .ok()
        .map(|large| Value::Number(large.into()))
}

/// `SafeConstructor.construct_yaml_int`.
fn int(text: &str) -> Option<BigInt> {
    let cleaned = text.replace('_', "");
    let (negative, body) = signed(&cleaned);
    let magnitude = if body == "0" {
        BigInt::from(0)
    } else if let Some(digits) = body.strip_prefix("0b") {
        BigInt::parse_bytes(digits.as_bytes(), 2)?
    } else if let Some(digits) = body.strip_prefix("0x") {
        BigInt::parse_bytes(digits.as_bytes(), 16)?
    } else if body.starts_with('0') {
        BigInt::parse_bytes(body.as_bytes(), 8)?
    } else if body.contains(':') {
        body.split(':').try_fold(BigInt::from(0), |total, part| {
            Some(total * 60 + number::int(part)?)
        })?
    } else {
        number::int(body)?
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// `SafeConstructor.construct_yaml_float`.
fn float(text: &str) -> Option<f64> {
    let cleaned = text.replace('_', "").to_lowercase();
    let (negative, body) = signed(&cleaned);
    let magnitude = match body {
        ".inf" => f64::INFINITY,
        ".nan" => return Some(f64::NAN),
        _ if body.contains(':') => body
            .split(':')
            .try_fold(0.0, |total, part| Some(total * 60.0 + number::float(part)?))?,
        _ => number::float(body)?,
    };
    Some(if negative { -magnitude } else { magnitude })
}

fn signed(text: &str) -> (bool, &str) {
    match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    }
}
