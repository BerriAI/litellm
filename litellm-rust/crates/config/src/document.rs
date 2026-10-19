use std::{collections::BTreeMap, fmt, sync::LazyLock};

use regex::Regex;
use serde_yaml_ng::{Mapping, Value};
use yaml_rust2::{
    parser::{Event, Parser},
    scanner::TScalarStyle,
};

use crate::{Config, Error};

static INTEGER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[-+]?(?:0b[0-1_]+|0[0-7_]+|0|[1-9][0-9_]*|0x[0-9a-fA-F_]+|[1-9][0-9_]*(?::[0-5]?[0-9])+)$").unwrap()
});
static FLOAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$").unwrap()
});
static TIMESTAMP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:$|[Tt \t])").unwrap());

#[derive(Clone)]
pub struct Document {
    value: Value,
    python_compatible: bool,
}

#[derive(Default)]
struct Anchors {
    values: BTreeMap<usize, Value>,
    present: bool,
    expanded_size: usize,
}

impl Anchors {
    fn account(&mut self, size: usize) -> Result<(), Error> {
        self.expanded_size = self
            .expanded_size
            .checked_add(size)
            .ok_or(Error::UnsupportedYaml)?;
        if self.expanded_size > 16 * 1024 * 1024 {
            return Err(Error::UnsupportedYaml);
        }
        Ok(())
    }
}

impl Document {
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        let mut parser = Parser::new_from_str(yaml);
        parser.next_token()?;
        let event = parser.next_token()?.0;
        if event == Event::StreamEnd {
            return Ok(Self {
                value: Value::Null,
                python_compatible: true,
            });
        }
        let first = parser.next_token()?.0;
        let mut anchors = Anchors::default();
        let value = node(&mut parser, first, 0, &mut anchors)?;
        if parser.next_token()?.0 != Event::DocumentEnd
            || parser.next_token()?.0 != Event::StreamEnd
        {
            return Err(Error::UnsupportedYaml);
        }
        Ok(Self {
            value,
            python_compatible: !anchors.present,
        })
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        Ok(Self {
            value: crate::includes::load(path.as_ref())?,
            python_compatible: false,
        })
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn typed(&self) -> Result<Config, Error> {
        Ok(serde_yaml_ng::from_value(self.value.clone())?)
    }

    pub fn python_compatible(&self) -> bool {
        self.python_compatible
    }
}

impl fmt::Debug for Document {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Document").finish_non_exhaustive()
    }
}

fn node<I: Iterator<Item = char>>(
    parser: &mut Parser<I>,
    event: Event,
    depth: usize,
    anchors: &mut Anchors,
) -> Result<Value, Error> {
    if depth > 128 {
        return Err(Error::UnsupportedYaml);
    }
    anchors.account(128)?;
    let anchor = match &event {
        Event::Scalar(_, _, anchor, _)
        | Event::SequenceStart(anchor, _)
        | Event::MappingStart(anchor, _) => *anchor,
        _ => 0,
    };
    let value = match event {
        Event::Alias(anchor) => {
            let value = anchors.values.get(&anchor).ok_or(Error::UnsupportedYaml)?;
            let size = value_size(value);
            anchors.account(size)?;
            Ok(anchors.values[&anchor].clone())
        }
        Event::Scalar(text, style, _, None) => {
            anchors.account(text.len())?;
            if style == TScalarStyle::Plain {
                scalar(text)
            } else {
                Ok(Value::String(text))
            }
        }
        Event::SequenceStart(_, None) => {
            let mut values = Vec::new();
            loop {
                let event = parser.next_token()?.0;
                if event == Event::SequenceEnd {
                    break Ok(Value::Sequence(values));
                }
                values.push(node(parser, event, depth + 1, anchors)?);
            }
        }
        Event::MappingStart(_, None) => {
            let mut values = Mapping::new();
            loop {
                let event = parser.next_token()?.0;
                if event == Event::MappingEnd {
                    break Ok(Value::Mapping(values));
                }
                let key = node(parser, event, depth + 1, anchors)?;
                if !matches!(&key, Value::String(text) if text != "<<") {
                    return Err(Error::UnsupportedYaml);
                }
                let event = parser.next_token()?.0;
                let value = node(parser, event, depth + 1, anchors)?;
                values.insert(key, value);
            }
        }
        _ => Err(Error::UnsupportedYaml),
    }?;
    if anchor != 0 {
        anchors.present = true;
        anchors.account(value_size(&value))?;
        anchors.values.insert(anchor, value.clone());
    }
    Ok(value)
}

fn value_size(value: &Value) -> usize {
    128 + match value {
        Value::String(text) => text.len(),
        Value::Sequence(values) => values.iter().map(value_size).sum(),
        Value::Mapping(values) => values
            .iter()
            .map(|(key, value)| value_size(key) + value_size(value))
            .sum(),
        _ => 0,
    }
}

fn scalar(text: String) -> Result<Value, Error> {
    match text.as_str() {
        "" | "~" | "null" | "Null" | "NULL" => return Ok(Value::Null),
        "yes" | "Yes" | "YES" | "true" | "True" | "TRUE" | "on" | "On" | "ON" => {
            return Ok(Value::Bool(true));
        }
        "no" | "No" | "NO" | "false" | "False" | "FALSE" | "off" | "Off" | "OFF" => {
            return Ok(Value::Bool(false));
        }
        _ => {}
    }
    if INTEGER.is_match(&text) {
        return integer(&text);
    }
    if FLOAT.is_match(&text) {
        return float(&text);
    }
    if TIMESTAMP.is_match(&text) || matches!(text.as_str(), "=" | "!") {
        return Err(Error::UnsupportedYaml);
    }
    Ok(Value::String(text))
}

fn integer(text: &str) -> Result<Value, Error> {
    let normalized = text.replace('_', "");
    let unsigned = normalized.trim_start_matches(['-', '+']);
    let magnitude = if let Some(binary) = unsigned.strip_prefix("0b") {
        u64::from_str_radix(binary, 2).ok()
    } else if let Some(hex) = unsigned.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else if unsigned.contains(':') {
        unsigned.split(':').try_fold(0_u64, |base, part| {
            base.checked_mul(60)?.checked_add(part.parse().ok()?)
        })
    } else if unsigned.starts_with('0') && unsigned.len() > 1 {
        u64::from_str_radix(unsigned, 8).ok()
    } else {
        unsigned.parse().ok()
    }
    .ok_or(Error::UnsupportedYaml)?;
    if normalized.starts_with('-') {
        let signed = i128::from(magnitude)
            .checked_neg()
            .and_then(|n| i64::try_from(n).ok());
        return signed.map(Value::from).ok_or(Error::UnsupportedYaml);
    }
    Ok(Value::from(magnitude))
}

fn float(text: &str) -> Result<Value, Error> {
    let normalized = text.replace('_', "").to_ascii_lowercase();
    let unsigned = normalized.trim_start_matches(['-', '+']);
    let magnitude = match unsigned {
        ".inf" => f64::INFINITY,
        ".nan" => f64::NAN,
        value if value.contains(':') => {
            if value.bytes().filter(|byte| *byte == b':').count() >= 8 {
                return Err(Error::UnsupportedYaml);
            }
            value
                .rsplit(':')
                .try_fold((1.0, 0.0), |(base, total), part| {
                    part.parse::<f64>()
                        .map(|digit| (base * 60.0, total + digit * base))
                })
                .map(|(_, total)| total)
                .map_err(|_| Error::UnsupportedYaml)?
        }
        value => value.parse().map_err(|_| Error::UnsupportedYaml)?,
    };
    Ok(Value::from(if normalized.starts_with('-') {
        -magnitude
    } else {
        magnitude
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::boolean("on", Value::Bool(true))]
    #[case::disabled("OFF", Value::Bool(false))]
    #[case::octal("012", Value::from(10))]
    #[case::binary("0b1_010", Value::from(10))]
    #[case::hex("-0xA", Value::from(-10))]
    #[case::sexagesimal("1:02:03", Value::from(3723))]
    #[case::float("1.0e+3", Value::from(1000.0))]
    #[case::scientific_string("1e3", Value::String("1e3".into()))]
    #[case::exponent_string("1.0e3", Value::String("1.0e3".into()))]
    #[case::invalid_octal("08", Value::String("08".into()))]
    #[case::yaml12_octal("0o12", Value::String("0o12".into()))]
    #[case::quoted("'on'", Value::String("on".into()))]
    #[case::null("null", Value::Null)]
    fn parses_config_scalars(#[case] source: &str, #[case] expected: Value) {
        assert_eq!(Document::from_yaml(source).unwrap().value(), &expected);
    }

    #[rstest]
    #[case::date("created: 2026-01-02")]
    #[case::binary("key: !!binary aGVsbG8=")]
    #[case::large_integer("key: 18446744073709551616")]
    #[case::non_string_key("1: one")]
    #[case::multiple_documents("---\none\n---\ntwo")]
    fn declines_values_requiring_python_compatibility(#[case] source: &str) {
        assert!(matches!(
            Document::from_yaml(source),
            Err(Error::UnsupportedYaml)
        ));
    }

    #[rstest]
    fn typed_native_config_accepts_aliases_while_python_keeps_their_identity() {
        let document = Document::from_yaml(
            "settings: &settings {store_model_in_db: on}\ngeneral_settings: *settings\n",
        )
        .unwrap();
        assert_eq!(
            document.typed().unwrap().general_settings.store_model_in_db,
            Some(true)
        );
        assert!(!document.python_compatible());
    }

    #[rstest]
    fn bounds_alias_expansion() {
        let source = std::iter::once("v0: &v0 [value]".to_owned())
            .chain(
                (1..20)
                    .map(|index| format!("v{index}: &v{index} [*v{}, *v{}]", index - 1, index - 1)),
            )
            .collect::<Vec<_>>()
            .join("\n");
        assert!(matches!(
            Document::from_yaml(&source),
            Err(Error::UnsupportedYaml)
        ));
    }

    #[rstest]
    fn bounds_alias_string_expansion() {
        let source = format!(
            "value: &value '{}'\nexpanded: [{}]",
            "s".repeat(1024 * 1024),
            "*value,".repeat(20)
        );
        assert!(matches!(
            Document::from_yaml(&source),
            Err(Error::UnsupportedYaml)
        ));
    }

    #[rstest]
    fn retains_the_document_independently_of_typed_defaults() {
        let document = Document::from_yaml("general_settings: {store_model_in_db: false, max_parallel_requests: 0}\nfuture: null\nprivate: os.environ/UNRESOLVED\n").unwrap();
        let config = document.typed().unwrap();
        assert_eq!(config.general_settings.store_model_in_db, Some(false));
        assert_eq!(config.general_settings.max_parallel_requests, Some(0));
        assert!(config.model_list.is_empty());
        assert_eq!(document.value().as_mapping().unwrap().len(), 3);
        assert_eq!(document.value()["future"], Value::Null);
        assert_eq!(
            document.value()["private"],
            Value::String("os.environ/UNRESOLVED".into())
        );
        assert!(!format!("{document:?}").contains("UNRESOLVED"));
    }

    #[rstest]
    fn duplicate_keys_replace_values_without_changing_order() {
        let document = Document::from_yaml("first: 1\nsecond: 2\nfirst: 3").unwrap();
        assert_eq!(document.value()["first"], Value::from(3));
        assert_eq!(
            document
                .value()
                .as_mapping()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            [
                &Value::String("first".into()),
                &Value::String("second".into())
            ]
        );
    }
}
