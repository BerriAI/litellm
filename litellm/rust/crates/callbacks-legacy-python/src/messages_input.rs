use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList},
};
use serde::ser::{Error as _, Serialize, SerializeMap, Serializer};
use serde_json::{Map, Value};

use crate::error::Error;

struct Ordered<'a> {
    order: &'a [String],
    sent: &'a Map<String, Value>,
    as_called: &'a Map<String, Value>,
}

impl Serialize for Ordered<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        for name in self.order {
            if let Some(value) = self.as_called.get(name).or_else(|| self.sent.get(name)) {
                map.serialize_entry(name, value)?;
            }
        }
        for (name, value) in self.sent {
            if !self.order.contains(name) {
                map.serialize_entry(name, value)?;
            }
        }
        map.end()
    }
}

pub(crate) fn is_caller_scalar(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
}

pub(crate) fn serialized(
    order: &[String],
    sent: &Value,
    as_called: &Map<String, Value>,
) -> Result<String, Error> {
    let bytes = match sent {
        Value::Object(sent) => litellm_host::raw_json::to_vec(&Ordered {
            order,
            sent,
            as_called,
        }),
        other => litellm_host::raw_json::to_vec(other),
    }
    .map_err(Error::LoggedInput)?;
    String::from_utf8(bytes).map_err(|error| Error::LoggedInput(serde_json::Error::custom(error)))
}

pub(crate) fn logged<'py>(py: Python<'py>, content: &str) -> PyResult<Bound<'py, PyAny>> {
    let entry = PyDict::new(py);
    entry.set_item("role", "user")?;
    entry.set_item("content", content)?;
    Ok(PyList::new(py, [entry])?.into_any())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn names(order: &[&str]) -> Vec<String> {
        order.iter().map(|name| (*name).to_owned()).collect()
    }

    #[rstest]
    #[case::caller_order(
        &["messages", "max_tokens", "model"],
        r#"{"messages":[],"max_tokens":8,"model":"m"}"#
    )]
    #[case::unlisted_keys_follow_in_sent_order(
        &["model"],
        r#"{"model":"m","max_tokens":8,"messages":[]}"#
    )]
    #[case::unknown_name_skipped(
        &["absent", "model", "max_tokens", "messages"],
        r#"{"model":"m","max_tokens":8,"messages":[]}"#
    )]
    fn top_level_keys_follow_the_given_order(#[case] order: &[&str], #[case] expected: &str) {
        let sent = json!({"model": "m", "max_tokens": 8, "messages": []});

        assert_eq!(
            serialized(&names(order), &sent, &Map::new()).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::int_for_a_float(json!(1.0), json!(1), r#"{"temperature":1}"#)]
    #[case::bool_for_an_int(json!(1), json!(true), r#"{"temperature":true}"#)]
    fn a_caller_scalar_keeps_its_spelling(
        #[case] sent: Value,
        #[case] called: Value,
        #[case] expected: &str,
    ) {
        let as_called = Map::from_iter([("temperature".to_owned(), called)]);

        assert_eq!(
            serialized(
                &names(&["temperature"]),
                &json!({"temperature": sent}),
                &as_called
            )
            .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::non_ascii(json!("caf\u{e9} \u{1f600}"), "\"caf\u{e9} \u{1f600}\"")]
    #[case::small_float(json!(1e-7), "1e-7")]
    #[case::wide_integer(
        json!(litellm_host::raw_json::encode("1180591620717411303424")),
        "1180591620717411303424"
    )]
    fn values_are_written_as_the_wire_writes_them(#[case] leaf: Value, #[case] spelled: &str) {
        let sent = json!({"system": leaf});

        let logged = serialized(&names(&["system"]), &sent, &Map::new()).unwrap();

        assert_eq!(logged, format!(r#"{{"system":{spelled}}}"#));
        assert_eq!(
            logged.as_bytes(),
            litellm_host::raw_json::to_vec(&sent).unwrap()
        );
    }
}
