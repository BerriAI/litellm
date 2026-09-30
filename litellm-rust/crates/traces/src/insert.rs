use std::collections::BTreeMap;

use serde_json::Value;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::Error;

pub fn encode_rows(rows: Vec<BTreeMap<String, Value>>) -> Result<String, Error> {
    rows.into_iter()
        .map(|row| {
            let encoded = row
                .into_iter()
                .map(|(name, value)| insert_value(&name, value).map(|value| (name, value)))
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            serde_json::to_string(&encoded).map_err(|_| Error::InvalidRow)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|rows| rows.join("\n"))
}

fn insert_value(name: &str, value: Value) -> Result<Value, Error> {
    let multiplier = match name {
        "Timestamp" => 1,
        "start_time" | "end_time" | "completion_start_time" => 1_000_000,
        _ => return Ok(value),
    };
    if name == "completion_start_time" && value.is_null() {
        return Ok(value);
    }
    let timestamp = value.as_i64().ok_or(Error::InvalidRow)?;
    let datetime = OffsetDateTime::from_unix_timestamp_nanos(i128::from(timestamp) * multiplier)
        .map_err(|_| Error::InvalidRow)?;
    datetime
        .format(&Rfc3339)
        .map(Value::String)
        .map_err(|_| Error::InvalidRow)
}
