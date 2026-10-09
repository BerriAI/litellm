use std::collections::BTreeSet;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_traces::Shared;
use serde_json::Value;

use crate::{Error, InsertRow};

pub(crate) const TABLE: &str = "lens_call_costs";

const IDENTITIES: [&str; 9] = [
    "request_id",
    "litellm_call_id",
    "response_id",
    "provider_request_id",
    "trace_id",
    "span_id",
    "team_id",
    "api_key",
    "user",
];

fn text<'a>(row: &'a InsertRow, key: &str) -> Result<&'a str, Error> {
    row.get(key)
        .map_or(Ok(""), |value| value.as_str().ok_or(Error::InvalidRow))
}

fn upstream(response: &str) -> String {
    response
        .strip_prefix("resp_")
        .and_then(|encoded| STANDARD.decode(encoded).ok())
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|decoded| {
            decoded
                .split_once("response_id:")
                .map(|(_, id)| id.split(';').next().unwrap_or_default().to_owned())
        })
        .unwrap_or_default()
}

pub(crate) fn project(rows: &[InsertRow]) -> Result<Vec<InsertRow>, Error> {
    rows.iter()
        .map(project_row)
        .collect::<Result<Vec<_>, _>>()
        .map(|rows| rows.into_iter().flatten().collect())
}

fn project_row(row: &InsertRow) -> Result<Vec<InsertRow>, Error> {
    let request = text(row, "request_id")?;
    let gateway = text(row, "litellm_call_id")?;
    let response = text(row, "response_id")?;
    let upstream = upstream(response);
    let aliases: BTreeSet<_> = [
        ("canonical", request),
        (
            "litellm_request",
            if gateway.is_empty() { request } else { gateway },
        ),
        ("provider_response", response),
        ("provider_response", upstream.as_str()),
        ("provider_request", text(row, "provider_request_id")?),
        ("transport", text(row, "trace_id")?),
    ]
    .into_iter()
    .filter(|(kind, value)| *kind == "canonical" || !value.is_empty())
    .collect();
    let identities = IDENTITIES
        .into_iter()
        .map(|name| {
            text(row, name).map(|value| {
                (
                    name.to_owned(),
                    Shared::new(Value::String(value.to_owned())),
                )
            })
        })
        .collect::<Result<InsertRow, Error>>()?;
    let spend = row
        .get("spend")
        .map_or(Value::Null, |value| (**value).clone());
    if !spend.is_null() && spend.as_f64().is_none_or(|value| !value.is_finite()) {
        return Err(Error::InvalidRow);
    }
    let common: InsertRow = identities
        .into_iter()
        .chain([
            (
                "upstream_response_id".into(),
                Shared::new(Value::String(upstream.clone())),
            ),
            ("spend".into(), Shared::new(spend)),
            (
                "start_time".into(),
                row.get("start_time")
                    .cloned()
                    .unwrap_or_else(|| Shared::new(Value::from(0))),
            ),
            (
                "end_time".into(),
                row.get("end_time")
                    .cloned()
                    .unwrap_or_else(|| Shared::new(Value::from(0))),
            ),
        ])
        .collect();
    Ok(aliases
        .into_iter()
        .map(|(kind, value)| {
            common
                .clone()
                .into_iter()
                .chain([
                    ("key_kind".into(), Shared::new(Value::String(kind.into()))),
                    ("key_value".into(), Shared::new(Value::String(value.into()))),
                ])
                .collect()
        })
        .collect())
}
