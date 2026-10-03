use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_traces::ReadQuery;

use super::query::{lens::*, named::*};
use super::{Connection, Error, Parameter};
use litellm_storage_clickhouse::{Query, fetch_json};

pub async fn execute_named_read(
    client: &Client,
    connection: &Connection,
    query: ReadQuery,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    match query {
        ReadQuery::ListTraces => named_json::<ListTraces>(client, connection, parameters).await,
        ReadQuery::TraceIdentity => {
            named_json::<TraceIdentity>(client, connection, parameters).await
        }
        ReadQuery::TraceSpans => tokio::time::timeout(
            std::time::Duration::from_secs(15),
            trace_spans_json(client, connection, parameters),
        )
        .await
        .map_err(|_| litellm_storage_clickhouse::Error::Transport)?,
        ReadQuery::SpanDetail => named_json::<SpanDetail>(client, connection, parameters).await,
        ReadQuery::SpanError => named_json::<SpanError>(client, connection, parameters).await,
        ReadQuery::SpendByResponseIds => {
            named_json::<SpendByResponseIds>(client, connection, parameters).await
        }
        ReadQuery::Availability => {
            named_json::<LensAvailability>(client, connection, parameters).await
        }
        ReadQuery::Agents => named_json::<LensAgents>(client, connection, parameters).await,
        ReadQuery::Sample => named_json::<LensSample>(client, connection, parameters).await,
        ReadQuery::Content => named_json::<LensContent>(client, connection, parameters).await,
        ReadQuery::Evidence => named_json::<LensEvidence>(client, connection, parameters).await,
    }
}

async fn trace_spans_json(
    client: &Client,
    connection: &Connection,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    let value = serde_json::to_value(parameters).map_err(|_| Error::InvalidParameters)?;
    let params: TraceSpansParams =
        serde_json::from_value(value).map_err(|_| Error::InvalidParameters)?;
    let mut parameters: BTreeMap<String, Parameter> =
        serde_json::from_value(serde_json::to_value(params).map_err(|_| Error::InvalidParameters)?)
            .map_err(|_| Error::InvalidParameters)?;
    let limits = litellm_storage_clickhouse::READ_LIMITS;
    let mut spans = Vec::new();
    let mut response_bytes = 0;
    let mut cursor = String::new();
    loop {
        parameters.insert("cursor_span_id".into(), Parameter::Text(cursor));
        parameters.insert(
            "first_page".into(),
            Parameter::Unsigned(u64::from(spans.is_empty())),
        );
        let sql = format!(
            "SELECT * FROM ({}) WHERE {{first_page:UInt8}} = 1 OR span_id > {{cursor_span_id:String}} \
             ORDER BY span_id LIMIT {}",
            TraceSpans::SQL,
            limits.result_rows,
        );
        let body = super::execute_read(client, connection, &sql, &parameters).await?;
        response_bytes += body.len();
        if response_bytes > limits.response_bytes {
            return Err(litellm_storage_clickhouse::Error::ResponseTooLarge.into());
        }
        let mut response: serde_json::Value =
            serde_json::from_str(&body).map_err(|_| Error::InvalidResponse)?;
        let page: Vec<TraceSpansRow> = serde_json::from_value(
            response
                .get_mut("data")
                .ok_or(Error::InvalidResponse)?
                .take(),
        )
        .map_err(|_| Error::InvalidResponse)?;
        let page_rows = page.len();
        cursor = page
            .last()
            .map(|row| row.0.span_id.clone())
            .unwrap_or_default();
        spans.extend(page);
        if page_rows < limits.result_rows as usize {
            spans.sort_by(|left, right| {
                (left.0.start_ns, &left.0.span_id).cmp(&(right.0.start_ns, &right.0.span_id))
            });
            response["rows"] = serde_json::json!(spans.len());
            response["data"] = serde_json::to_value(spans).map_err(|_| Error::InvalidResponse)?;
            if let Some(envelope) = response.as_object_mut() {
                envelope.remove("statistics");
                envelope.remove("rows_before_limit_at_least");
            }
            let result = serde_json::to_string(&response).map_err(|_| Error::InvalidResponse)?;
            if result.len() > limits.response_bytes {
                return Err(litellm_storage_clickhouse::Error::ResponseTooLarge.into());
            }
            return Ok(result);
        }
    }
}

async fn named_json<Q: Query>(
    client: &Client,
    connection: &Connection,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error>
where
    Q::Params: serde::de::DeserializeOwned,
{
    let value = serde_json::to_value(parameters).map_err(|_| Error::InvalidParameters)?;
    let params =
        serde_json::from_value::<Q::Params>(value).map_err(|_| Error::InvalidParameters)?;
    fetch_json::<Q>(client, connection, &params)
        .await
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::missing_span(serde_json::json!({}))]
    #[case::negative_offset(serde_json::json!({"span_id": "span", "error_offset": -1, "error_version": ""}))]
    #[case::overflow(serde_json::json!({"span_id": "span", "error_offset": "18446744073709551616", "error_version": ""}))]
    #[tokio::test]
    async fn named_read_rejects_invalid_parameters_before_transport(
        #[case] specific: serde_json::Value,
    ) {
        let common = serde_json::json!({
            "all_teams": 1, "user_id": "", "team_ids": [], "trace_id": "trace", "trace_ref": ""
        });
        let parameters: BTreeMap<String, Parameter> = common
            .as_object()
            .unwrap()
            .iter()
            .chain(specific.as_object().unwrap().iter())
            .map(|(name, value)| (name.clone(), serde_json::from_value(value.clone()).unwrap()))
            .collect();
        let client = Client::no_redirect_for_test();
        let connection = Connection::parse("http://127.0.0.1:1").unwrap();
        assert!(matches!(
            execute_named_read(&client, &connection, ReadQuery::SpanError, &parameters).await,
            Err(Error::InvalidParameters)
        ));
    }
}
