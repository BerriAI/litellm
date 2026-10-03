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
        ReadQuery::TraceSpans => named_json::<TraceSpans>(client, connection, parameters).await,
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
