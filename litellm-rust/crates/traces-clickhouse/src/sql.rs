use std::collections::BTreeMap;

use litellm_http::Client;
use litellm_storage_clickhouse::{Query, fetch_json};
use litellm_traces::ReadQuery;

use super::{Connection, Error, Parameter, query::lens::*};

pub async fn execute_named_read(
    client: &Client,
    connection: &Connection,
    query: ReadQuery,
    parameters: &BTreeMap<String, Parameter>,
) -> Result<String, Error> {
    match query {
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
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::missing_cursor(serde_json::json!({"offset": 1}))]
    #[case::negative_offset(serde_json::json!({"cursor": "", "offset": -1}))]
    #[case::overflow(serde_json::json!({"cursor": "", "offset": "4294967296"}))]
    #[tokio::test]
    async fn named_read_rejects_invalid_parameters_before_transport(
        #[case] specific: serde_json::Value,
    ) {
        let common = serde_json::json!({
            "all_teams": 1, "team": "", "key_hash": "", "source": "traces", "id": "trace",
            "record_team": "team", "trace_ref": ""
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
            execute_named_read(&client, &connection, ReadQuery::Content, &parameters).await,
            Err(Error::InvalidParameters)
        ));
    }
}
