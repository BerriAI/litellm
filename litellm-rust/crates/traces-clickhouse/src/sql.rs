use std::{collections::BTreeMap, time::Duration};

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
        ReadQuery::TraceSpans => {
            let value = serde_json::to_value(parameters).map_err(|_| Error::InvalidParameters)?;
            let params = serde_json::from_value::<TraceSpansParams>(value)
                .map_err(|_| Error::InvalidParameters)?;
            let rows = trace_spans(client, connection, &params).await?;
            serde_json::to_string(&serde_json::json!({"data": rows}))
                .map_err(|_| Error::InvalidResponse)
        }
        ReadQuery::TracePageSpans => {
            named_json::<TracePageSpans>(client, connection, parameters).await
        }
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

pub(crate) async fn trace_spans(
    client: &Client,
    connection: &Connection,
    params: &TraceSpansParams,
) -> Result<Vec<TraceSpansRow>, Error> {
    use litellm_storage_clickhouse::{READ_LIMITS, execute_read};

    let value = serde_json::to_value(params).map_err(|_| Error::InvalidParameters)?;
    let parameters: BTreeMap<String, Parameter> =
        serde_json::from_value(value).map_err(|_| Error::InvalidParameters)?;
    let sql = format!(
        "SELECT * FROM ({}) WHERE {{first_page:UInt8}} = 1 OR span_id > {{cursor:String}} \
          ORDER BY span_id LIMIT {}",
        TraceSpans::SQL,
        READ_LIMITS.result_rows,
    );
    let read = async {
        // rebind-ok: Owned pagination state avoids copying accumulated rows on each page
        let mut rows = Vec::new();
        let mut response_bytes = "{\"data\":[]}".len();
        let mut cursor = String::new();
        loop {
            let page_parameters = serde_json::from_value::<BTreeMap<String, Parameter>>(
                serde_json::to_value(&parameters).map_err(|_| Error::InvalidParameters)?,
            )
            .map_err(|_| Error::InvalidParameters)?;
            let page_parameters = page_parameters
                .into_iter()
                .chain([
                    (
                        "first_page".into(),
                        Parameter::Unsigned(u64::from(rows.is_empty())),
                    ),
                    ("cursor".into(), Parameter::Text(cursor.clone())),
                ])
                .collect();
            let body = execute_read(client, connection, &sql, &page_parameters).await?;
            #[derive(serde::Deserialize)]
            struct Page {
                data: Vec<TraceSpansRow>,
            }
            let page: Page = serde_json::from_str(&body).map_err(|_| Error::InvalidResponse)?;
            let encoded = serde_json::to_vec(&page.data).map_err(|_| Error::InvalidResponse)?;
            response_bytes +=
                encoded.len() - 2 + usize::from(!rows.is_empty() && !page.data.is_empty());
            if response_bytes > READ_LIMITS.response_bytes {
                return Err(Error::Storage(
                    litellm_storage_clickhouse::Error::ResponseTooLarge,
                ));
            }
            let count = page.data.len();
            if let Some(last) = page.data.last() {
                cursor.clone_from(&last.0.span_id);
            }
            rows.extend(page.data);
            if count < READ_LIMITS.result_rows as usize {
                break;
            }
        }
        rows.sort_by(|a, b| (a.0.start_ns, &a.0.span_id).cmp(&(b.0.start_ns, &b.0.span_id)));
        Ok(rows)
    };
    tokio::time::timeout(Duration::from_secs(15), read)
        .await
        .map_err(|_| Error::Storage(litellm_storage_clickhouse::Error::Transport))?
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
