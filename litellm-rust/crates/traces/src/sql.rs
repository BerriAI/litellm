use std::time::Duration;

use litellm_http::Client;

use crate::{Connection, Error};

const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

pub async fn execute_admin_sql(
    client: &Client,
    connection: &Connection,
    sql: &str,
) -> Result<String, Error> {
    if sql.trim().is_empty() {
        return Err(Error::EmptySql);
    }

    let mut url = connection.url().clone();

    let existing_pairs: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(key, _)| {
            !matches!(
                key.as_ref(),
                "query"
                    | "readonly"
                    | "default_format"
                    | "max_result_rows"
                    | "result_overflow_mode"
                    | "max_execution_time"
                    | "wait_end_of_query"
            )
        })
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.query_pairs_mut()
        .clear()
        .extend_pairs(existing_pairs)
        .append_pair("readonly", "1")
        .append_pair("max_result_rows", "1000")
        .append_pair("result_overflow_mode", "throw")
        .append_pair("max_execution_time", "10")
        .append_pair("wait_end_of_query", "1")
        .append_pair("default_format", "JSON");

    let request = client
        .post(url)
        .timeout(Duration::from_secs(15))
        .body(sql.to_owned());
    let mut response = request.send().await.map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        return Err(Error::QueryFailed(response.status().as_u16()));
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| Error::Transport)? {
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(Error::ResponseTooLarge);
        }
        body.extend_from_slice(&chunk);
    }

    let json: serde_json::Value =
        serde_json::from_slice(&body).map_err(|_| Error::InvalidResponse)?;
    if json.get("exception").is_some() || !json.get("data").is_some_and(serde_json::Value::is_array)
    {
        return Err(Error::InvalidResponse);
    }
    String::from_utf8(body).map_err(|_| Error::InvalidResponse)
}
