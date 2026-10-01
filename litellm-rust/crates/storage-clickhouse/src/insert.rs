use std::{io::Write, time::Duration};

use flate2::{Compression, write::GzEncoder};
use litellm_http::Client;

use crate::{Connection, Error, valid_identifier};

const INSERT_TIMEOUT: Duration = Duration::from_secs(30);

pub async fn insert_encoded_rows(
    client: &Client,
    connection: &Connection,
    database: &str,
    table: &str,
    token: &str,
    encoded: &str,
) -> Result<(), Error> {
    if !valid_identifier(database) {
        return Err(Error::InvalidSchema);
    }
    if !valid_identifier(table) {
        return Err(Error::InvalidTable);
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(encoded.as_bytes())
        .map_err(|_| Error::InvalidRow)?;
    let body = encoder.finish().map_err(|_| Error::InvalidRow)?;
    insert_compressed_rows(client, connection, database, table, token, body).await
}

pub async fn insert_compressed_rows(
    client: &Client,
    connection: &Connection,
    database: &str,
    table: &str,
    token: &str,
    body: Vec<u8>,
) -> Result<(), Error> {
    if !valid_identifier(database) {
        return Err(Error::InvalidSchema);
    }
    if !valid_identifier(table) {
        return Err(Error::InvalidTable);
    }
    let mut url = connection.url().clone();
    let existing_pairs: Vec<(String, String)> = url
        .query_pairs()
        .filter(|(key, _)| {
            !matches!(
                key.as_ref(),
                "query"
                    | "async_insert"
                    | "async_insert_deduplicate"
                    | "wait_for_async_insert"
                    | "input_format_skip_unknown_fields"
                    | "date_time_input_format"
            )
        })
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.query_pairs_mut()
        .clear()
        .extend_pairs(existing_pairs)
        .append_pair(
            "query",
            &format!("INSERT INTO `{database}`.{} FORMAT JSONEachRow", table),
        )
        .append_pair("insert_deduplication_token", token)
        .append_pair("async_insert", "1")
        .append_pair("async_insert_deduplicate", "1")
        .append_pair("wait_for_async_insert", "1")
        .append_pair("input_format_skip_unknown_fields", "0")
        .append_pair("date_time_input_format", "best_effort");
    let response = client
        .post(url)
        .timeout(INSERT_TIMEOUT)
        .header("Content-Encoding", "gzip")
        .body(body)
        .send()
        .await
        .map_err(|_| Error::Transport)?;
    if !response.status().is_success() {
        return Err(Error::InsertFailed(response.status().as_u16()));
    }
    Ok(())
}
