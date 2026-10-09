use std::{collections::BTreeMap, io::Write, process::ExitCode, sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_storage_clickhouse::{Connection, Query, Storage, fetch};
use litellm_traces_clickhouse::{Error, project_spend_rows};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const WINDOW_MS: i64 = 24 * 60 * 60 * 1000;
const PAGE_ROWS: usize = 500;
const MAX_PAGES: usize = 20;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct Page {
    team: String,
    start_ms: i64,
    end_ms: i64,
    after_ms: i64,
    after_id: String,
}

impl Page {
    fn arguments(args: &[String]) -> Result<Self, Error> {
        let [team, start, end, rest @ ..] = args else {
            return Err(Error::InvalidParameters);
        };
        let start_ms: i64 = start.parse().map_err(|_| Error::InvalidParameters)?;
        let end_ms: i64 = end.parse().map_err(|_| Error::InvalidParameters)?;
        if rest.len() > 1 || start_ms < 0 || end_ms <= start_ms || end_ms - start_ms > WINDOW_MS {
            return Err(Error::InvalidParameters);
        }
        let first = Self {
            team: team.clone(),
            start_ms,
            end_ms,
            after_ms: start_ms - 1,
            after_id: String::new(),
        };
        let Some(cursor) = rest.first() else {
            return Ok(first);
        };
        let resumed: Self = serde_json::from_slice(
            &URL_SAFE_NO_PAD
                .decode(cursor)
                .map_err(|_| Error::InvalidParameters)?,
        )
        .map_err(|_| Error::InvalidParameters)?;
        if resumed.team != first.team
            || resumed.start_ms != start_ms
            || resumed.end_ms != end_ms
            || resumed.after_ms < start_ms
            || resumed.after_ms >= end_ms
        {
            return Err(Error::InvalidParameters);
        }
        Ok(resumed)
    }

    fn after(&self, row: &BTreeMap<String, Value>) -> Result<Self, Error> {
        let after_ms = row
            .get("start_time")
            .and_then(Value::as_i64)
            .ok_or(Error::InvalidRow)?;
        let after_id = row
            .get("request_id")
            .and_then(Value::as_str)
            .ok_or(Error::InvalidRow)?;
        if after_ms < self.start_ms
            || after_ms >= self.end_ms
            || (after_ms, after_id) <= (self.after_ms, self.after_id.as_str())
        {
            return Err(Error::InvalidRow);
        }
        Ok(Self {
            after_ms,
            after_id: after_id.into(),
            ..self.clone()
        })
    }

    fn cursor(&self) -> Result<String, Error> {
        serde_json::to_vec(self)
            .map(|value| URL_SAFE_NO_PAD.encode(value))
            .map_err(|_| Error::InvalidParameters)
    }
}

struct SpendPage;
impl Query for SpendPage {
    type Params = Page;
    type Row = BTreeMap<String, Value>;
    const SQL: &'static str = "
        SELECT request_id, litellm_call_id, response_id, provider_request_id, trace_id, span_id,
               team_id, api_key, user, spend,
               toUnixTimestamp64Milli(spend_logs.start_time) AS start_time,
               toUnixTimestamp64Milli(spend_logs.end_time) AS end_time
        FROM spend_logs FINAL
        WHERE team_id = {team:String}
          AND spend_logs.start_time >= fromUnixTimestamp64Milli({start_ms:Int64})
          AND spend_logs.start_time < fromUnixTimestamp64Milli({end_ms:Int64})
          AND (spend_logs.start_time, request_id) > (fromUnixTimestamp64Milli({after_ms:Int64}), {after_id:String})
        ORDER BY spend_logs.start_time, request_id LIMIT 500";
}

fn bounded_reader(connection: &Connection) -> Result<Connection, Error> {
    const SETTINGS: [(&str, &str); 5] = [
        ("max_rows_to_read", "1000000"),
        ("max_bytes_to_read", "67108864"),
        ("read_overflow_mode", "throw"),
        ("max_memory_usage", "134217728"),
        ("output_format_json_quote_64bit_integers", "0"),
    ];
    let mut url = connection.url().clone();
    let pairs: Vec<_> = url
        .query_pairs()
        .filter(|(key, _)| !SETTINGS.iter().any(|(name, _)| key == name))
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    url.query_pairs_mut()
        .clear()
        .extend_pairs(pairs)
        .extend_pairs(SETTINGS);
    Ok(Connection::parse(url.as_str())?)
}

async fn run(initial_page: Page) -> Result<(), Error> {
    let url = std::env::var("CLICKHOUSE_URL").map_err(|_| Error::InvalidParameters)?;
    let database = std::env::var("CLICKHOUSE_DATABASE").unwrap_or_else(|_| "litellm".into());
    let storage = Storage::new(database, &url)?;
    let reader = bounded_reader(storage.reader())?;
    let settings = HttpSettings {
        connect_timeout: Duration::from_secs(5),
        ..HttpSettings::default()
    };
    let client = HttpClientPool::new(Arc::new(PublicDnsResolver))
        .client(
            &Resolution::from(&settings).config,
            ClientVariant::NoRedirect,
        )
        .map_err(|_| Error::Task)?;
    let mut page = initial_page;
    for _ in 0..MAX_PAGES {
        let rows = fetch::<SpendPage>(&client, &reader, &page).await?;
        let Some(last) = rows.last() else {
            println!("{{\"complete\":true}}");
            return Ok(());
        };
        let next = page.after(last)?;
        let count = rows.len();
        project_spend_rows(&client, storage.writer(), storage.database(), rows).await?;
        println!(
            "{}",
            serde_json::json!({"rows": count, "cursor": next.cursor()?})
        );
        std::io::stdout().flush().map_err(|_| Error::Task)?;
        if count < PAGE_ROWS {
            println!("{{\"complete\":true}}");
            return Ok(());
        }
        page = next;
    }
    println!("{{\"complete\":false,\"reason\":\"page_limit\"}}");
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).take(5).collect();
    let Ok(page) = Page::arguments(&args) else {
        eprintln!(
            "Usage: backfill_call_costs TEAM START_MS END_MS [CURSOR]; UTC milliseconds, window <= 24 hours"
        );
        return ExitCode::FAILURE;
    };
    match tokio::time::timeout(Duration::from_secs(300), run(page)).await {
        Ok(Ok(())) => ExitCode::SUCCESS,
        Ok(Err(error)) => {
            eprintln!(
                "Backfill stopped: {error}; resume from the last printed cursor or replay the window"
            );
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!(
                "Backfill stopped after 5 minutes; resume from the last printed cursor or replay the window"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn page() -> Page {
        Page {
            team: "team".into(),
            start_ms: 1000,
            end_ms: 2000,
            after_ms: 1500,
            after_id: "request-a".into(),
        }
    }

    #[rstest]
    #[case::negative("-1", "1000")]
    #[case::empty("1000", "1000")]
    #[case::reversed("2000", "1000")]
    #[case::oversized("0", "86400001")]
    #[case::overflow("0", "9223372036854775808")]
    fn rejects_unbounded_windows(#[case] start: &str, #[case] end: &str) {
        assert!(Page::arguments(&["team".into(), start.into(), end.into()]).is_err());
    }

    #[rstest]
    #[case::unassigned_team("", 0)]
    #[case::assigned_team("team", 1000)]
    fn first_page_includes_the_start_boundary(#[case] team: &str, #[case] start: i64) {
        let page = Page::arguments(&[
            team.into(),
            start.to_string(),
            (start + WINDOW_MS).to_string(),
        ])
        .unwrap();
        let row = BTreeMap::from([
            ("start_time".into(), Value::from(start)),
            ("request_id".into(), Value::from("")),
        ]);
        assert_eq!(page.team, team);
        assert_eq!(page.after(&row).unwrap().after_ms, start);
    }

    #[rstest]
    #[case::same_scope("team", "1000", "2000", true)]
    #[case::other_team("other", "1000", "2000", false)]
    #[case::other_start("team", "900", "2000", false)]
    #[case::other_end("team", "1000", "2001", false)]
    fn cursor_stays_with_its_window(
        page: Page,
        #[case] team: &str,
        #[case] start: &str,
        #[case] end: &str,
        #[case] valid: bool,
    ) {
        let resumed = Page::arguments(&[
            team.into(),
            start.into(),
            end.into(),
            page.cursor().unwrap(),
        ]);
        assert_eq!(resumed.is_ok(), valid);
        if valid {
            assert_eq!(resumed.unwrap(), page);
        }
    }

    #[rstest]
    #[case::same_timestamp_next_request(1500, "request-b", true)]
    #[case::later_timestamp(1501, "request-a", true)]
    #[case::repeated_row(1500, "request-a", false)]
    #[case::earlier_request(1500, "request-0", false)]
    #[case::earlier_timestamp(1499, "request-z", false)]
    #[case::end_is_exclusive(2000, "request-z", false)]
    fn continuation_advances_without_skipping_tied_timestamps(
        page: Page,
        #[case] start: i64,
        #[case] request: &str,
        #[case] valid: bool,
    ) {
        let row = BTreeMap::from([
            ("start_time".into(), Value::from(start)),
            ("request_id".into(), Value::from(request)),
        ]);
        let next = page.after(&row);
        assert_eq!(next.is_ok(), valid);
        if valid {
            let next = next.unwrap();
            assert_eq!((next.after_ms, next.after_id.as_str()), (start, request));
        }
    }
}
