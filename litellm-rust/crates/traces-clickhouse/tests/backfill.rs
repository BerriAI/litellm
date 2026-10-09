use std::{process::Output, time::Duration};

use rstest::rstest;
use serde_json::{Value, json};

#[path = "queries/support.rs"]
mod fixtures;
mod support;

use fixtures::{DATABASE, SeededDatabase, migrated_database};
use support::TestResult;

const START_MS: i64 = 1_790_000_000_000;

async fn execute(fixture: &SeededDatabase, sql: &str) -> TestResult<String> {
    Ok(fixture
        .database
        .client
        .post(&fixture.database.url)
        .body(sql.to_owned())
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?)
}

async fn backfill(fixture: &SeededDatabase, cursor: Option<&str>) -> TestResult<Output> {
    Ok(tokio::time::timeout(
        Duration::from_secs(45),
        tokio::process::Command::new(env!("CARGO_BIN_EXE_backfill_call_costs"))
            .env("CLICKHOUSE_URL", &fixture.database.url)
            .env("CLICKHOUSE_DATABASE", DATABASE)
            .args([
                "team-backfill",
                &START_MS.to_string(),
                &(START_MS + 1).to_string(),
            ])
            .args(cursor)
            .kill_on_drop(true)
            .output(),
    )
    .await??)
}

fn output_lines(output: &Output) -> TestResult<Vec<Value>> {
    Ok(std::str::from_utf8(&output.stdout)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?)
}

async fn canonical_costs(fixture: &SeededDatabase) -> TestResult<String> {
    execute(
        fixture,
        "SELECT count(), uniqExact(request_id),
        countIf(isNull(spend) OR spend != (toUInt64(substring(request_id, 9)) + 1) / 8.)
        FROM trace_test.lens_call_costs FINAL
        WHERE key_kind = 'canonical' FORMAT TabSeparated",
    )
    .await
}

#[rstest]
#[tokio::test]
async fn cli_checkpoints_only_persisted_pages_then_resumes_and_replays(
    #[future(awt)] migrated_database: TestResult<SeededDatabase>,
) -> TestResult {
    let fixture = migrated_database?;
    execute(
        &fixture,
        &format!(
            "INSERT INTO trace_test.spend_logs
        (request_id, litellm_call_id, team_id, api_key, user, start_time, end_time, spend)
        SELECT concat('history-', leftPad(toString(number), 4, '0')),
               concat('gateway-', toString(number)), 'team-backfill', 'key', 'user',
               fromUnixTimestamp64Milli({START_MS}), fromUnixTimestamp64Milli({START_MS} + 1),
               (number + 1) / 8. FROM numbers(750)"
        ),
    )
    .await?;
    assert_eq!(canonical_costs(&fixture).await?, "0\t0\t0\n");
    execute(
        &fixture,
        "ALTER TABLE trace_test.lens_call_costs
        ADD CONSTRAINT backfill_boundary CHECK request_id < 'history-0500'",
    )
    .await?;

    let failed = backfill(&fixture, None).await?;
    assert!(
        !failed.status.success(),
        "{}",
        String::from_utf8_lossy(&failed.stderr)
    );
    let checkpoints = output_lines(&failed)?;
    assert_eq!(
        checkpoints.len(),
        1,
        "failed pages must not publish a cursor: {checkpoints:?}"
    );
    assert_eq!(checkpoints[0]["rows"], json!(500));
    let cursor = checkpoints[0]["cursor"]
        .as_str()
        .ok_or("missing checkpoint")?;
    assert_eq!(canonical_costs(&fixture).await?, "500\t500\t0\n");

    execute(
        &fixture,
        "ALTER TABLE trace_test.lens_call_costs DROP CONSTRAINT backfill_boundary",
    )
    .await?;
    let resumed = backfill(&fixture, Some(cursor)).await?;
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let resumed_lines = output_lines(&resumed)?;
    assert_eq!(resumed_lines.len(), 2);
    assert_eq!(resumed_lines[0]["rows"], json!(250));
    assert!(resumed_lines[0]["cursor"].is_string());
    assert_eq!(resumed_lines[1], json!({"complete": true}));
    assert_eq!(canonical_costs(&fixture).await?, "750\t750\t0\n");

    let replayed = backfill(&fixture, None).await?;
    assert!(
        replayed.status.success(),
        "{}",
        String::from_utf8_lossy(&replayed.stderr)
    );
    let replayed_lines = output_lines(&replayed)?;
    assert_eq!(replayed_lines.len(), 3);
    assert_eq!(replayed_lines[2], json!({"complete": true}));
    assert_eq!(canonical_costs(&fixture).await?, "750\t750\t0\n");
    assert_eq!(
        execute(
            &fixture,
            "SELECT count() FROM trace_test.spend_logs FORMAT TabSeparated"
        )
        .await?,
        "750\n"
    );
    Ok(())
}
