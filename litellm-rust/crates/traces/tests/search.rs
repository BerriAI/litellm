use litellm_traces::{
    search::{AgentRuns, FieldFilter, RunField, RunSearch, histogram},
    store::RunCount,
};
use rstest::rstest;

fn filter(field: RunField, pattern: &str, exclude: bool) -> FieldFilter {
    FieldFilter {
        field,
        pattern: pattern.into(),
        exclude,
    }
}

#[rstest]
#[case::empty("", &[], vec![])]
#[case::words("foo  bar", &["foo", "bar"], vec![])]
#[case::quoted_phrase(r#""foo bar""#, &["foo bar"], vec![])]
#[case::unclosed_quote(r#""foo bar"#, &["foo bar"], vec![])]
#[case::inner_quote_kept(r#"a"b"c"#, &[r#"a"b"c"#], vec![])]
#[case::like_metacharacters_stay_literal("50%_off\\", &["50%_off\\"], vec![])]
#[case::exact(r#"name:"plan trip""#, &[], vec![filter(RunField::Name, "plan trip", false)])]
#[case::glob("agent:res*er", &[], vec![filter(RunField::Agent, "res*er", false)])]
#[case::glob_keeps_the_rest("model:gpt_4*", &[], vec![filter(RunField::Model, "gpt_4*", false)])]
#[case::negated("-status:error", &[], vec![filter(RunField::Status, "error", true)])]
#[case::key_ignores_case("Trace_ID:abc", &[], vec![filter(RunField::TraceId, "abc", false)])]
#[case::value_keeps_colons("input:a:b", &[], vec![filter(RunField::Input, "a:b", false)])]
#[case::missing_value_narrows_nothing("status: foo", &["foo"], vec![])]
#[case::unknown_key_is_text("color:red", &["color:red"], vec![])]
#[case::non_word_key_is_text("k1:v", &["k1:v"], vec![])]
#[case::negated_text_stays_text("-foo", &["-foo"], vec![])]
fn parse_matches_the_dashboard_search_grammar(
    #[case] q: &str,
    #[case] text: &[&str],
    #[case] filters: Vec<FieldFilter>,
) {
    assert_eq!(
        RunSearch::parse(q),
        RunSearch {
            text: text.iter().map(|term| (*term).to_owned()).collect(),
            filters,
        }
    );
}

fn row(bucket: u32, failed: bool, agent: &str, runs: u64) -> RunCount {
    RunCount {
        bucket,
        failed,
        value: agent.into(),
        runs,
    }
}

#[test]
fn histogram_fills_every_bucket_and_splits_failures_from_agents() {
    let shaped = histogram(
        &[
            row(0, true, "a", 3),
            row(0, false, "b", 1),
            row(0, false, "a", 2),
            row(0, true, "c", 1),
            row(2, true, "a", 1),
        ],
        100,
        110,
        3,
    );
    let summary: Vec<_> = shaped
        .buckets
        .iter()
        .map(|bucket| (bucket.start_ms, bucket.end_ms, bucket.total, bucket.failed))
        .collect();
    assert_eq!(
        summary,
        [(100, 103, 7, 4), (103, 106, 0, 0), (106, 110, 1, 1)]
    );
    assert_eq!(
        shaped.buckets[0].agents,
        [
            AgentRuns {
                agent: "a".into(),
                runs: 2
            },
            AgentRuns {
                agent: "b".into(),
                runs: 1
            },
        ]
    );
    assert!(shaped.buckets[2].agents.is_empty());
}
