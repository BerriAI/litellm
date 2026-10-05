use litellm_traces::{
    search::{AgentRuns, FieldFilter, RunField, RunSearch, SearchKey, histogram},
    store::RunCount,
};
use rstest::rstest;

fn filter(field: RunField, pattern: &str, exclude: bool) -> FieldFilter {
    FieldFilter {
        key: SearchKey::Field(field),
        pattern: pattern.into(),
        exclude,
    }
}

#[rstest]
#[case::empty("", &[], vec![])]
#[case::words("foo  bar", &["foo", "bar"], vec![])]
#[case::quoted_phrase(r#""foo bar""#, &["foo bar"], vec![])]
#[case::like_metacharacters_stay_literal("50%_off\\", &["50%_off\\"], vec![])]
#[case::exact(r#"name:"plan trip""#, &[], vec![filter(RunField::Name, "plan trip", false)])]
#[case::glob("agent:res*er", &[], vec![filter(RunField::Agent, "res*er", false)])]
#[case::glob_keeps_the_rest("model:gpt_4*", &[], vec![filter(RunField::Model, "gpt_4*", false)])]
#[case::negated("-has_error:true", &[], vec![filter(RunField::HasError, "true", true)])]
#[case::key_ignores_case("Trace_ID:abc", &[], vec![filter(RunField::TraceId, "abc", false)])]
#[case::value_keeps_colons("input:a:b", &[], vec![filter(RunField::Input, "a:b", false)])]
#[case::negated_text_stays_text("-foo", &["-foo"], vec![])]
#[case::service("service:billing", &[], vec![filter(RunField::Service, "billing", false)])]
#[case::team("-team:acme", &[], vec![filter(RunField::Team, "acme", true)])]
#[case::attribute("attr.gen_ai.system:openai", &[], vec![FieldFilter { key: SearchKey::Attribute("gen_ai.system".into()), pattern: "openai".into(), exclude: false }])]
fn parse_matches_the_dashboard_search_grammar(
    #[case] q: &str,
    #[case] text: &[&str],
    #[case] filters: Vec<FieldFilter>,
) {
    assert_eq!(
        RunSearch::parse(q).unwrap(),
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
        litellm_traces::api::TraceQueryWindow {
            start_ms: 100,
            end_ms: 110,
            as_of_ms: 120,
        },
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

#[rstest]
#[case::unknown("color:red")]
#[case::deprecated_status("status:error")]
#[case::missing_value("root_status:")]
#[case::invalid_status("root_status:success")]
#[case::invalid_boolean("has_error:yes")]
#[case::unclosed_quote("name:\"hello")]
#[case::embedded_quote("a\"b\"c")]
#[case::empty_attribute("attr.:x")]
fn invalid_search_is_rejected(#[case] q: &str) {
    assert!(RunSearch::parse(q).is_err());
}

#[rstest]
fn oversized_search_is_rejected() {
    assert!(RunSearch::parse(&"x".repeat(1001)).is_err());
}

#[rstest]
#[case::full_range(i64::MIN, i64::MAX)]
#[case::minimum_to_zero(i64::MIN, 0)]
#[case::negative_to_maximum(-1, i64::MAX)]
fn histogram_extreme_windows_preserve_edges_and_totals(#[case] start_ms: i64, #[case] end_ms: i64) {
    let window = litellm_traces::api::TraceQueryWindow {
        start_ms,
        end_ms,
        as_of_ms: 1,
    };
    let result = histogram(&[row(0, false, "a", 2), row(1, true, "b", 3)], window, 2);
    assert_eq!(result.window, window);
    assert_eq!(result.buckets[0].start_ms, start_ms);
    assert_eq!(result.buckets[1].end_ms, end_ms);
    assert_eq!(result.buckets[0].end_ms, result.buckets[1].start_ms);
    assert!(
        result
            .buckets
            .iter()
            .all(|bucket| bucket.start_ms < bucket.end_ms)
    );
    assert_eq!(
        result
            .buckets
            .iter()
            .map(|bucket| bucket.total)
            .sum::<u64>(),
        5
    );
    assert_eq!(
        result
            .buckets
            .iter()
            .map(|bucket| bucket.failed)
            .sum::<u64>(),
        3
    );
}
