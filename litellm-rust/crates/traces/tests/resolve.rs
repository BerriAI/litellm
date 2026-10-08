use litellm_traces::{
    AgentNode, RunSourceType, SpanStatus, SpendMatch, iso_time, listed_summary,
    query::named::{ListTracesRow, SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace,
};
use rstest::rstest;

const T0: i64 = 1_790_742_989_000_000_000;
const MS: i64 = 1_000_000;

fn row(span_id: &str, parent: &str, name: &str, kind: &str, agent: &str) -> TraceSpansRow {
    TraceSpansRow {
        trace_id: String::new(),
        original_trace_id: String::new(),
        span_id: span_id.into(),
        parent_span_id: parent.into(),
        name: name.into(),
        kind: kind.parse().unwrap(),
        wrapper_candidate: false,
        agent: agent.into(),
        framework: String::new(),
        status: SpanStatus::Ok,
        status_message: String::new(),
        error_truncated: false,
        start_ns: T0,
        duration_ns: 10 * MS as u64,
        service: "agent-demo".into(),
        input_preview: format!("input of {name}"),
        model: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        litellm_request_id: String::new(),
        call_keys: Vec::new(),
        call_evidence: None,
        tool_call_id: String::new(),
        source_type: String::new(),
        source_url: String::new(),
        source_title: String::new(),
        source_user: String::new(),
        team_id: "team".into(),
        api_key_hash: "key".into(),
        user_id: String::new(),
    }
}

fn at(mut span: TraceSpansRow, start_ms: i64, duration_ms: u64) -> TraceSpansRow {
    span.start_ns = T0 + start_ms * MS;
    span.duration_ns = duration_ms * MS as u64;
    span
}

fn llm(span_id: &str, parent: &str, agent: &str, response_id: &str) -> TraceSpansRow {
    TraceSpansRow {
        model: "claude-sonnet-4-5".into(),
        input_tokens: 100,
        output_tokens: 20,
        litellm_request_id: response_id.into(),
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..at(row(span_id, parent, "ChatOpenAI", "llm", agent), 1, 100)
    }
}

fn spend(request_id: &str, response_id: &str, cost: f64) -> SpendByResponseIdsRow {
    SpendByResponseIdsRow {
        request_id: request_id.into(),
        litellm_call_id: String::new(),
        response_id: response_id.into(),
        upstream_response_id: String::new(),
        provider_request_id: String::new(),
        trace_id: String::new(),
        span_id: String::new(),
        api_key: "key".into(),
        user: String::new(),
        team_id: "team".into(),
        spend: Some(cost),
        start_ms: T0 / MS,
    }
}

/// root agent -> llm, task tool -> researcher subagent (N times) -> llm + search tool + middleware.
fn deep_agent(researchers: usize) -> Vec<TraceSpansRow> {
    let mut rows = vec![
        at(
            row(
                "root",
                "",
                "deep_research_agent",
                "agent",
                "deep_research_agent",
            ),
            0,
            1000,
        ),
        llm("llm-root", "root", "deep_research_agent", "chatcmpl-root"),
        at(
            row("task", "root", "task", "tool", "deep_research_agent"),
            200,
            700,
        ),
    ];
    for index in 0..researchers {
        let researcher = format!("res-{index}");
        rows.extend([
            at(
                row(&researcher, "task", "researcher", "agent", "researcher"),
                201,
                5,
            ),
            at(
                llm(
                    &format!("res-llm-{index}"),
                    &researcher,
                    "researcher",
                    &format!("chatcmpl-res-{index}"),
                ),
                202,
                100,
            ),
            at(
                row(
                    &format!("res-tool-{index}"),
                    &researcher,
                    "search_docs",
                    "tool",
                    "researcher",
                ),
                203,
                1,
            ),
            row(
                &format!("res-mw-{index}"),
                &researcher,
                "FilesystemMiddleware.wrap_model_call",
                "framework",
                "researcher",
            ),
        ]);
    }
    rows
}

fn agents(rows: &[TraceSpansRow]) -> Vec<AgentNode> {
    resolve_trace("t", "", rows, &[])
        .map(|trace| trace.agents)
        .unwrap_or_default()
}

#[rstest]
fn no_rows_is_no_trace() {
    assert_eq!(resolve_trace("t", "", &[], &[]), None);
}

#[rstest]
fn summary_counts_model_calls_tools_and_agents() {
    let mut rows = deep_agent(1);
    rows[2].status = SpanStatus::Error;
    let summary = resolve_trace("t1", "ref", &rows, &[]).unwrap().summary;
    assert_eq!(summary.trace_id, "t1");
    assert_eq!(summary.trace_ref, "ref");
    assert_eq!(summary.name, "deep_research_agent");
    assert_eq!(summary.input_preview, "input of deep_research_agent");
    assert_eq!(summary.status, SpanStatus::Ok);
    assert_eq!(summary.error_count, 1);
    assert_eq!(
        (
            summary.span_count,
            summary.agent_count,
            summary.llm_calls,
            summary.tool_calls
        ),
        (7, 2, 2, 2)
    );
    assert_eq!((summary.input_tokens, summary.output_tokens), (200, 40));
    assert_eq!(summary.models, ["claude-sonnet-4-5"]);
    assert_eq!(summary.duration_ms, 1000.0);
    assert_eq!(summary.start_time, "2026-09-30T04:36:29+00:00");
    assert_eq!(summary.spend, None);
}

fn sourced(mut span: TraceSpansRow, url: &str, title: &str) -> TraceSpansRow {
    span.source_url = url.into();
    span.source_title = title.into();
    span
}

const THREAD: &str = "https://acme.slack.com/archives/C1/p1";

#[rstest]
#[case::root_wins(
    vec![sourced(at(row("root", "", "agent", "agent", "agent"), 5, 10), THREAD, "root thread"),
         sourced(at(row("tool", "root", "tool", "tool", "agent"), 0, 1), "https://other.example/", "child")],
    Some((THREAD, "root thread")),
)]
#[case::earliest_child_when_root_has_none(
    vec![at(row("root", "", "agent", "agent", "agent"), 0, 10),
         sourced(at(row("late", "root", "tool", "tool", "agent"), 5, 1), "https://late.example/", "late"),
         sourced(at(row("early", "root", "tool", "tool", "agent"), 2, 1), THREAD, "early")],
    Some((THREAD, "early")),
)]
#[case::non_https_is_dropped(
    vec![sourced(row("root", "", "agent", "agent", "agent"), "javascript:alert(1)", "x")],
    None,
)]
#[case::absent(vec![row("root", "", "agent", "agent", "agent")], None)]
fn summary_source_links_where_the_run_started(
    #[case] rows: Vec<TraceSpansRow>,
    #[case] expected: Option<(&str, &str)>,
) {
    let source = resolve_trace("t", "", &rows, &[]).unwrap().summary.source;
    assert_eq!(
        source
            .as_ref()
            .map(|source| (source.url.as_str(), source.title.as_str())),
        expected
    );
}

#[rstest]
#[case::slack("slack", RunSourceType::Slack)]
#[case::teams("teams", RunSourceType::Teams)]
#[case::custom("custom", RunSourceType::Custom)]
#[case::unknown_is_custom("my-bot", RunSourceType::Custom)]
#[case::missing_is_custom("", RunSourceType::Custom)]
fn summary_source_type_picks_the_app(#[case] source_type: &str, #[case] expected: RunSourceType) {
    let mut root = sourced(row("root", "", "agent", "agent", "agent"), THREAD, "t");
    root.source_type = source_type.into();
    let source = resolve_trace("t", "", &[root], &[]).unwrap().summary.source;
    assert_eq!(source.map(|source| source.kind), Some(expected));
}

#[rstest]
#[case::set("tin@berri.ai")]
#[case::missing("")]
fn summary_source_carries_who_started_it(#[case] user: &str) {
    let mut root = sourced(row("root", "", "agent", "agent", "agent"), THREAD, "t");
    root.source_user = user.into();
    let source = resolve_trace("t", "", &[root], &[]).unwrap().summary.source;
    assert_eq!(source.map(|source| source.user), Some(user.to_owned()));
}

#[rstest]
fn spans_are_offset_from_the_trace_start() {
    let trace = resolve_trace("t1", "", &deep_agent(1), &[]).unwrap();
    let span = |id: &str| trace.spans.iter().find(|span| span.span_id == id).unwrap();
    assert_eq!(
        (
            span("root").start_offset_ms,
            span("root").parent_span_id.clone()
        ),
        (0.0, None)
    );
    assert_eq!(
        (span("task").start_offset_ms, span("task").duration_ms),
        (200.0, 700.0)
    );
    assert_eq!(span("task").parent_span_id.as_deref(), Some("root"));
    assert_eq!(
        span("llm-root").litellm_request_id.as_deref(),
        Some("chatcmpl-root")
    );
    assert_eq!(span("task").litellm_request_id, None);
}

#[rstest]
fn repeated_subagent_invocations_aggregate_into_one_node() {
    let trace = resolve_trace("t1", "", &deep_agent(200), &[]).unwrap();
    assert_eq!(
        trace.agents[0],
        AgentNode {
            name: "deep_research_agent".into(),
            parent_agent: None,
            invocations: 1,
            llm_calls: 1,
            tool_calls: 1,
            duration_ms: 1000.0,
            spend: None,
            priced_calls: 0,
        }
    );
    let researcher = &trace.agents[1];
    assert_eq!(
        researcher.parent_agent.as_deref(),
        Some("deep_research_agent")
    );
    assert_eq!(
        (
            researcher.invocations,
            researcher.llm_calls,
            researcher.tool_calls
        ),
        (200, 200, 200)
    );
    assert!((researcher.duration_ms - 1000.0).abs() < 1e-6);
    assert_eq!(trace.summary.span_count, 3 + 4 * 200);
}

#[rstest]
fn parent_agent_skips_same_name_ancestors_and_stops_at_cycles() {
    let recursive = agents(&[
        row("root", "", "lead", "agent", "lead"),
        row("r1", "root", "researcher", "agent", "researcher"),
        row("r2", "r1", "researcher", "agent", "researcher"),
    ]);
    assert_eq!(recursive[1].parent_agent.as_deref(), Some("lead"));
    assert_eq!(recursive[1].invocations, 2);
    let cyclic = agents(&[
        row("self", "self", "researcher", "agent", "researcher"),
        row("first", "second", "researcher", "agent", "researcher"),
        row("second", "first", "researcher", "agent", "researcher"),
    ]);
    assert_eq!(cyclic[0].parent_agent, None);
}

#[rstest]
fn unnamed_calls_belong_to_the_nearest_agent_and_wrappers_are_not_agents() {
    let crew = TraceSpansRow {
        wrapper_candidate: true,
        ..row("crew", "", "crew.kickoff", "agent", "")
    };
    let nodes = agents(&[
        crew,
        row(
            "a",
            "crew",
            "researcher._execute_core",
            "agent",
            "researcher",
        ),
        row("chain", "a", "step", "chain", ""),
        llm("llm", "chain", "", "req-1"),
        row("tool", "a", "search", "tool", ""),
        llm("orphan", "missing", "", "req-2"),
    ]);
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].name, "researcher");
    assert_eq!((nodes[0].llm_calls, nodes[0].tool_calls), (1, 1));
    assert_eq!(nodes[0].parent_agent, None);
}

#[rstest]
fn named_wrapper_inside_the_same_agent_is_a_chain() {
    let wrapper = TraceSpansRow {
        wrapper_candidate: true,
        ..row("w", "a", "researcher.run", "agent", "researcher")
    };
    let trace = resolve_trace(
        "t",
        "",
        &[row("a", "", "researcher", "agent", "researcher"), wrapper],
        &[],
    )
    .unwrap();
    assert_eq!(trace.spans[1].kind, litellm_traces::ObservationType::Chain);
    assert_eq!(trace.agents[0].invocations, 1);
}

#[rstest]
fn agents_named_only_by_their_tools_are_agents() {
    let nodes = agents(&[row("t", "", "tool", "tool", "ghost")]);
    assert_eq!(
        (
            nodes[0].name.as_str(),
            nodes[0].invocations,
            nodes[0].tool_calls
        ),
        ("ghost", 1, 1)
    );
}

#[rstest]
fn overlapping_tool_spans_count_one_call() {
    let tool = |span_id: &str| TraceSpansRow {
        tool_call_id: "call-1".into(),
        ..row(span_id, "a", "search", "tool", "")
    };
    let trace = resolve_trace(
        "t",
        "",
        &[
            row("a", "", "agent", "agent", "agent"),
            tool("x"),
            tool("y"),
        ],
        &[],
    )
    .unwrap();
    assert_eq!(trace.summary.tool_calls, 1);
    assert_eq!(trace.agents[0].tool_calls, 1);
}

#[rstest]
fn names_and_frameworks_are_sorted_and_distinct() {
    let framed = |span: TraceSpansRow, framework: &str| TraceSpansRow {
        framework: framework.into(),
        ..span
    };
    let trace = resolve_trace(
        "t1",
        "",
        &[
            framed(
                row(
                    "root",
                    "",
                    "invoke_agent research_agent",
                    "agent",
                    "research_agent",
                ),
                "claude-code",
            ),
            framed(
                row(
                    "r1",
                    "root",
                    "researcher._execute_core",
                    "agent",
                    "researcher",
                ),
                "claude-agent-sdk",
            ),
            framed(
                row("r2", "r1", "invoke_agent researcher", "agent", "researcher"),
                "",
            ),
            row("llm", "r2", "chat", "llm", "researcher"),
        ],
        &[],
    )
    .unwrap();
    assert_eq!(trace.summary.agent_names, ["research_agent", "researcher"]);
    assert_eq!(
        trace.summary.frameworks,
        ["claude-agent-sdk", "claude-code"]
    );
    assert_eq!(trace.summary.name, "invoke_agent research_agent");
    assert_eq!(trace.agents[1].invocations, 2);
    assert_eq!(trace.agents[1].llm_calls, 1);
}

#[rstest]
fn repeated_response_id_counts_once() {
    let rows = [
        row("root", "", "agent", "agent", "agent"),
        llm("llm-1", "root", "agent", "response-1"),
        llm("llm-2", "root", "agent", "response-1"),
    ];
    let spend = [
        spend("request-1", "response-1", 0.25),
        spend("request-other", "unrelated-response", 50.0),
    ];
    let trace = resolve_trace("trace-1", "ref", &rows, &spend).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.agents[0].spend, Some(0.25));
    assert_eq!(
        trace
            .spans
            .iter()
            .map(|span| span.spend)
            .collect::<Vec<_>>(),
        [None, Some(0.25), Some(0.25)]
    );
}

#[rstest]
fn model_calls_link_the_spend_log_they_were_priced_from() {
    let rows = [
        row("agent", "", "agent", "agent", "agent"),
        llm("matched", "agent", "agent", "matched"),
        llm("no-log", "agent", "agent", "missing"),
        llm("no-id", "agent", "agent", ""),
        llm("ambiguous", "agent", "agent", "cached"),
    ];
    let logs = [
        spend("request-matched", "matched", 0.25),
        spend("request-cached-a", "cached", 0.25),
        spend("request-cached-b", "cached", 0.0),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    let links: Vec<_> = trace
        .spans
        .iter()
        .map(|span| (span.spend_log_request_id.as_deref(), span.spend_match))
        .collect();
    assert_eq!(
        links,
        [
            (None, None),
            (Some("request-matched"), Some(SpendMatch::Matched)),
            (None, Some(SpendMatch::NoSpendLog)),
            (None, Some(SpendMatch::NoCallId)),
            (None, Some(SpendMatch::Ambiguous)),
        ]
    );
}

#[rstest]
fn model_call_span_cost_agrees_with_the_run_total_when_priced_from_a_wrapper() {
    let rows = [
        TraceSpansRow {
            call_keys: vec![litellm_traces::CallKey::LiteLlmRequest("gateway".into())],
            call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
            ..llm("wrapper", "", "agent", "")
        },
        llm("call", "wrapper", "agent", ""),
    ];
    let logs = [SpendByResponseIdsRow {
        litellm_call_id: "gateway".into(),
        ..spend("request", "chatcmpl-request", 0.25)
    }];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(
        (
            trace.spans[1].spend,
            trace.spans[1].spend_log_request_id.as_deref()
        ),
        (Some(0.25), Some("request"))
    );
}

#[rstest]
fn unpriced_call_leaves_a_partial_total_of_the_priced_calls() {
    let rows = [
        row("agent", "", "agent", "agent", "agent"),
        llm("first", "agent", "agent", "first"),
        llm("second", "agent", "agent", "second"),
        llm("third", "agent", "agent", ""),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &[spend("first", "first", 0.25)]).unwrap();
    assert_eq!(
        (
            trace.summary.spend,
            trace.summary.priced_calls,
            trace.summary.llm_calls
        ),
        (Some(0.25), 1, 3)
    );
    assert_eq!(
        (trace.agents[0].spend, trace.agents[0].priced_calls),
        (Some(0.25), 1)
    );
    assert_eq!(
        trace
            .spans
            .iter()
            .map(|span| span.spend)
            .collect::<Vec<_>>(),
        [None, Some(0.25), None, None]
    );
}

#[rstest]
fn no_priced_call_leaves_cost_unknown() {
    let rows = [llm("call", "", "agent", "response")];
    let trace = resolve_trace("trace", "ref", &rows, &[spend("other", "other", 0.25)]).unwrap();
    assert_eq!((trace.summary.spend, trace.summary.priced_calls), (None, 0));
}

fn gateway_logged(request_id: &str, call_id: &str, cost: f64) -> SpendByResponseIdsRow {
    SpendByResponseIdsRow {
        litellm_call_id: call_id.into(),
        ..spend(request_id, &format!("chatcmpl-{request_id}"), cost)
    }
}

#[rstest]
#[case::response_id(litellm_traces::CallKey::ProviderResponse("chatcmpl-request".into()), Some(0.25))]
#[case::gateway_call_id(litellm_traces::CallKey::LiteLlmRequest("gateway".into()), Some(0.25))]
#[case::other_call_id(litellm_traces::CallKey::LiteLlmRequest("other".into()), None)]
#[case::request_id_is_not_a_call_id(litellm_traces::CallKey::LiteLlmRequest("request".into()), None)]
#[case::transport(litellm_traces::CallKey::Transport, None)]
#[case::gateway_attempt(litellm_traces::CallKey::GatewayAttempt, None)]
fn only_ids_litellm_assigned_join_spend(
    #[case] key: litellm_traces::CallKey,
    #[case] expected: Option<f64>,
) {
    let rows = [TraceSpansRow {
        trace_id: "trace".into(),
        call_keys: vec![key],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [SpendByResponseIdsRow {
        upstream_response_id: "trace".into(),
        ..gateway_logged("request", "gateway", 0.25)
    }];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (trace.summary.spend, trace.summary.priced_calls),
        (expected, u64::from(expected.is_some()))
    );
}

#[rstest]
#[case::same_team("team", Some(0.25), SpendMatch::Matched)]
#[case::other_team("other-team", None, SpendMatch::NoSpendLog)]
fn spend_logs_price_only_calls_in_their_own_team(
    #[case] logged_team: &str,
    #[case] expected: Option<f64>,
    #[case] matched: SpendMatch,
) {
    let rows = [TraceSpansRow {
        team_id: "team".into(),
        ..llm("call", "", "agent", "response")
    }];
    let logs = [SpendByResponseIdsRow {
        team_id: logged_team.into(),
        ..spend("request", "response", 0.25)
    }];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (trace.summary.spend, trace.spans[0].spend_match),
        (expected, Some(matched))
    );
}

#[rstest]
#[case::legacy_row_without_call_id("", Some(0.25), SpendMatch::Matched)]
#[case::call_id_names_nothing("missing", None, SpendMatch::Ambiguous)]
fn recorded_gateway_id_must_agree_unless_spend_predates_gateway_ids(
    #[case] logged_call_id: &str,
    #[case] expected: Option<f64>,
    #[case] matched: SpendMatch,
) {
    let rows = [TraceSpansRow {
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("chatcmpl-request".into()),
            litellm_traces::CallKey::LiteLlmRequest("gateway".into()),
        ],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [gateway_logged("request", logged_call_id, 0.25)];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (trace.spans[0].spend, trace.spans[0].spend_match),
        (expected, Some(matched))
    );
}

#[rstest]
#[case::missing_cost(None, SpendMatch::Matched)]
#[case::finite_cost(Some(0.25), SpendMatch::Matched)]
fn a_single_matched_row_is_matched_whatever_its_cost(
    #[case] cost: Option<f64>,
    #[case] matched: SpendMatch,
) {
    let rows = [llm("call", "", "agent", "response")];
    let logs = [SpendByResponseIdsRow {
        spend: cost,
        ..spend("request", "response", 0.0)
    }];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (
            trace.spans[0].spend,
            trace.spans[0].spend_match,
            trace.spans[0].spend_log_request_id.as_deref()
        ),
        (cost, Some(matched), Some("request"))
    );
    assert_eq!(trace.summary.priced_calls, u64::from(cost.is_some()));
}

#[rstest]
#[case::same_request("gateway", Some(0.25), SpendMatch::Matched)]
#[case::conflicting_requests("other", None, SpendMatch::Ambiguous)]
fn response_id_and_call_id_must_name_the_same_request(
    #[case] call_id: &str,
    #[case] expected: Option<f64>,
    #[case] matched: SpendMatch,
) {
    let rows = [TraceSpansRow {
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("chatcmpl-request".into()),
            litellm_traces::CallKey::LiteLlmRequest(call_id.into()),
        ],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [
        gateway_logged("request", "gateway", 0.25),
        gateway_logged("other-request", "other", 0.5),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (
            trace.spans[0].spend,
            trace.summary.spend,
            trace.spans[0].spend_match
        ),
        (expected, expected, Some(matched))
    );
}

#[rstest]
fn call_id_picks_the_call_when_its_response_id_has_a_cache_hit_twin() {
    let rows = [TraceSpansRow {
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("chatcmpl-shared".into()),
            litellm_traces::CallKey::LiteLlmRequest("gateway".into()),
        ],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [
        SpendByResponseIdsRow {
            litellm_call_id: "gateway".into(),
            ..spend("request", "chatcmpl-shared", 0.25)
        },
        SpendByResponseIdsRow {
            litellm_call_id: "cache-hit".into(),
            ..spend("request_cache_hit", "chatcmpl-shared", 0.0)
        },
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (
            trace.summary.spend,
            trace.spans[0].spend_log_request_id.as_deref()
        ),
        (Some(0.25), Some("request"))
    );
}

#[rstest]
fn wrapper_recording_a_retry_and_its_final_attempt_prices_both() {
    let rows = [TraceSpansRow {
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("chatcmpl-retry".into()),
            litellm_traces::CallKey::ProviderResponse("chatcmpl-final".into()),
        ],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [
        spend("retry", "chatcmpl-retry", 0.25),
        spend("final", "chatcmpl-final", 0.5),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(
        (trace.summary.spend, trace.spans[0].spend_match),
        (Some(0.75), Some(SpendMatch::Matched))
    );
}

#[rstest]
fn call_id_prices_a_legacy_spend_log_without_litellm_call_id() {
    let rows = [TraceSpansRow {
        call_keys: vec![litellm_traces::CallKey::LiteLlmRequest("legacy".into())],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..llm("call", "", "agent", "")
    }];
    let logs = [spend("legacy", "chatcmpl-legacy", 0.25)];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
}

#[rstest]
fn spend_lookup_collects_only_assigned_ids() {
    let recorded = TraceSpansRow {
        trace_id: "trace".to_owned(),
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("response".to_owned()),
            litellm_traces::CallKey::LiteLlmRequest("request".to_owned()),
            litellm_traces::CallKey::Transport,
        ],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Unknown),
        ..row("span", "", "operation", "llm", "")
    };
    assert_eq!(
        litellm_traces::SpendLookup::new(&[recorded]),
        litellm_traces::SpendLookup {
            response_ids: vec!["response".into()],
            provider_request_ids: Vec::new(),
            request_ids: vec!["request".into()],
            trace_ids: vec!["trace".into()],
        }
    );
}

#[rstest]
fn upstream_response_id_inside_a_managed_id_joins_spend() {
    let rows = [llm("call", "", "agent", "chatcmpl-upstream")];
    let logs = [SpendByResponseIdsRow {
        upstream_response_id: "chatcmpl-upstream".into(),
        ..spend("request", "resp_managed", 0.25)
    }];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
}

#[rstest]
fn conflicting_complete_wrapper_cannot_price_an_unrelated_call() {
    let rows = [
        TraceSpansRow {
            call_keys: vec![litellm_traces::CallKey::ProviderResponse(
                "retry-response".into(),
            )],
            call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
            ..llm("wrapper", "", "agent", "")
        },
        llm("call", "wrapper", "agent", "response"),
    ];
    let logs = [
        spend("retry", "retry-response", 0.25),
        spend("final", "response", 0.5),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!((trace.summary.spend, trace.summary.priced_calls), (None, 0));
}

#[rstest]
fn ambiguous_response_id_keeps_cost_unknown() {
    let rows = [llm("llm-1", "", "agent", "response-1")];
    let spend = [
        spend("response-1", "response-1", 0.25),
        spend("response-1_cache_hit123", "response-1", 0.0),
    ];
    let trace = resolve_trace("trace-1", "ref", &rows, &spend).unwrap();
    assert_eq!((trace.summary.spend, trace.spans[0].spend), (None, None));
}

#[rstest]
fn listed_summary_keeps_rollup_counts_with_unknown_cost() {
    let summary = listed_summary(&ListTracesRow {
        trace_id: "t1".into(),
        trace_ref: "ref".into(),
        team_id: "team".into(),
        api_key_hash: "key".into(),
        user_id: "owner".into(),
        name: "deep_research_agent".into(),
        service: "agent-demo".into(),
        input_preview: "hi".into(),
        status: SpanStatus::Ok,
        start_ms: 1_790_742_989_377,
        duration_ms: 51_385,
        span_count: 126,
        agent_count: 2,
        agent_invocations: 0,
        agent_names: vec!["deep_research_agent".into()],
        frameworks: vec!["claude-agent-sdk".into()],
        llm_calls: 7,
        tool_calls: 26,
        input_tokens: 30_175,
        output_tokens: 2_620,
        models: vec!["claude-sonnet-4-5".into()],
        error_count: 1,
        request_ids: Vec::new(),
    });
    assert_eq!(summary.spend, None);
    assert_eq!(summary.status, SpanStatus::Ok);
    assert_eq!(
        (
            summary.span_count,
            summary.error_count,
            summary.agent_invocations
        ),
        (126, 1, 2)
    );
    assert_eq!(summary.start_time, "2026-09-30T04:36:29.377000+00:00");
}

#[rstest]
#[case::whole_second(1_790_742_989_000, "2026-09-30T04:36:29+00:00")]
#[case::milliseconds(1_790_742_989_007, "2026-09-30T04:36:29.007000+00:00")]
#[case::before_epoch(-500, "1969-12-31T23:59:59.500000+00:00")]
fn iso_time_matches_python_isoformat(#[case] ms: i64, #[case] expected: &str) {
    assert_eq!(iso_time(ms), expected);
}

#[rstest]
#[case::missing(None, None)]
#[case::free(Some(0.0), Some(0.0))]
#[case::paid(Some(0.25), Some(0.25))]
#[case::nan(Some(f64::NAN), None)]
#[case::infinity(Some(f64::INFINITY), None)]
fn complete_correlation_requires_known_finite_cost(
    #[case] cost: Option<f64>,
    #[case] expected: Option<f64>,
) {
    let rows = [llm("call", "", "agent", "response")];
    let logged = SpendByResponseIdsRow {
        spend: cost,
        ..spend("request", "response", 0.25)
    };
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
    assert_eq!(trace.spans[0].spend, expected);
}

#[rstest]
#[case::finite(0.25, Some(0.5))]
#[case::overflow(f64::MAX, None)]
fn trace_cost_requires_a_finite_total(#[case] cost: f64, #[case] expected: Option<f64>) {
    let rows = [
        llm("first", "", "agent", "response-a"),
        llm("second", "", "agent", "response-b"),
    ];
    let logs = [
        spend("request-a", "response-a", cost),
        spend("request-b", "response-b", cost),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
}

#[rstest]
#[case::legacy(None, Some(0.25))]
#[case::unknown(Some(litellm_traces::CallEvidenceKind::Unknown), None)]
#[case::partial(Some(litellm_traces::CallEvidenceKind::Partial), None)]
#[case::complete(Some(litellm_traces::CallEvidenceKind::Complete), Some(0.25))]
fn stored_response_id_requires_complete_call_evidence(
    #[case] evidence: Option<litellm_traces::CallEvidenceKind>,
    #[case] expected: Option<f64>,
) {
    let span = TraceSpansRow {
        call_evidence: evidence,
        ..llm("call", "", "agent", "response")
    };
    let stored = serde_json::to_value(span).unwrap();
    let decoded: TraceSpansRow = serde_json::from_value(stored).unwrap();
    let logs = [spend("request", "response", 0.25)];
    let trace = resolve_trace("trace", "ref", &[decoded], &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.spans[0].spend, expected);
}

#[rstest]
#[case::wrapper("wrapper_candidate", serde_json::json!(2))]
#[case::truncation("error_truncated", serde_json::json!(2))]
#[case::call_key("call_keys", serde_json::json!(["provider_response:"]))]
#[case::call_evidence("call_evidence", serde_json::json!("invalid"))]
#[case::role("type", serde_json::json!("invalid"))]
fn malformed_stored_span_fields_are_rejected(
    #[case] field: &str,
    #[case] value: serde_json::Value,
) {
    let mut encoded = serde_json::to_value(row("span", "", "agent", "agent", "agent")).unwrap();
    encoded[field] = value;
    assert!(serde_json::from_value::<TraceSpansRow>(encoded).is_err());
}

#[rstest]
#[case::parent_first(false)]
#[case::child_first(true)]
fn overlapping_model_spans_count_leaf_usage_and_keep_agent_ownership(#[case] reverse: bool) {
    let root = TraceSpansRow {
        input_tokens: 900,
        output_tokens: 800,
        ..row("root", "", "planner", "agent", "planner")
    };
    let wrapper = TraceSpansRow {
        input_tokens: 700,
        output_tokens: 600,
        ..llm("wrapper", "root", "", "")
    };
    let call = llm("call", "wrapper", "", "");
    let rows = if reverse {
        [call, wrapper, root]
    } else {
        [root, wrapper, call]
    };
    let trace = resolve_trace("trace", "ref", &rows, &[]).unwrap();
    assert_eq!(trace.summary.name, "planner");
    assert_eq!(trace.summary.llm_calls, 1);
    assert_eq!(
        (trace.summary.input_tokens, trace.summary.output_tokens),
        (100, 20)
    );
    assert_eq!(trace.agents.len(), 1);
    assert_eq!(trace.agents[0].name, "planner");
    assert_eq!(trace.agents[0].llm_calls, 1);
}

#[rstest]
fn empty_root_preview_uses_the_earliest_agent_or_model_input() {
    let rows = [
        at(
            TraceSpansRow {
                input_preview: "later input".into(),
                ..llm("later", "root", "", "")
            },
            20,
            1,
        ),
        TraceSpansRow {
            input_preview: String::new(),
            ..row("root", "", "planner", "agent", "planner")
        },
        at(row("tool", "root", "search", "tool", ""), 1, 1),
        at(
            TraceSpansRow {
                input_preview: "earlier input".into(),
                ..llm("earlier", "root", "", "")
            },
            10,
            1,
        ),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &[]).unwrap();
    assert_eq!(trace.summary.input_preview, rows[3].input_preview);
    assert_eq!(trace.summary.name, rows[1].name);
    assert_eq!(trace.spans[0].start_offset_ms, 20.0);
    assert_eq!(trace.spans[3].start_offset_ms, 10.0);
}

#[rstest]
#[case::oldest_first(false)]
#[case::newest_first(true)]
fn repeated_request_ids_preserve_storage_identity(#[case] reverse: bool) {
    let first = SpendByResponseIdsRow {
        start_ms: 100,
        ..spend("same", "response", 0.25)
    };
    let second = SpendByResponseIdsRow {
        start_ms: 200,
        ..spend("same", "response", 0.5)
    };
    let logs = if reverse {
        [second, first]
    } else {
        [first, second]
    };
    let rows = [llm("call", "", "agent", "response")];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, None);
    assert_eq!(trace.spans[0].spend, None);
}

#[rstest]
#[case::same_row(true, Some(0.25))]
#[case::distinct_rows(false, Some(0.75))]
fn totals_deduplicate_only_equal_storage_identities(
    #[case] duplicate: bool,
    #[case] expected: Option<f64>,
) {
    let rows = [
        llm("first", "", "agent", "a"),
        llm("second", "", "agent", if duplicate { "a" } else { "b" }),
    ];
    let logs = [
        SpendByResponseIdsRow {
            start_ms: 100,
            ..spend("same", "a", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: if duplicate { 100 } else { 200 },
            ..spend(
                "same",
                if duplicate { "a" } else { "b" },
                if duplicate { 0.25 } else { 0.5 },
            )
        },
    ];
    assert_eq!(
        resolve_trace("trace", "ref", &rows, &logs)
            .unwrap()
            .summary
            .spend,
        expected
    );
}

#[rstest]
#[case::matching("call-one", "claude_code.tool.execution", SpanStatus::Error)]
#[case::other_tool("other-call", "claude_code.tool.execution", SpanStatus::Ok)]
#[case::child_agent("call-one", "child agent", SpanStatus::Ok)]
fn native_tool_status_uses_only_its_own_execution_error(
    #[case] call: &str,
    #[case] name: &str,
    #[case] expected: SpanStatus,
) {
    let tool = TraceSpansRow {
        framework: "claude-code".to_owned(),
        tool_call_id: "call-one".to_owned(),
        ..row("tool", "", "Bash", "tool", "claude-code")
    };
    let execution = TraceSpansRow {
        status: SpanStatus::Error,
        status_message: "exit 3".to_owned(),
        tool_call_id: call.to_owned(),
        ..row("execution", "tool", name, "framework", "claude-code")
    };
    let trace = resolve_trace("trace", "", &[tool, execution], &[]).unwrap();
    assert_eq!(trace.spans[0].status, expected);
    assert_eq!(
        trace.spans[0].error.as_deref(),
        if expected == SpanStatus::Error {
            Some("exit 3")
        } else {
            None
        }
    );
}

#[rstest]
#[case::matching("call-one", SpanStatus::Error)]
#[case::other_tool("other-call", SpanStatus::Ok)]
fn native_tool_failure_log_matches_by_call_id_without_double_counting(
    #[case] call: &str,
    #[case] expected: SpanStatus,
) {
    let tool = TraceSpansRow {
        framework: "claude-code".into(),
        tool_call_id: "call-one".into(),
        ..row("tool", "root", "Bash", "tool", "claude-code")
    };
    let log = TraceSpansRow {
        framework: "claude-code".into(),
        status: SpanStatus::Error,
        status_message: "Permission denied".into(),
        tool_call_id: call.into(),
        ..row(
            "log",
            "root",
            "claude_code.tool_result",
            "framework",
            "claude-code",
        )
    };
    let trace = resolve_trace("trace", "", &[tool, log], &[]).unwrap();
    assert_eq!(trace.spans[0].status, expected);
    assert_eq!(trace.summary.error_count, 1);
}

fn owned_spend(
    request_id: &str,
    response_id: &str,
    team: &str,
    user: &str,
    key: &str,
    cost: f64,
) -> SpendByResponseIdsRow {
    SpendByResponseIdsRow {
        request_id: request_id.into(),
        response_id: response_id.into(),
        litellm_call_id: String::new(),
        upstream_response_id: String::new(),
        provider_request_id: String::new(),
        trace_id: String::new(),
        span_id: String::new(),
        team_id: team.into(),
        api_key: key.into(),
        user: user.into(),
        spend: Some(cost),
        start_ms: T0 / MS,
    }
}

fn owned(mut span: TraceSpansRow, team: &str, user: &str, key: &str) -> TraceSpansRow {
    span.team_id = team.into();
    span.user_id = user.into();
    span.api_key_hash = key.into();
    span
}

#[rstest]
fn repeated_response_counts_once_and_other_owners_are_ignored() {
    let rows = [
        owned(
            row("root", "", "agent", "agent", "agent"),
            "team-a",
            "",
            "key-a",
        ),
        owned(
            llm("llm-1", "root", "agent", "response-1"),
            "team-a",
            "",
            "key-a",
        ),
        owned(
            llm("llm-2", "root", "agent", "response-1"),
            "team-a",
            "",
            "key-a",
        ),
    ];
    let spend = [
        owned_spend("request-other", "response-1", "team-b", "", "key-b", 99.0),
        owned_spend("request-1", "response-1", "team-a", "", "key-a", 0.25),
        owned_spend(
            "request-other-key",
            "unrelated-response",
            "team-a",
            "",
            "key-c",
            50.0,
        ),
    ];
    let trace = resolve_trace("trace-1", "ref", &rows, &spend).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.agents[0].spend, Some(0.25));
    assert_eq!(
        trace
            .spans
            .iter()
            .map(|span| span.spend)
            .collect::<Vec<_>>(),
        [None, Some(0.25), Some(0.25)]
    );
}

#[rstest]
#[case::key_differs("team", "", "export", "team", "", "request", false)]
#[case::shared_key("team", "", "export", "team", "", "export", true)]
#[case::shared_user("", "user", "export", "", "user", "request", true)]
#[case::teamless_key("", "", "key", "", "", "key", true)]
#[case::other_team("team", "user", "key", "other-team", "user", "key", false)]
#[case::other_user("", "user", "export", "", "other-user", "request", false)]
#[case::no_shared_identity("", "", "export", "", "", "request", false)]
#[case::no_identity("", "", "", "", "", "", false)]
#[case::master_key_without_spend_key("", "", "master", "", "", "", false)]
fn cost_requires_shared_ownership(
    #[case] trace_team: &str,
    #[case] trace_user: &str,
    #[case] trace_key: &str,
    #[case] spend_team: &str,
    #[case] spend_user: &str,
    #[case] spend_key: &str,
    #[case] known: bool,
) {
    let rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            trace_team,
            trace_user,
            trace_key,
        ),
        owned(
            llm("llm", "agent", "agent", "response"),
            trace_team,
            trace_user,
            trace_key,
        ),
    ];
    let spend = [owned_spend(
        "request", "response", spend_team, spend_user, spend_key, 0.25,
    )];
    let trace = resolve_trace("trace", "visible-reference", &rows, &spend).unwrap();
    let expected = known.then_some(0.25);
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
    assert_eq!(trace.spans[1].spend, expected);
}

#[rstest]
#[case::missing_id("missing_id")]
#[case::missing_spend("missing_spend")]
#[case::duplicate_spend("duplicate_spend")]
fn partial_trace_total_counts_only_complete_calls(#[case] failure: &str) {
    let second_id = if failure == "missing_id" {
        ""
    } else {
        "second"
    };
    let rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "export",
        ),
        owned(
            llm("first", "agent", "agent", "first"),
            "team",
            "",
            "export",
        ),
        owned(
            llm("second", "agent", "agent", second_id),
            "team",
            "",
            "export",
        ),
    ];
    let first = owned_spend("first", "first", "team", "", "export", 0.25);
    let second = owned_spend("second", "second", "team", "", "export", 0.25);
    let duplicate = owned_spend("duplicate", "second", "team", "", "export", 0.25);
    let spend = if failure == "duplicate_spend" {
        vec![first, second, duplicate]
    } else {
        vec![first]
    };
    let trace = resolve_trace("trace", "ref", &rows, &spend).unwrap();
    assert_eq!(trace.spans[1].spend, Some(0.25));
    assert_eq!(trace.spans[2].spend, None);
    assert_eq!(
        (trace.summary.spend, trace.summary.priced_calls),
        (Some(0.25), 1)
    );
    assert_eq!(
        (trace.agents[0].spend, trace.agents[0].priced_calls),
        (Some(0.25), 1)
    );
}

#[rstest]
fn transport_spans_complete_a_call_without_its_own_id() {
    let mut transport = row("http", "llm", "POST", "framework", "");
    transport.trace_id = "trace".into();
    transport.call_keys = vec!["transport:".parse().unwrap()];
    transport.call_evidence = Some(litellm_traces::CallEvidenceKind::Complete);
    let mut call = llm("llm", "agent", "agent", "");
    call.trace_id = "trace".into();
    let rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "key",
        ),
        owned(call, "team", "", "key"),
        owned(transport, "team", "", "key"),
    ];
    let mut logged = owned_spend("request", "", "team", "", "key", 0.5);
    logged.trace_id = "trace".into();
    logged.span_id = "http".into();
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, Some(0.5));
}

#[rstest]
#[case::lone_call(1, Some(0.5))]
#[case::two_calls(2, None)]
fn sibling_transports_belong_to_the_only_model_call_under_their_parent(
    #[case] calls: usize,
    #[case] expected: Option<f64>,
) {
    let mut transport = at(
        row("http", "step", "gateway.request", "framework", ""),
        2,
        10,
    );
    transport.trace_id = "trace".into();
    transport.call_keys = vec![litellm_traces::CallKey::GatewayAttempt];
    transport.call_evidence = Some(litellm_traces::CallEvidenceKind::Complete);
    let mut rows = vec![
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "key",
        ),
        owned(row("step", "agent", "step", "chain", ""), "team", "", "key"),
        owned(transport, "team", "", "key"),
    ];
    for index in 0..calls {
        let mut call = llm(&format!("chat-{index}"), "step", "agent", "");
        call.call_evidence = None;
        rows.push(owned(call, "team", "", "key"));
    }
    let mut logged = owned_spend("request", "", "team", "", "key", 0.5);
    logged.trace_id = "trace".into();
    logged.span_id = "http".into();
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
}

#[rstest]
#[case::without_tool_http_sibling(None, false, litellm_traces::CallKey::Transport, Some(0.5))]
#[case::after_call(Some((200, 10)), false, litellm_traces::CallKey::Transport, Some(0.5))]
#[case::inside_call_without_spend(Some((10, 10)), false, litellm_traces::CallKey::Transport, Some(0.5))]
#[case::inside_call_with_unrelated_spend(Some((10, 10)), true, litellm_traces::CallKey::Transport, Some(0.5))]
#[case::missing_gateway_attempt(Some((10, 10)), false, litellm_traces::CallKey::GatewayAttempt, None)]
fn sibling_transport_does_not_lose_model_call_spend(
    #[case] transport_timing: Option<(i64, u64)>,
    #[case] unrelated_spend: bool,
    #[case] key: litellm_traces::CallKey,
    #[case] expected: Option<f64>,
) {
    let call = owned(
        TraceSpansRow {
            trace_id: "trace".into(),
            call_keys: vec![litellm_traces::CallKey::ProviderResponse(
                "chatcmpl-1".into(),
            )],
            call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
            ..llm("chat", "step", "agent", "chatcmpl-1")
        },
        "team",
        "",
        "key",
    );
    let base_rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "key",
        ),
        owned(row("step", "agent", "step", "chain", ""), "team", "", "key"),
        call,
    ];
    let rows: Vec<_> = base_rows
        .into_iter()
        .chain(transport_timing.map(|(start, duration)| {
            let mut transport = at(
                row("tool-http", "step", "GET", "framework", ""),
                start,
                duration,
            );
            transport.trace_id = "trace".into();
            transport.call_keys = vec![key];
            transport.call_evidence = Some(litellm_traces::CallEvidenceKind::Complete);
            owned(transport, "team", "", "key")
        }))
        .collect();
    let logs: Vec<_> = std::iter::once(owned_spend(
        "chatcmpl-1",
        "chatcmpl-1",
        "team",
        "",
        "key",
        0.5,
    ))
    .chain(unrelated_spend.then(|| SpendByResponseIdsRow {
        trace_id: "trace".into(),
        span_id: "tool-http".into(),
        ..owned_spend("unrelated", "unrelated", "team", "", "key", 0.75)
    }))
    .collect();
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
}

#[rstest]
#[case::agreeing_ids(
    litellm_traces::CallKey::Transport,
    "call-a",
    Some("response-a"),
    Some(0.25)
)]
#[case::conflicting_gateway_id(litellm_traces::CallKey::Transport, "call-b", None, None)]
#[case::conflicting_response_id(
    litellm_traces::CallKey::Transport,
    "call-a",
    Some("response-b"),
    None
)]
#[case::conflicting_gateway_and_response(
    litellm_traces::CallKey::Transport,
    "call-b",
    Some("response-b"),
    None
)]
#[case::agreeing_gateway_attempt(
    litellm_traces::CallKey::GatewayAttempt,
    "call-a",
    Some("response-a"),
    Some(0.25)
)]
#[case::conflicting_gateway_attempt(litellm_traces::CallKey::GatewayAttempt, "call-b", None, None)]
fn gateway_attempt_identifiers_must_match_one_spend_row(
    #[case] transport: litellm_traces::CallKey,
    #[case] call_id: &str,
    #[case] response_id: Option<&str>,
    #[case] expected: Option<f64>,
) {
    let keys = [
        transport,
        litellm_traces::CallKey::LiteLlmRequest(call_id.into()),
    ]
    .into_iter()
    .chain(response_id.map(|id| litellm_traces::CallKey::ProviderResponse(id.into())))
    .collect();
    let rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "key",
        ),
        owned(llm("call", "agent", "agent", ""), "team", "", "key"),
        owned(
            TraceSpansRow {
                trace_id: "trace".into(),
                call_keys: keys,
                call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
                ..row("attempt", "call", "gateway.request", "framework", "")
            },
            "team",
            "",
            "key",
        ),
    ];
    let logs = [
        SpendByResponseIdsRow {
            litellm_call_id: "call-a".into(),
            trace_id: "trace".into(),
            span_id: "attempt".into(),
            ..owned_spend("request-a", "response-a", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            litellm_call_id: "call-b".into(),
            trace_id: "trace".into(),
            span_id: "other-attempt".into(),
            ..owned_spend("request-b", "response-b", "team", "", "key", 0.5)
        },
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
    assert_eq!(trace.spans[2].spend, expected);
}

#[rstest]
#[case::legacy_row("", Some(0.5))]
#[case::other_call("other-call", None)]
fn gateway_id_miss_only_vetoes_rows_that_carry_a_call_id(
    #[case] logged_call_id: &str,
    #[case] expected: Option<f64>,
) {
    let mut transport = row("http", "llm", "gateway.request", "framework", "");
    transport.trace_id = "trace".into();
    transport.call_keys = vec![
        "transport:".parse().unwrap(),
        "litellm_request:gateway-call".parse().unwrap(),
    ];
    transport.call_evidence = Some(litellm_traces::CallEvidenceKind::Complete);
    let mut call = llm("llm", "agent", "agent", "");
    call.trace_id = "trace".into();
    let rows = [
        owned(
            row("agent", "", "agent", "agent", "agent"),
            "team",
            "",
            "key",
        ),
        owned(call, "team", "", "key"),
        owned(transport, "team", "", "key"),
    ];
    let mut logged = owned_spend("chatcmpl-1", "chatcmpl-1", "team", "", "key", 0.5);
    logged.trace_id = "trace".into();
    logged.span_id = "http".into();
    logged.litellm_call_id = logged_call_id.into();
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, expected);
}

#[rstest]
#[case::narrows_ambiguity("request-a", Some(0.25))]
#[case::conflicting_exact_request("request-c", None)]
fn complete_wrapper_reconciles_ambiguous_response(
    #[case] exact_id: &str,
    #[case] expected: Option<f64>,
) {
    let wrapper = TraceSpansRow {
        call_keys: vec![litellm_traces::CallKey::LiteLlmRequest(exact_id.to_owned())],
        call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
        ..owned(llm("wrapper", "", "agent", ""), "team", "", "key")
    };
    let rows = [
        wrapper,
        owned(
            llm("call", "wrapper", "agent", "response"),
            "team",
            "",
            "key",
        ),
    ];
    let logs = [
        owned_spend("request-a", "response", "team", "", "key", 0.25),
        owned_spend("request-b", "response", "team", "", "key", 0.5),
        owned_spend("request-c", "other-response", "team", "", "key", 0.75),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
}

#[rstest]
#[case::complete_retry(true, litellm_traces::CallEvidenceKind::Complete, Some(0.75))]
#[case::missing_retry(false, litellm_traces::CallEvidenceKind::Complete, None)]
#[case::unknown_retry(true, litellm_traces::CallEvidenceKind::Unknown, None)]
#[case::partial_retry(true, litellm_traces::CallEvidenceKind::Partial, None)]
fn transports_preserve_retry_spend_without_counting_unrelated_cached_rows(
    #[case] retry_logged: bool,
    #[case] retry_evidence: litellm_traces::CallEvidenceKind,
    #[case] expected: Option<f64>,
) {
    let transport = |id: &str| {
        owned(
            TraceSpansRow {
                trace_id: "trace".into(),
                call_keys: vec!["transport:".parse().unwrap()],
                call_evidence: Some(if id == "first" {
                    retry_evidence
                } else {
                    litellm_traces::CallEvidenceKind::Complete
                }),
                ..row(id, "call", "POST", "framework", "")
            },
            "team",
            "",
            "key",
        )
    };
    let rows = [
        owned(
            llm("call", "", "agent", "final-response"),
            "team",
            "",
            "key",
        ),
        transport("first"),
        transport("second"),
    ];
    let logs = [
        SpendByResponseIdsRow {
            trace_id: "trace".into(),
            span_id: "first".into(),
            ..owned_spend("retry", "retry-response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            trace_id: "trace".into(),
            span_id: "second".into(),
            ..owned_spend("final", "final-response", "team", "", "key", 0.5)
        },
        owned_spend("cached", "final-response", "team", "", "key", 0.0),
    ];
    let available = if retry_logged { &logs[..] } else { &logs[1..] };
    let trace = resolve_trace("trace", "ref", &rows, available).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
}

#[rstest]
#[case::same_request(false)]
#[case::ambiguous_response(true)]
fn multiple_identifiers_for_one_request_count_its_spend_once(#[case] cached_row: bool) {
    let rows = [owned(
        TraceSpansRow {
            call_keys: vec![
                "provider_response:response".parse().unwrap(),
                "litellm_request:request".parse().unwrap(),
            ],
            call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
            ..llm("call", "", "agent", "response")
        },
        "team",
        "",
        "key",
    )];
    let logs = [
        owned_spend("request", "response", "team", "", "key", 0.25),
        owned_spend("cached", "response", "team", "", "key", 0.5),
    ];
    let available = if cached_row { &logs[..] } else { &logs[..1] };
    let trace = resolve_trace("trace", "ref", &rows, available).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.agents[0].spend, Some(0.25));
    assert_eq!(trace.spans[0].spend, Some(0.25));
}

#[rstest]
fn complete_wrapper_accounts_for_retries_missing_from_the_call_span() {
    let rows = [
        owned(
            TraceSpansRow {
                call_keys: vec![
                    "litellm_request:retry".parse().unwrap(),
                    "litellm_request:final".parse().unwrap(),
                ],
                call_evidence: Some(litellm_traces::CallEvidenceKind::Complete),
                ..llm("wrapper", "", "agent", "")
            },
            "team",
            "",
            "key",
        ),
        owned(
            llm("call", "wrapper", "agent", "response"),
            "team",
            "",
            "key",
        ),
    ];
    let logs = [
        owned_spend("retry", "retry-response", "team", "", "key", 0.25),
        owned_spend("final", "response", "team", "", "key", 0.5),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.75));
    assert_eq!(trace.agents[0].spend, Some(0.75));
}

#[rstest]
#[case::legacy(None, Some(0.25))]
#[case::unknown(Some(litellm_traces::CallEvidenceKind::Unknown), None)]
#[case::partial(Some(litellm_traces::CallEvidenceKind::Partial), None)]
#[case::complete(Some(litellm_traces::CallEvidenceKind::Complete), Some(0.25))]
fn legacy_request_id_fallback_respects_recorded_evidence(
    #[case] evidence: Option<litellm_traces::CallEvidenceKind>,
    #[case] expected: Option<f64>,
) {
    let span = owned(
        TraceSpansRow {
            call_evidence: evidence,
            ..llm("call", "", "agent", "response")
        },
        "team",
        "",
        "key",
    );
    let stored = serde_json::to_value(span).unwrap();
    let decoded: TraceSpansRow = serde_json::from_value(stored).unwrap();
    let logs = [owned_spend("request", "response", "team", "", "key", 0.25)];
    let trace = resolve_trace("trace", "ref", &[decoded], &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.spans[0].spend, expected);
}

#[rstest]
#[case::unknown(litellm_traces::CallEvidenceKind::Unknown)]
#[case::complete(litellm_traces::CallEvidenceKind::Complete)]
fn spend_lookup_fetches_recorded_keys_before_resolving_completeness(
    #[case] evidence: litellm_traces::CallEvidenceKind,
) {
    let recorded = TraceSpansRow {
        trace_id: "trace".to_owned(),
        call_keys: vec![
            litellm_traces::CallKey::ProviderResponse("response".to_owned()),
            litellm_traces::CallKey::LiteLlmRequest("request".to_owned()),
            litellm_traces::CallKey::Transport,
        ],
        call_evidence: Some(evidence),
        ..row("span", "", "operation", "llm", "")
    };
    let lookup = litellm_traces::SpendLookup::new(&[recorded]);
    assert_eq!(lookup.response_ids, ["response"]);
    assert_eq!(lookup.request_ids, ["request"]);
    assert_eq!(lookup.trace_ids, ["trace"]);
}

#[rstest]
#[case::gateway(false)]
#[case::transport(true)]
fn independent_key_disambiguates_repeated_request_ids(#[case] transport: bool) {
    let rows = [owned(
        TraceSpansRow {
            trace_id: "trace".into(),
            call_keys: vec![
                litellm_traces::CallKey::ProviderResponse("response".into()),
                if transport {
                    litellm_traces::CallKey::Transport
                } else {
                    litellm_traces::CallKey::LiteLlmRequest("gateway".into())
                },
            ],
            ..llm("call", "", "agent", "response")
        },
        "team",
        "",
        "key",
    )];
    let logs = [
        SpendByResponseIdsRow {
            start_ms: 100,
            litellm_call_id: "gateway".into(),
            trace_id: "trace".into(),
            span_id: "call".into(),
            ..owned_spend("same", "response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: 200,
            litellm_call_id: "other".into(),
            ..owned_spend("same", "response", "team", "", "key", 0.5)
        },
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.spans[0].spend, Some(0.25));
}

#[rstest]
fn conflicting_keys_cannot_agree_on_request_id_alone() {
    let rows = [
        owned(
            TraceSpansRow {
                call_keys: vec![litellm_traces::CallKey::LiteLlmRequest("gateway".into())],
                ..llm("wrapper", "", "agent", "")
            },
            "team",
            "",
            "key",
        ),
        owned(
            llm("call", "wrapper", "agent", "response"),
            "team",
            "",
            "key",
        ),
    ];
    let logs = [
        SpendByResponseIdsRow {
            start_ms: 100,
            ..owned_spend("same", "response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: 200,
            litellm_call_id: "gateway".into(),
            ..owned_spend("same", "other", "team", "", "key", 0.5)
        },
    ];
    assert_eq!(
        resolve_trace("trace", "ref", &rows, &logs)
            .unwrap()
            .summary
            .spend,
        None
    );
}

#[rstest]
#[case::gateway("gateway", "provider-id", "team", "key", Some(0.25))]
#[case::legacy("", "gateway", "team", "key", Some(0.25))]
#[case::conflict("other", "gateway", "team", "key", None)]
#[case::other_team("gateway", "provider-id", "other-team", "key", None)]
#[case::other_key("gateway", "provider-id", "team", "other-key", None)]
fn gateway_lookup_respects_legacy_fallback_and_ownership(
    #[case] call_id: &str,
    #[case] request_id: &str,
    #[case] team: &str,
    #[case] key: &str,
    #[case] expected: Option<f64>,
) {
    let rows = [owned(
        TraceSpansRow {
            call_keys: vec![litellm_traces::CallKey::LiteLlmRequest("gateway".into())],
            ..llm("call", "", "agent", "")
        },
        "team",
        "",
        "key",
    )];
    let logs = [SpendByResponseIdsRow {
        litellm_call_id: call_id.into(),
        ..owned_spend(request_id, "provider", team, "", key, 0.25)
    }];
    assert_eq!(
        resolve_trace("trace", "ref", &rows, &logs)
            .unwrap()
            .summary
            .spend,
        expected
    );
}

#[rstest]
#[case::native_id("provider_request:req_native", false, true)]
#[case::historical_id("provider_response:req_native", false, true)]
#[case::legacy_message_id("provider_response:msg_native", false, true)]
#[case::transport("", true, true)]
#[case::no_correlation("", false, false)]
fn native_cost_uses_typed_ids_or_the_original_transport(
    #[case] encoded: &str,
    #[case] traceparent: bool,
    #[case] priced: bool,
    #[values(false, true)] grouped: bool,
) {
    let call = TraceSpansRow {
        trace_id: if grouped {
            "session-trace"
        } else {
            "original-trace"
        }
        .into(),
        original_trace_id: if grouped { "original-trace" } else { "" }.into(),
        name: "claude_code.llm_request".into(),
        framework: "claude-code".into(),
        call_keys: encoded.parse().into_iter().collect(),
        call_evidence: Some(litellm_traces::CallEvidenceKind::Unknown),
        ..llm("native-call", "", "agent", "")
    };
    let log = SpendByResponseIdsRow {
        provider_request_id: "req_native".into(),
        trace_id: if traceparent { "original-trace" } else { "" }.into(),
        span_id: if traceparent { "native-call" } else { "" }.into(),
        ..spend("gateway-log", "msg_native", 0.25)
    };
    let lookup = litellm_traces::SpendLookup::new(std::slice::from_ref(&call));
    let trace = resolve_trace("trace", "ref", &[call], &[log]).unwrap();
    assert_eq!(
        (trace.summary.spend, trace.summary.priced_calls),
        (priced.then_some(0.25), u64::from(priced))
    );
    assert_eq!(trace.spans[0].spend, priced.then_some(0.25));
    if encoded.is_empty() {
        assert_eq!(lookup.trace_ids, ["original-trace"]);
    } else if !encoded.ends_with("msg_native") {
        assert_eq!(lookup.provider_request_ids, ["req_native"]);
        assert!(lookup.response_ids.is_empty());
    }
}

#[rstest]
fn provider_request_id_cannot_match_a_message_id_of_the_same_value() {
    let call = TraceSpansRow {
        call_keys: vec![litellm_traces::CallKey::ProviderRequest(
            "req_native".into(),
        )],
        ..llm("call", "", "agent", "")
    };
    let rows = [spend("unrelated", "req_native", 0.25)];
    let trace = resolve_trace("trace", "ref", &[call], &rows).unwrap();
    assert_eq!(trace.summary.spend, None);
    assert_eq!(trace.spans[0].spend_match, Some(SpendMatch::NoSpendLog));
}
