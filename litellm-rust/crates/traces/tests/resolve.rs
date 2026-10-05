use litellm_traces::{
    AgentNode, SpanStatus, iso_time, listed_summary,
    query::named::{ListTracesRow, SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace,
};
use rstest::rstest;

const T0: i64 = 1_790_742_989_000_000_000;
const MS: i64 = 1_000_000;

fn row(span_id: &str, parent: &str, name: &str, kind: &str, agent: &str) -> TraceSpansRow {
    TraceSpansRow {
        trace_id: String::new(),
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
        team_id: String::new(),
        api_key_hash: String::new(),
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

fn owned(mut span: TraceSpansRow, team: &str, user: &str, key: &str) -> TraceSpansRow {
    span.team_id = team.into();
    span.user_id = user.into();
    span.api_key_hash = key.into();
    span
}

fn spend(
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
        trace_id: String::new(),
        span_id: String::new(),
        team_id: team.into(),
        api_key: key.into(),
        user: user.into(),
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
        spend("request-other", "response-1", "team-b", "", "key-b", 99.0),
        spend("request-1", "response-1", "team-a", "", "key-a", 0.25),
        spend(
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
fn ambiguous_response_id_keeps_cost_unknown() {
    let rows = [owned(
        llm("llm-1", "", "agent", "response-1"),
        "",
        "user",
        "key-a",
    )];
    let spend = [
        spend("response-1", "response-1", "", "user", "key-a", 0.25),
        spend(
            "response-1_cache_hit123",
            "response-1",
            "",
            "user",
            "key-a",
            0.0,
        ),
    ];
    let trace = resolve_trace("trace-1", "ref", &rows, &spend).unwrap();
    assert_eq!((trace.summary.spend, trace.spans[0].spend), (None, None));
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
    let spend = [spend(
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
fn incomplete_call_cost_never_becomes_a_partial_total(#[case] failure: &str) {
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
    let first = spend("first", "first", "team", "", "export", 0.25);
    let second = spend("second", "second", "team", "", "export", 0.25);
    let duplicate = spend("duplicate", "second", "team", "", "export", 0.25);
    let spend = if failure == "duplicate_spend" {
        vec![first, second, duplicate]
    } else {
        vec![first]
    };
    let trace = resolve_trace("trace", "ref", &rows, &spend).unwrap();
    assert_eq!(trace.spans[1].spend, Some(0.25));
    assert_eq!(trace.spans[2].spend, None);
    assert_eq!(trace.summary.spend, None);
    assert_eq!(trace.agents[0].spend, None);
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
    let mut logged = spend("request", "", "team", "", "key", 0.5);
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
    let mut logged = spend("request", "", "team", "", "key", 0.5);
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
    let logs: Vec<_> = std::iter::once(spend("chatcmpl-1", "chatcmpl-1", "team", "", "key", 0.5))
        .chain(unrelated_spend.then(|| SpendByResponseIdsRow {
            trace_id: "trace".into(),
            span_id: "tool-http".into(),
            ..spend("unrelated", "unrelated", "team", "", "key", 0.75)
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
            ..spend("request-a", "response-a", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            litellm_call_id: "call-b".into(),
            trace_id: "trace".into(),
            span_id: "other-attempt".into(),
            ..spend("request-b", "response-b", "team", "", "key", 0.5)
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
    let mut logged = spend("chatcmpl-1", "chatcmpl-1", "team", "", "key", 0.5);
    logged.trace_id = "trace".into();
    logged.span_id = "http".into();
    logged.litellm_call_id = logged_call_id.into();
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, expected);
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
        spend("request-a", "response", "team", "", "key", 0.25),
        spend("request-b", "response", "team", "", "key", 0.5),
        spend("request-c", "other-response", "team", "", "key", 0.75),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
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
    let rows = [owned(
        llm("call", "", "agent", "response"),
        "team",
        "",
        "key",
    )];
    let logged = SpendByResponseIdsRow {
        spend: cost,
        ..spend("request", "response", "team", "", "key", 0.25)
    };
    let trace = resolve_trace("trace", "ref", &rows, &[logged]).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
    assert_eq!(trace.spans[0].spend, expected);
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
            ..spend("retry", "retry-response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            trace_id: "trace".into(),
            span_id: "second".into(),
            ..spend("final", "final-response", "team", "", "key", 0.5)
        },
        spend("cached", "final-response", "team", "", "key", 0.0),
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
        spend("request", "response", "team", "", "key", 0.25),
        spend("cached", "response", "team", "", "key", 0.5),
    ];
    let available = if cached_row { &logs[..] } else { &logs[..1] };
    let trace = resolve_trace("trace", "ref", &rows, available).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.agents[0].spend, Some(0.25));
    assert_eq!(trace.spans[0].spend, Some(0.25));
}

#[rstest]
#[case::finite(0.25, Some(0.5))]
#[case::overflow(f64::MAX, None)]
fn trace_cost_requires_a_finite_total(#[case] cost: f64, #[case] expected: Option<f64>) {
    let rows = [
        owned(llm("first", "", "agent", "response-a"), "team", "", "key"),
        owned(llm("second", "", "agent", "response-b"), "team", "", "key"),
    ];
    let logs = [
        spend("request-a", "response-a", "team", "", "key", cost),
        spend("request-b", "response-b", "team", "", "key", cost),
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, expected);
    assert_eq!(trace.agents[0].spend, expected);
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
        spend("retry", "retry-response", "team", "", "key", 0.25),
        spend("final", "response", "team", "", "key", 0.5),
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
    let logs = [spend("request", "response", "team", "", "key", 0.25)];
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
        ..spend("same", "response", "team", "", "key", 0.25)
    };
    let second = SpendByResponseIdsRow {
        start_ms: 200,
        ..spend("same", "response", "team", "", "key", 0.5)
    };
    let logs = if reverse {
        [second, first]
    } else {
        [first, second]
    };
    let rows = [owned(
        llm("call", "", "agent", "response"),
        "team",
        "",
        "key",
    )];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, None);
    assert_eq!(trace.spans[0].spend, None);
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
            ..spend("same", "response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: 200,
            litellm_call_id: "other".into(),
            ..spend("same", "response", "team", "", "key", 0.5)
        },
    ];
    let trace = resolve_trace("trace", "ref", &rows, &logs).unwrap();
    assert_eq!(trace.summary.spend, Some(0.25));
    assert_eq!(trace.spans[0].spend, Some(0.25));
}

#[rstest]
#[case::same_row(true, Some(0.25))]
#[case::distinct_rows(false, Some(0.75))]
fn totals_deduplicate_only_equal_storage_identities(
    #[case] duplicate: bool,
    #[case] expected: Option<f64>,
) {
    let rows = [
        owned(llm("first", "", "agent", "a"), "team", "", "key"),
        owned(
            llm("second", "", "agent", if duplicate { "a" } else { "b" }),
            "team",
            "",
            "key",
        ),
    ];
    let logs = [
        SpendByResponseIdsRow {
            start_ms: 100,
            ..spend("same", "a", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: if duplicate { 100 } else { 200 },
            ..spend(
                "same",
                if duplicate { "a" } else { "b" },
                "team",
                "",
                "key",
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
            ..spend("same", "response", "team", "", "key", 0.25)
        },
        SpendByResponseIdsRow {
            start_ms: 200,
            litellm_call_id: "gateway".into(),
            ..spend("same", "other", "team", "", "key", 0.5)
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
        ..spend(request_id, "provider", team, "", key, 0.25)
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
