import type { Span, SpanDetail, Trace } from "@/components/lens/traces/types";

export function withReleaseCases(run: { trace: Trace; details: SpanDetail[] }) {
  const { trace } = run;
  const root = trace.spans[0];
  const final = trace.spans.at(-1)!;
  const caseCount = 120;
  const caseSpans: Span[] = [];
  const caseDetails: SpanDetail[] = [];
  const checks = ["Unicode queries", "Empty results", "Pagination", "Ranking", "Filters", "Permissions"];
  const failedCases = new Set([17, 63, 104]);
  for (let index = 1; index <= caseCount; index++) {
    const failed = failedCases.has(index);
    const id = (step: number) => (0x70000 + index * 10 + step).toString(16).padStart(16, "0");
    const question = `Case ${index}: ${checks[(index - 1) % checks.length]}`;
    const result = failed ? "Expected matching results; received an empty result set" : "Expected results matched";
    const start = (index - 1) * 2200;
    const agent: Span = {
      ...root,
      span_id: id(0),
      parent_span_id: root.span_id,
      agent: "search_case",
      name: "search_case",
      input_preview: question,
      start_offset_ms: start,
      duration_ms: 2100,
    };
    const tool: Span = {
      ...agent,
      span_id: id(1),
      parent_span_id: agent.span_id,
      name: "run_search_check",
      type: "tool",
      start_offset_ms: start + 100,
      duration_ms: 800,
      status: failed ? "error" : "ok",
      error: failed ? result : null,
    };
    const model: Span = {
      ...final,
      span_id: id(2),
      parent_span_id: agent.span_id,
      agent: agent.agent,
      name: "Review case result",
      input_preview: question,
      start_offset_ms: start + 950,
      duration_ms: 1100,
    };
    caseSpans.push(agent, tool, model);
    for (const span of [agent, tool, model]) {
      const detail: SpanDetail = {
        span_id: span.span_id,
        input:
          span.type === "tool"
            ? JSON.stringify({ case: index, check: question })
            : JSON.stringify([{ role: "user", content: question }]),
        output:
          span.type === "tool"
            ? result
            : JSON.stringify([
                {
                  role: "assistant",
                  content: failed ? `Hold this case for review. ${result}.` : `Case ${index} passed. ${result}.`,
                },
              ]),
        attributes: { "gen_ai.agent.name": "search_case", "test.case": String(index), demo: "true" },
      };
      caseDetails.push(detail);
    }
  }
  const finalSpan = { ...final, start_offset_ms: caseCount * 2200 };
  const duration = finalSpan.start_offset_ms + finalSpan.duration_ms;
  const spans = [{ ...root, duration_ms: duration }, ...caseSpans, finalSpan];
  const models = spans.filter((span) => span.type === "llm");
  const summary = {
    ...trace.summary,
    duration_ms: duration,
    span_count: spans.length,
    agent_count: 2,
    agent_invocations: caseCount + 1,
    agent_names: [root.name, "search_case"],
    llm_calls: models.length,
    tool_calls: caseCount,
    error_count: failedCases.size,
    input_tokens: models.reduce((sum, span) => sum + span.input_tokens, 0),
    output_tokens: models.reduce((sum, span) => sum + span.output_tokens, 0),
    spend: models.reduce((sum, span) => sum + (span.spend ?? 0), 0),
  };
  const finalDetail = run.details.at(-1)!;
  const history = caseDetails.filter((_, index) => index % 3 === 2);
  return {
    trace: {
      summary,
      spans,
      agents: [
        { ...trace.agents[0], duration_ms: duration, tool_calls: 0 },
        {
          name: "search_case",
          parent_agent: root.name,
          duration_ms: caseCount * 2100,
          invocations: caseCount,
          llm_calls: caseCount,
          tool_calls: caseCount,
          spend: summary.spend - (final.spend ?? 0),
        },
      ],
    },
    details: [
      run.details[0],
      ...caseDetails,
      {
        ...finalDetail,
        input: JSON.stringify(
          history.map((detail) => ({ role: "user", content: JSON.parse(detail.output)[0].content })),
        ),
      },
    ],
  };
}
