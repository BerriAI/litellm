import { describe, expect, it } from "vitest";

import deepAgentTrace from "./__fixtures__/deep_agent_trace.json";
import researchTrace from "./__fixtures__/research_trace.json";
import swarmTrace from "./__fixtures__/swarm_trace.json";
import { type SpanTreeState, type TreeRow } from "./traceTree";
import type { Span, Trace } from "./traceTypes";
import {
  buildTreeRows,
  buildVisibleTree,
  errorSource,
  firstErrorSpan,
  findTraceSteps,
  fmtMs,
  GROUP_PAGE_SIZE,
  groupRowId,
  isFrameworkSpan,
  median,
  messageText,
  parseMessages,
  previewText,
  revealSpanInState,
  ROOT_KEY,
  treeGuides,
} from "./traceUtils";

const swarm = swarmTrace as Trace;
const research = researchTrace as Trace;
const deepAgent = deepAgentTrace as Trace;

const STATE: SpanTreeState = {
  hideFramework: true,
  collapsedSpanIds: new Set(),
  expandedGroupIds: new Set(),
  groupRevealCounts: {},
};

type SpanOverrides = Partial<Span> & Pick<Span, "span_id">;

const span = (overrides: SpanOverrides): Span => ({
  parent_span_id: null,
  name: overrides.span_id,
  type: "chain",
  agent: "root",
  start_offset_ms: 0,
  duration_ms: 1,
  status: "ok",
  error: null,
  error_truncated: false,
  framework: "",
  input_preview: "",
  model: null,
  input_tokens: 0,
  output_tokens: 0,
  litellm_request_id: null,
  spend: null,
  ...overrides,
});

const groups = (rows: TreeRow[]) => rows.filter((r): r is Extract<TreeRow, { kind: "group" }> => r.kind === "group");
const spanRows = (rows: TreeRow[]) => rows.filter((r): r is Extract<TreeRow, { kind: "span" }> => r.kind === "span");

describe("formatting", () => {
  it("formats durations", () => {
    expect(fmtMs(4.25)).toBe("4.3ms");
    expect(fmtMs(950)).toBe("950ms");
    expect(fmtMs(51386)).toBe("51.39s");
  });

  it("pulls the user message out of a truncated JSON preview", () => {
    expect(previewText('[{"role": "user", "content": "Customer acme-404 says billing is wrong."}]')).toBe(
      "Customer acme-404 says billing is wrong.",
    );
    expect(previewText('[{"role": "system", "content": "sys"}, {"role": "user", "content": "Compare ingest thr')).toBe(
      "Compare ingest thr",
    );
    expect(previewText("plain text")).toBe("plain text");
  });

  it("decodes JSON escapes in a truncated preview and treats an empty array as no preview", () => {
    expect(previewText('[{"role": "user", "content": "Thanks\\u2014keep the \\u00a55,000 budget\\nplease \\u20')).toBe(
      "Thanks—keep the ¥5,000 budget please ",
    );
    expect(previewText("[]")).toBe("");
  });

  it("pulls the user message out of an OpenInference LangChain input", () => {
    const input =
      '{"messages": [{"type": "human", "data": {"content": "Customer acme-7 keeps hitting 429s", "type": "human"';
    expect(previewText(input)).toBe("Customer acme-7 keeps hitting 429s");
  });

  it("takes the upper median", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(3);
    expect(median([])).toBe(0);
  });
});

describe("buildVisibleTree / isFrameworkSpan", () => {
  it("hides framework spans and re-parents their children", () => {
    const middleware: SpanOverrides = {
      span_id: "mw",
      parent_span_id: "root",
      type: "framework",
      name: "X.wrap_model_call",
    };
    const modelNode: SpanOverrides = { span_id: "model", parent_span_id: "mw", type: "chain", name: "model" };
    const llmCall: SpanOverrides = { span_id: "llm", parent_span_id: "model", type: "llm", start_offset_ms: 5 };
    const planner: SpanOverrides = {
      span_id: "step",
      parent_span_id: "root",
      type: "chain",
      name: "planner",
      start_offset_ms: 1,
    };
    const spans = [
      span({ span_id: "root", type: "agent" }),
      span(middleware),
      span(modelNode),
      span(llmCall),
      span(planner),
    ];
    const compact = buildVisibleTree(spans, false);
    expect(compact.children.get(ROOT_KEY)?.map((s) => s.span_id)).toEqual(["root"]);
    expect(compact.children.get("root")?.map((s) => s.span_id)).toEqual(["step", "llm"]);
    expect(buildVisibleTree(spans, true).visibleCount).toBe(5);
  });

  it("never hides the root span", () => {
    expect(isFrameworkSpan(span({ span_id: "r", type: "framework" }))).toBe(false);
  });
});

describe("buildTreeRows", () => {
  it("starts the tree at the root span", () => {
    const rows = buildTreeRows(research.spans, STATE);
    expect(rows[0]).toMatchObject({
      kind: "span",
      depth: 0,
      id: research.spans.find((s) => s.parent_span_id === null)?.span_id,
    });
  });

  it("hides every framework / graph-node span of the real Deep Agents trace, keeping all LLM calls", () => {
    const rows = buildTreeRows(research.spans, STATE);
    const shown = new Set(spanRows(rows).map((r) => r.span.span_id));
    expect(research.spans.filter(isFrameworkSpan).every((s) => !shown.has(s.span_id))).toBe(true);
    const all = buildTreeRows(research.spans, { ...STATE, hideFramework: false });
    expect(spanRows(all).length).toBeGreaterThan(spanRows(rows).length);
  });

  it("folds the swarm's 12 researcher invocations into one group row", () => {
    const rows = buildTreeRows(swarm.spans, STATE);
    const researcher = groups(rows).find((g) => g.name === "researcher" && g.type === "agent");
    expect(researcher?.members).toHaveLength(12);
    expect(researcher?.expanded).toBe(false);
    expect(researcher?.p50Duration).toBeGreaterThan(0);
    // folded members are not rendered until the group is expanded
    expect(spanRows(rows).some((r) => r.span.name === "researcher")).toBe(false);
  });

  it("folds as few as 3 failed siblings of the same tool into a failure group", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const failing = [0, 1, 2].map((i) => {
      const failedGrep: SpanOverrides = {
        span_id: `t${i}`,
        parent_span_id: "p",
        type: "tool",
        name: "grep_code",
        status: "error",
        start_offset_ms: i,
      };
      return span(failedGrep);
    });
    const rows = buildTreeRows([parent, ...failing], STATE);
    const group = groups(rows)[0];
    expect(group).toMatchObject({ name: "grep_code", failedCount: 3, isFailureGroup: true });
  });

  it("never folds same-named calls from different agents into one group", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const calls = [0, 1, 2, 3, 4, 5].map((i) => {
      const call: SpanOverrides = {
        span_id: `c${i}`,
        parent_span_id: "p",
        type: "llm",
        name: "ChatOpenAI",
        agent: i < 3 ? "planner" : "critic",
      };
      return span(call);
    });
    const rows = buildTreeRows([parent, ...calls], STATE);
    expect(groups(rows)).toHaveLength(0);
    expect(spanRows(rows).filter((r) => r.span.name === "ChatOpenAI")).toHaveLength(6);
  });

  it("leaves 5 healthy same-named siblings unfolded", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = [0, 1, 2, 3, 4].map((i) => {
      const search: SpanOverrides = { span_id: `k${i}`, parent_span_id: "p", type: "tool", name: "search" };
      return span(search);
    });
    expect(groups(buildTreeRows([parent, ...kids], STATE))).toHaveLength(0);
  });

  it("pages expanded groups 20 at a time with a load-more row", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = Array.from({ length: 45 }, (_, i) => {
      const worker: SpanOverrides = {
        span_id: `k${i}`,
        parent_span_id: "p",
        type: "agent",
        name: "worker",
        start_offset_ms: i,
      };
      return span(worker);
    });
    const all = [parent, ...kids];
    const id = groupRowId("p", kids[0]);
    const page1 = buildTreeRows(all, { ...STATE, expandedGroupIds: new Set([id]) });
    expect(spanRows(page1).filter((r) => r.span.name === "worker")).toHaveLength(GROUP_PAGE_SIZE);
    expect(page1.find((r) => r.kind === "load-more")).toMatchObject({ groupId: id, remaining: 25 });
    const everything = buildTreeRows(all, {
      ...STATE,
      expandedGroupIds: new Set([id]),
      groupRevealCounts: { [id]: 60 },
    });
    expect(spanRows(everything).filter((r) => r.span.name === "worker")).toHaveLength(45);
    expect(everything.some((r) => r.kind === "load-more")).toBe(false);
  });

  it("hides the children of collapsed spans", () => {
    const root = swarm.spans.find((s) => s.parent_span_id === null) as Span;
    const rows = buildTreeRows(swarm.spans, { ...STATE, collapsedSpanIds: new Set([root.span_id]) });
    expect(rows.map((r) => r.kind)).toEqual(["span"]);
  });
});

describe("revealSpanInState", () => {
  it("opens the path to a span nested in a folded group so the view can land on it", () => {
    const failed = firstErrorSpan(swarm.spans) as Span;
    const state = revealSpanInState(swarm.spans, STATE, failed.span_id);
    const rows = buildTreeRows(swarm.spans, state);
    expect(rows.some((r) => r.id === failed.span_id)).toBe(true);
  });
});

describe("errorSource", () => {
  it("blames the tool, LiteLLM, or the model", () => {
    expect(errorSource(span({ span_id: "a", status: "ok" }))).toBeNull();
    const failure = (span_id: string, type: Span["type"], error: string): Span => {
      const failed: SpanOverrides = { span_id, type, error, status: "error" };
      return span(failed);
    };
    expect(errorSource(failure("t", "tool", "boom"))).toBe("tool");
    expect(errorSource(failure("l", "llm", "429 Rate limit exceeded"))).toBe("litellm");
    expect(errorSource(failure("g", "llm", "Blocked by guardrail"))).toBe("litellm");
    expect(errorSource(failure("m", "llm", "context length exceeded"))).toBe("model");
  });

  it("classifies the swarm's failing lookup_benchmark calls as tool errors", () => {
    const failedTool = swarm.spans.find((s) => s.status === "error" && s.type === "tool") as Span;
    expect(failedTool.name).toBe("lookup_benchmark");
    expect(errorSource(failedTool)).toBe("tool");
  });
});

describe("payload helpers", () => {
  it("finds the earliest failing non-root span", () => {
    const failed = firstErrorSpan(swarm.spans);
    expect(failed?.status).toBe("error");
    expect(failed?.parent_span_id).not.toBeNull();
    expect(firstErrorSpan(deepAgent.spans)).toBeNull();
  });

  it("parses llm message payloads and rejects non-message JSON", () => {
    expect(parseMessages('[{"role":"user","content":"hi"}]')).toEqual([{ role: "user", content: "hi" }]);
    expect(parseMessages('{"file_path":"/tmp/x"}')).toBeNull();
    expect(parseMessages("not json")).toBeNull();
  });

  it("shows block-list message content as its text and drops reasoning blocks", () => {
    const reasoning = { type: "reasoning", summary: [], encrypted_content: "gAAAAB-opaque" };
    const content = JSON.stringify([reasoning, { type: "text", text: "Part one" }, { type: "text", text: "Part two" }]);
    const [message] = parseMessages(JSON.stringify({ role: "assistant", content })) ?? [];
    expect(message.content).toBe("Part one\n\nPart two");
    expect(messageText(JSON.stringify([reasoning]))).toBe("");
    const image = JSON.stringify([{ type: "image_url", image_url: { url: "https://x.test/a.png" } }]);
    expect(messageText(image)).toBe(image);
    expect(messageText("[not json")).toBe("[not json");
  });

  it("reads GenAI message parts and native content arrays without crashing previews", () => {
    const question = "What is an agent trace?";
    const parts = [{ type: "text", content: question }];
    const input = JSON.stringify([{ role: "user", parts }]);
    expect(parseMessages(input)).toEqual([{ role: "user", parts, content: question }]);
    expect(previewText(input)).toBe(question);
    expect(
      parseMessages(JSON.stringify({ role: "assistant", content: [{ type: "text", text: "An execution record" }] })),
    ).toEqual([{ role: "assistant", content: "An execution record" }]);
    expect(parseMessages('[{"role":"assistant","tool_calls":[]}]')).toBeNull();
    expect(parseMessages('[{"role":"user","content":42}]')).toBeNull();
  });
});

describe("treeGuides", () => {
  it("draws a rail only for ancestors that still have later siblings", () => {
    const guides = treeGuides([0, 1, 2, 2, 1, 2, 3]);
    expect(guides.map((g) => g.last)).toEqual([true, false, false, true, true, true, true]);
    expect(guides[2].rails).toEqual([true]);
    expect(guides[5].rails).toEqual([false]);
    expect(guides[6].rails).toEqual([false, false]);
  });

  it("treats a sibling after a deeper subtree as continuing the branch", () => {
    const guides = treeGuides([0, 1, 2, 3, 1]);
    expect(guides[1].last).toBe(false);
    expect(guides[3].rails).toEqual([true, false]);
  });

  it("drops a stem from a row only when the next row is its child", () => {
    const guides = treeGuides([0, 1, 2, 1, 1, 0]);
    expect(guides.map((g) => g.stem)).toEqual([true, true, false, false, false, false]);
  });
});

describe("findTraceSteps", () => {
  it("searches names, agents, models, IDs and inputs in time order without changing the trace", () => {
    const spans = [
      span({ span_id: "late", name: "Check", start_offset_ms: 20 }),
      span({ span_id: "early", model: "check-model", start_offset_ms: 1 }),
      span({ span_id: "agent", agent: "check-agent", start_offset_ms: 2 }),
      span({ span_id: "check-id", start_offset_ms: 3 }),
      span({ span_id: "input", input_preview: "Check this case", start_offset_ms: 4 }),
    ];
    expect(findTraceSteps(spans, " CHECK ", false, true).map((item) => item.span_id)).toEqual([
      "early",
      "agent",
      "check-id",
      "input",
      "late",
    ]);
    expect(spans[0].span_id).toBe("late");
  });

  it("combines search and errors while respecting the framework display setting", () => {
    const tool: SpanOverrides = { span_id: "tool", name: "check", type: "tool", status: "error" };
    const framework: SpanOverrides = {
      span_id: "framework",
      parent_span_id: "root",
      name: "check",
      type: "framework",
      status: "error",
    };
    const spans = [span(tool), span({ span_id: "ok", name: "check", type: "tool" }), span(framework)];
    expect(findTraceSteps(spans, "check", true, true).map((item) => item.span_id)).toEqual(["tool"]);
    expect(findTraceSteps(spans, "check", true, false).map((item) => item.span_id)).toEqual(["tool", "framework"]);
    expect(findTraceSteps(spans, "missing", false, false)).toEqual([]);
  });
});
