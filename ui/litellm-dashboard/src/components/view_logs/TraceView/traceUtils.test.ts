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
  fmtMs,
  GROUP_PAGE_SIZE,
  groupRowId,
  isFrameworkSpan,
  median,
  parseMessages,
  previewText,
  revealSpanInState,
  ROOT_KEY,
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

const span = (overrides: Partial<Span> & Pick<Span, "span_id">): Span => ({
  parent_span_id: null,
  name: overrides.span_id,
  type: "chain",
  agent: "root",
  start_offset_ms: 0,
  duration_ms: 1,
  status: "ok",
  error: null,
  input_preview: "",
  model: null,
  input_tokens: 0,
  output_tokens: 0,
  litellm_request_id: null,
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
    const spans = [
      span({ span_id: "root", type: "agent" }),
      span({ span_id: "mw", parent_span_id: "root", type: "framework", name: "X.wrap_model_call" }),
      span({ span_id: "model", parent_span_id: "mw", type: "chain", name: "model" }),
      span({ span_id: "llm", parent_span_id: "model", type: "llm", start_offset_ms: 5 }),
      span({ span_id: "step", parent_span_id: "root", type: "chain", name: "planner", start_offset_ms: 1 }),
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
    const failing = [0, 1, 2].map((i) =>
      span({
        span_id: `t${i}`,
        parent_span_id: "p",
        type: "tool",
        name: "grep_code",
        status: "error",
        start_offset_ms: i,
      }),
    );
    const rows = buildTreeRows([parent, ...failing], STATE);
    const group = groups(rows)[0];
    expect(group).toMatchObject({ name: "grep_code", failedCount: 3, isFailureGroup: true });
  });

  it("leaves 5 healthy same-named siblings unfolded", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = [0, 1, 2, 3, 4].map((i) =>
      span({ span_id: `k${i}`, parent_span_id: "p", type: "tool", name: "search" }),
    );
    expect(groups(buildTreeRows([parent, ...kids], STATE))).toHaveLength(0);
  });

  it("pages expanded groups 20 at a time with a load-more row", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = Array.from({ length: 45 }, (_, i) =>
      span({ span_id: `k${i}`, parent_span_id: "p", type: "agent", name: "worker", start_offset_ms: i }),
    );
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
    expect(errorSource(span({ span_id: "t", type: "tool", status: "error", error: "boom" }))).toBe("tool");
    expect(errorSource(span({ span_id: "l", type: "llm", status: "error", error: "429 Rate limit exceeded" }))).toBe(
      "litellm",
    );
    expect(errorSource(span({ span_id: "g", type: "llm", status: "error", error: "Blocked by guardrail" }))).toBe(
      "litellm",
    );
    expect(errorSource(span({ span_id: "m", type: "llm", status: "error", error: "context length exceeded" }))).toBe(
      "model",
    );
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
});
