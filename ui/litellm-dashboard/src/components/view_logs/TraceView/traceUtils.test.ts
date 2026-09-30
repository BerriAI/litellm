import { describe, expect, it } from "vitest";

import deepAgentTrace from "./__fixtures__/deep_agent_trace.json";
import researchTrace from "./__fixtures__/research_trace.json";
import swarmTrace from "./__fixtures__/swarm_trace.json";
import { groupSummary } from "./SpanTree";
import type { Span, Trace } from "./traceTypes";
import {
  agentBadgeLabel,
  buildVisibleTree,
  firstErrorSpan,
  flattenTree,
  fmtCost,
  fmtMs,
  GROUP_PAGE_SIZE,
  groupSiblingAgents,
  interleaveByTime,
  isFrameworkSpan,
  layoutAgentGraph,
  median,
  parseMessages,
  previewText,
  revealSpan,
  ROOT_KEY,
  sortStepsByCost,
  spanLabel,
  spanRowIds,
  stepsFromSpans,
  subtreeStats,
  traceHasErrors,
  type TreeUiState,
} from "./traceUtils";

const swarm = swarmTrace as Trace;
const research = researchTrace as Trace;
const deepAgent = deepAgentTrace as Trace;
const EMPTY_UI: TreeUiState = { collapsed: new Set(), groupShown: {} };

const span = (overrides: Partial<Span> & Pick<Span, "span_id">): Span => ({
  parent_span_id: null,
  name: overrides.span_id,
  type: "chain",
  agent: "root",
  start_offset_ms: 0,
  duration_ms: 1,
  status: "ok",
  input_preview: "",
  model: null,
  input_tokens: 0,
  output_tokens: 0,
  litellm: null,
  ...overrides,
});

describe("formatting", () => {
  it("formats durations and costs the way the table shows them", () => {
    expect(fmtMs(4.25)).toBe("4.3ms");
    expect(fmtMs(950)).toBe("950ms");
    expect(fmtMs(51386)).toBe("51.39s");
    expect(fmtCost(0.0832806)).toBe("$0.0833");
    expect(fmtCost(0.0004)).toBe("$0.00040");
    expect(fmtCost(null)).toBe("—");
  });

  it("labels agent rows with agent, LLM and tool counts", () => {
    expect(agentBadgeLabel(swarm.summary)).toBe("◆ 4 agents · 53 LLM · 45 tool");
    expect(agentBadgeLabel({ agent_count: 1, llm_calls: 2, tool_calls: 1 })).toBe("◆ Agent · 2 LLM · 1 tool");
  });

  it("headlines LLM spans with the model group, not the deployment", () => {
    const llm = swarm.spans.find((s) => s.litellm?.model.startsWith("openai/"));
    expect(llm).toBeDefined();
    expect(spanLabel(llm as Span)).toBe((llm as Span).litellm?.model_group);
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

  it("takes the median of an even and an odd list", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBe(0);
  });
});

describe("buildVisibleTree", () => {
  it("hides framework spans and re-parents their children to the nearest visible ancestor", () => {
    const middleware = { span_id: "mw", parent_span_id: "root", type: "framework", name: "X.wrap_model_call" } as const;
    const model = { span_id: "model", parent_span_id: "mw", type: "chain", name: "model" } as const;
    const llm = { span_id: "llm", parent_span_id: "model", type: "llm", start_offset_ms: 5 } as const;
    const step = {
      span_id: "step",
      parent_span_id: "root",
      type: "chain",
      name: "planner",
      start_offset_ms: 1,
    } as const;
    const spans = [span({ span_id: "root", type: "agent" }), span(middleware), span(model), span(llm), span(step)];
    const compact = buildVisibleTree(spans, false);
    expect(compact.visibleCount).toBe(3);
    expect(compact.children.get(ROOT_KEY)?.map((s) => s.span_id)).toEqual(["root"]);
    expect(compact.children.get("root")?.map((s) => s.span_id)).toEqual(["step", "llm"]);

    const raw = buildVisibleTree(spans, true);
    expect(raw.visibleCount).toBe(5);
    expect(raw.children.get("model")?.map((s) => s.span_id)).toEqual(["llm"]);
  });

  it("never hides the root span even if it looks like framework plumbing", () => {
    expect(isFrameworkSpan(span({ span_id: "r", type: "framework" }))).toBe(false);
  });

  it("drops every framework / graph-node span of the real Deep Agents trace", () => {
    const { visibleCount, children } = buildVisibleTree(research.spans, false);
    const hidden = research.spans.filter(isFrameworkSpan).length;
    expect(hidden).toBeGreaterThan(100);
    expect(visibleCount).toBe(research.spans.length - hidden);
    const visibleIds = new Set([...children.values()].flat().map((s) => s.span_id));
    expect(research.spans.filter((s) => s.type === "llm").every((s) => visibleIds.has(s.span_id))).toBe(true);
  });
});

describe("groupSiblingAgents", () => {
  const stats = subtreeStats(swarm.spans);
  const children = buildVisibleTree(swarm.spans, false).children;
  const root = swarm.spans.find((s) => s.parent_span_id === null) as Span;

  it("collapses more than 10 same-named sibling agents into one group row", () => {
    const items = groupSiblingAgents(root.span_id, children.get(root.span_id) ?? [], stats);
    const groups = items.filter((i) => i.kind === "group");
    expect(groups).toHaveLength(1);
    const group = groups[0].kind === "group" ? groups[0].group : null;
    expect(group?.name).toBe("researcher");
    expect(group?.spans).toHaveLength(12);
    // critic (1 invocation) stays a plain row
    expect(items.some((i) => i.kind === "span" && i.span.name === "critic")).toBe(true);
  });

  it("rolls up subtree spend and counts invocations that contain errors", () => {
    const items = groupSiblingAgents(root.span_id, children.get(root.span_id) ?? [], stats);
    const group = items.find((i) => i.kind === "group");
    if (group?.kind !== "group") throw new Error("expected a group");
    const researcherAgent = swarm.agents.find((a) => a.name === "researcher");
    const factChecker = swarm.agents.find((a) => a.name === "fact_checker");
    // researcher subtrees include the nested fact_checker invocations
    expect(group.group.spend).toBeCloseTo((researcherAgent?.spend ?? 0) + (factChecker?.spend ?? 0), 6);
    expect(group.group.errors).toBeGreaterThan(0);
    expect(group.group.p50Ms).toBeGreaterThan(0);
  });

  it("leaves groups of 10 or fewer alone", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = Array.from({ length: 10 }, (_, i) => {
      const props = { span_id: `k${i}`, parent_span_id: "p", type: "agent", name: "worker" } as const;
      return span(props);
    });
    const items = groupSiblingAgents("p", kids, subtreeStats([parent, ...kids]));
    expect(items.every((i) => i.kind === "span")).toBe(true);
  });

  it("does not show a zero cost when grouped spans have no spend rows", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const children = Array.from({ length: 11 }, (_, index) => {
      const props = { span_id: `child-${index}`, parent_span_id: "p", type: "agent", name: "worker" } as const;
      return span(props);
    });
    const items = groupSiblingAgents("p", children, subtreeStats([parent, ...children]));
    const group = items.find((item) => item.kind === "group");
    if (group?.kind !== "group") throw new Error("expected a group");
    expect(groupSummary(group.group)).toBe("worker ×11 · p50 1.0ms");
  });
});

describe("flattenTree + revealSpan", () => {
  const children = buildVisibleTree(swarm.spans, false).children;
  const stats = subtreeStats(swarm.spans);

  it("shows the group summary row but none of its invocations by default", () => {
    const rows = flattenTree(children, stats, EMPTY_UI);
    expect(rows.filter((r) => r.kind === "group")).toHaveLength(1);
    expect(rows.some((r) => r.kind === "span" && r.span.name === "researcher")).toBe(false);
  });

  it("pages group invocations 20 at a time with a 'more' row until all are shown", () => {
    const parent = span({ span_id: "p", type: "agent" });
    const kids = Array.from({ length: 45 }, (_, i) => {
      const props = {
        span_id: `k${i}`,
        parent_span_id: "p",
        type: "agent",
        name: "worker",
        start_offset_ms: i,
      } as const;
      return span(props);
    });
    const all = [parent, ...kids];
    const tree = buildVisibleTree(all, false).children;
    const key = "p::worker";
    const page1 = flattenTree(tree, subtreeStats(all), {
      collapsed: new Set(),
      groupShown: { [key]: GROUP_PAGE_SIZE },
    });
    expect(page1.filter((r) => r.kind === "span" && r.span.name === "worker")).toHaveLength(20);
    expect(page1.at(-1)?.kind).toBe("more");
    const page3 = flattenTree(tree, subtreeStats(all), { collapsed: new Set(), groupShown: { [key]: 60 } });
    expect(page3.filter((r) => r.kind === "span" && r.span.name === "worker")).toHaveLength(45);
    expect(page3.some((r) => r.kind === "more")).toBe(false);
  });

  it("reveals a span nested inside a collapsed group so the drawer can open on it", () => {
    const failed = firstErrorSpan(swarm.spans) as Span;
    const ui = revealSpan(children, stats, EMPTY_UI, failed.span_id);
    expect(spanRowIds(flattenTree(children, stats, ui))).toContain(failed.span_id);
  });

  it("hides children of collapsed spans", () => {
    const root = swarm.spans.find((s) => s.parent_span_id === null) as Span;
    const rows = flattenTree(children, stats, { collapsed: new Set([root.span_id]), groupShown: {} });
    expect(rows).toHaveLength(1);
  });
});

describe("stepsFromSpans", () => {
  it("numbers every LLM call, tool call and subagent invocation in time order", () => {
    const steps = stepsFromSpans(deepAgent.spans);
    expect(steps).toHaveLength(7 + 26 + 1);
    expect(steps.map((s) => s.number)).toEqual(steps.map((_, i) => i + 1));
    const starts = steps.map((s) => s.span.start_offset_ms);
    expect(starts).toEqual([...starts].sort((a, b) => a - b));
  });

  it("marks steps that ran inside the researcher subagent and indents them", () => {
    const steps = stepsFromSpans(deepAgent.spans);
    const inResearcher = steps.filter((s) => s.subagent === "researcher");
    expect(inResearcher.length).toBeGreaterThan(0);
    expect(inResearcher.every((s) => s.depth === 1)).toBe(true);
    const first = steps[0];
    expect(first.span.type).toBe("llm");
    expect(first.subagent).toBeNull();
    // the lead's first decision runs write_file and task
    expect(first.toolNames).toEqual(["write_file", "task"]);
  });

  it("sorts by cost without renumbering", () => {
    const sorted = sortStepsByCost(stepsFromSpans(deepAgent.spans));
    const spend = sorted.map((s) => s.span.litellm?.spend ?? 0);
    expect(spend).toEqual([...spend].sort((a, b) => b - a));
    expect(sorted[0].span.type).toBe("llm");
  });
});

describe("trace-level helpers", () => {
  it("detects errors and finds the earliest failing non-root span", () => {
    expect(traceHasErrors(swarm)).toBe(true);
    expect(traceHasErrors(research)).toBe(false);
    const failed = firstErrorSpan(swarm.spans);
    expect(failed?.status).toBe("error");
    expect(failed?.parent_span_id).not.toBeNull();
  });

  it("lays out one node per agent with an edge labelled by invocations", () => {
    const layout = layoutAgentGraph(swarm.agents);
    expect(layout.nodes).toHaveLength(4);
    const labels = layout.edges.map((e) => `${e.from.agent.name}->${e.to.agent.name} ${e.label}`);
    expect(labels).toEqual(
      expect.arrayContaining([
        "orchestrator->researcher ×12",
        "researcher->fact_checker ×2",
        "orchestrator->critic ×1",
      ]),
    );
    const researcherNode = layout.nodes.find((n) => n.agent.name === "researcher");
    const criticNode = layout.nodes.find((n) => n.agent.name === "critic");
    expect(researcherNode?.width).toBeGreaterThan(criticNode?.width ?? Infinity);
    expect(researcherNode?.x).toBeGreaterThan(layout.nodes[0].x);
  });

  it("parses llm message payloads and rejects non-message JSON", () => {
    expect(parseMessages('[{"role":"user","content":"hi"}]')).toEqual([{ role: "user", content: "hi" }]);
    expect(
      parseMessages('{"role":"assistant","content":"","tool_calls":[{"name":"t","args":{}}]}')?.[0].tool_calls,
    ).toHaveLength(1);
    expect(parseMessages('{"file_path":"/tmp/x"}')).toBeNull();
    expect(parseMessages("not json")).toBeNull();
  });

  it("interleaves two newest-first lists by time", () => {
    const merged = interleaveByTime(
      [
        { at: 30, item: "t1" },
        { at: 10, item: "t2" },
      ],
      [{ at: 20, item: "r1" }],
    );
    expect(merged.map((m) => m.item)).toEqual(["t1", "r1", "t2"]);
  });
});
