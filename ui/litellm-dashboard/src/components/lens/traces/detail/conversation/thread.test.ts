import { describe, expect, it } from "vitest";
import research from "../../__fixtures__/research_trace.json";
import type { Span, TraceMessage } from "../../types";
import type { ConversationGroup, ConversationItem } from "./conversation";
import { buildThread, replyErrorSpanIds, threadDurationMs, withoutErrorsOf } from "./thread";

const base = research.spans[0] as Span;
const span = (id: string, type: Span["type"], start: number, timing: Partial<Span> = {}): Span => ({
  ...base,
  span_id: id,
  type,
  start_offset_ms: start,
  duration_ms: 100,
  status: "ok",
  ...timing,
});
const item = (id: string, type: Span["type"], start: number, messages: TraceMessage[]): ConversationGroup => ({
  kind: "item",
  item: { id, span: span(id, type, start), messages } as ConversationItem,
});
const tool = (id: string, start: number, status: Span["status"] = "ok"): ConversationGroup => ({
  kind: "item",
  item: { id, span: span(id, "tool", start, { duration_ms: 50, status }), messages: [], toolResult: "ok" },
});

const ask = (content: string): TraceMessage => ({ role: "user", content });
const say = (content: string): TraceMessage => ({ role: "assistant", content });
const plan = (content: string): TraceMessage => ({
  role: "assistant",
  content,
  tool_calls: [{ name: "search", args: {} }],
});

describe("buildThread", () => {
  it("folds the calls between a prompt and its final answer into one turn", () => {
    const turns = buildThread([
      item("root", "agent", 0, [ask("fix the bug")]),
      item("llm1", "llm", 10, [plan("looking")]),
      tool("t1", 200),
      item("llm2", "llm", 300, [say("Fixed it")]),
    ]);
    expect(turns).toHaveLength(1);
    expect(turns[0].prompt.map((m) => m.content)).toEqual(["fix the bug"]);
    expect(turns[0].reply?.content).toBe("Fixed it");
    expect(turns[0].work.map((w) => (w.kind === "step" ? w.item.id : w.id))).toEqual(["llm1", "t1"]);
    expect(turns[0].llmCalls).toBe(1);
    expect(turns[0].toolCalls).toBe(1);
    expect(threadDurationMs(turns[0])).toBe(400);
  });

  it("starts a new turn at every user prompt", () => {
    const turns = buildThread([
      item("u1", "agent", 0, [ask("first")]),
      item("a1", "llm", 10, [say("one")]),
      item("u2", "agent", 500, [ask("second")]),
      item("a2", "llm", 510, [say("two")]),
    ]);
    expect(turns.map((t) => [t.prompt[0]?.content, t.reply?.content])).toEqual([
      ["first", "one"],
      ["second", "two"],
    ]);
  });

  it("keeps a reply's tool calls out of the reply and in the work", () => {
    const turns = buildThread([
      item("u", "agent", 0, [ask("go")]),
      item("llm", "llm", 10, [plan("calling tools"), say("done")]),
    ]);
    expect(turns[0].reply?.content).toBe("done");
    const leftover = turns[0].work.at(-1);
    expect(leftover?.kind === "step" && leftover.item.messages.map((m) => m.content)).toEqual(["calling tools"]);
  });

  it("has no reply when the agent only made tool calls", () => {
    const turns = buildThread([item("u", "agent", 0, [ask("go")]), item("llm", "llm", 10, [plan("x")]), tool("t", 20)]);
    expect(turns[0].reply).toBeNull();
    expect(turns[0].work).toHaveLength(2);
  });

  it("separates system context from the prompt", () => {
    const turns = buildThread([item("u", "agent", 0, [{ role: "system", content: "You are helpful" }, ask("hi")])]);
    expect(turns[0].context.map((m) => m.content)).toEqual(["You are helpful"]);
    expect(turns[0].prompt.map((m) => m.content)).toEqual(["hi"]);
  });

  it("counts subagent work and marks the turn failed when a nested step failed", () => {
    const turns = buildThread([
      item("u", "agent", 0, [ask("research")]),
      {
        kind: "branch",
        id: "sub",
        name: "Explore",
        children: [item("s1", "llm", 50, [plan("searching")]), tool("s2", 80, "error")],
      },
      item("a", "llm", 900, [say("summary")]),
    ]);
    expect(turns[0].work[0]).toMatchObject({ kind: "subagent", name: "Explore" });
    expect(turns[0].llmCalls).toBe(1);
    expect(turns[0].toolCalls).toBe(1);
    expect(turns[0].failed).toBe(true);
  });

  it("keeps work before the first prompt as its own turn", () => {
    const turns = buildThread([
      tool("warmup", 0),
      item("u", "agent", 10, [ask("hi")]),
      item("a", "llm", 20, [say("hey")]),
    ]);
    expect(turns).toHaveLength(2);
    expect(turns[0].prompt).toEqual([]);
    expect(turns[1].reply?.content).toBe("hey");
  });

  it("treats an LLM call that re-sends the same prompt as part of the turn, not a new one", () => {
    const turns = buildThread([
      item("root", "agent", 0, [ask("What is an agent trace?")]),
      item("llm1", "llm", 10, [{ role: "system", content: "Be brief" }, ask("What is an agent trace?"), plan("look")]),
      tool("t1", 100),
      item("llm2", "llm", 200, [say("A recording of one agent run")]),
    ]);
    expect(turns).toHaveLength(1);
    expect(turns[0].prompt.map((m) => m.content)).toEqual(["What is an agent trace?"]);
    expect(turns[0].context.map((m) => m.content)).toEqual(["Be brief"]);
    expect(turns[0].llmCalls).toBe(1);
    expect(turns[0].reply?.content).toBe("A recording of one agent run");
  });

  it("starts a new turn when the user asks the same thing again after an answer", () => {
    const turns = buildThread([
      item("u1", "agent", 0, [ask("again?")]),
      item("a1", "llm", 10, [say("yes")]),
      item("u2", "agent", 100, [ask("again?")]),
      item("a2", "llm", 110, [say("still yes")]),
    ]);
    expect(turns.map((t) => t.reply?.content)).toEqual(["yes", "still yes"]);
  });

  it("keeps a failure that happened after the reply in the work and marks the turn failed", () => {
    const turns = buildThread([
      item("u", "agent", 0, [ask("deploy")]),
      item("llm", "llm", 10, [say("Deployed")]),
      tool("cleanup", 200, "error"),
    ]);
    expect(turns[0].reply?.content).toBe("Deployed");
    expect(turns[0].work.map((w) => (w.kind === "step" ? w.item.id : w.id))).toEqual(["cleanup"]);
    expect(turns[0].failed).toBe(true);
  });

  it("uses a subagent's answer as the reply when the root only forwarded it", () => {
    const turns = buildThread([
      item("u", "agent", 0, [ask("research")]),
      {
        kind: "branch",
        id: "sub",
        name: "Explore",
        children: [item("s1", "llm", 50, [plan("searching")]), item("s2", "llm", 90, [say("Found the answer")])],
      },
    ]);
    expect(turns[0].reply?.content).toBe("Found the answer");
    expect(turns[0].replyItem?.id).toBe("s2");
    expect(turns[0].work[0]).toMatchObject({ kind: "subagent", name: "Explore" });
  });

  it("picks the subagent answer that finished last when branches run in parallel", () => {
    const branch = (id: string, start: number, answer: string): ConversationGroup => ({
      kind: "branch",
      id,
      name: id,
      children: [item(`${id}-a`, "llm", start, [say(answer)])],
    });
    const turns = buildThread([
      item("u", "agent", 0, [ask("compare")]),
      branch("slow", 500, "Slow finished last"),
      branch("fast", 100, "Fast finished first"),
    ]);
    expect(turns[0].reply?.content).toBe("Slow finished last");
  });

  it("drops a run-level error from the Worked bar so it shows once at the top", () => {
    const failed: ConversationGroup = {
      kind: "item",
      item: { id: "root-out", span: span("root", "agent", 0, { status: "error" }), messages: [], showError: true },
    };
    const [turn] = buildThread([item("u", "agent", 0, [ask("go")]), tool("t", 10), failed]);
    expect(turn.work.map((w) => (w.kind === "step" ? w.item.id : w.id))).toEqual(["t", "root-out"]);
    const cleaned = withoutErrorsOf(turn, new Set(["root"]));
    expect(cleaned.work.map((w) => (w.kind === "step" ? w.item.id : w.id))).toEqual(["t"]);
    expect(replyErrorSpanIds([turn]).size).toBe(0);
  });

  it("returns no turns for an empty conversation", () => {
    expect(buildThread([])).toEqual([]);
  });
});
