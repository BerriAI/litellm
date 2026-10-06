import { describe, expect, it } from "vitest";
import {
  buildConversation,
  conversationSteps,
  newConversationMessages,
  groupConversation,
  conversationWarnings,
  pendingConversationBranches,
} from "./conversation";
import type { Span, SpanDetail, TraceMessage } from "../../types";
import research from "../../__fixtures__/research_trace.json";

const root = { ...research.spans[0], span_id: "root", parent_span_id: null, type: "agent" } as Span;
const user: TraceMessage = { role: "user", content: "Find my order" };
const call: TraceMessage = {
  role: "assistant",
  content: "Checking the order",
  tool_calls: [{ name: "lookup", args: { order: 42 } }],
};
const result: TraceMessage = { role: "tool", content: "Shipped" };
const answer: TraceMessage = { role: "assistant", content: "Your order shipped" };
const detail = (span_id: string, input: unknown, output: unknown): SpanDetail => ({
  span_id,
  input: JSON.stringify(input),
  output: JSON.stringify(output),
  attributes: {},
});

describe("trace conversation", () => {
  it("keeps missing nested details pending through framework and native agent ancestors", () => {
    const agent = {
      ...root,
      span_id: "agent",
      parent_span_id: "root",
      type: "tool",
      name: "Agent",
      framework: "claude-code",
    };
    const framework = { ...root, span_id: "framework", parent_span_id: "agent", type: "framework" };
    const child = { ...root, span_id: "child", parent_span_id: "framework" };
    const other = { ...root, span_id: "other", parent_span_id: "root" };
    const spans = [root, agent, framework, child, other] as Span[];
    const details = new Map(
      spans.filter((span) => span !== child).map((span) => [span.span_id, detail(span.span_id, [], [])]),
    );
    expect([...pendingConversationBranches(spans, details, false)].toSorted()).toEqual([
      "agent",
      "child",
      "framework",
      "root",
    ]);
    const completeDetails = new Map([...details, [child.span_id, detail(child.span_id, [], [])]]);
    expect(pendingConversationBranches(spans, completeDetails, false).size).toBe(0);
  });

  it.each(["child", "missing-parent"])("stops pending ancestry at cycles or missing parents (%s)", (parentId) => {
    const agent = { ...root, span_id: "agent", parent_span_id: parentId };
    const child = { ...root, span_id: "child", parent_span_id: "agent" };
    const spans = [root, agent, child];
    const details = new Map([
      [root.span_id, detail(root.span_id, [], [])],
      [agent.span_id, detail(agent.span_id, [], [])],
    ]);
    expect([...pendingConversationBranches(spans, details, false)].toSorted()).toEqual(["agent", "child"]);
  });

  it.each([false, true])("handles 20,000-level pending ancestry without overflowing (cycle: %s)", (cycle) => {
    const rootParent = cycle ? "deep-19999" : null;
    const chain = Array.from({ length: 20_000 }, (_, index) => ({
      ...root,
      span_id: `deep-${index}`,
      parent_span_id: index === 0 ? rootParent : `deep-${index - 1}`,
    }));
    const spans = [...chain, root];
    const details = new Map(
      spans.filter((span) => span.span_id !== "deep-19999").map((span) => [span.span_id, detail(span.span_id, [], [])]),
    );
    const pending = pendingConversationBranches(spans, details, false);
    expect(pending.size).toBe(chain.length);
    expect(chain.every((span) => pending.has(span.span_id))).toBe(true);
    expect(pending.has(root.span_id)).toBe(false);
  });

  it.each([
    { boundary: 10, morePages: true, pending: true },
    { boundary: 20, morePages: true, pending: true },
    { boundary: 21, morePages: true, pending: false },
    { boundary: 10, morePages: false, pending: false },
  ])("only waits for pages that could contain later branch children (%j)", ({ boundary, morePages, pending }) => {
    const agent = { ...root, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 5 };
    const child = { ...agent, span_id: "child", parent_span_id: "agent", start_offset_ms: 2, duration_ms: 18 };
    const other = { ...root, span_id: "other", parent_span_id: "root", start_offset_ms: boundary };
    const spans = [root, agent, child, other];
    const details = new Map(spans.map((span) => [span.span_id, detail(span.span_id, [], [])]));
    expect(pendingConversationBranches(spans, details, morePages).has("agent")).toBe(pending);
  });

  it.each(["reviewer", "__proto__", "constructor"])("keeps repeated agent name %s stable as steps load", (name) => {
    const first = { ...root, span_id: "first", name, start_offset_ms: 1 };
    const second = { ...first, span_id: "second", start_offset_ms: 2 };
    const spans = [second, first];
    const firstDetails = new Map([["first", detail("first", "Review code", "")]]);
    const partial = buildConversation(spans, firstDetails, false);
    const complete = buildConversation(
      spans,
      new Map([...firstDetails, ["second", detail("second", "Review tests", "")]]),
      true,
    );
    expect(partial.map(({ agentId, agentName }) => ({ agentId, agentName }))).toEqual([
      { agentId: "first", agentName: `${name} (1)` },
    ]);
    expect(complete.map(({ agentId, agentName }) => ({ agentId, agentName }))).toEqual([
      { agentId: "first", agentName: `${name} (1)` },
      { agentId: "second", agentName: `${name} (2)` },
    ]);
  });

  it("breaks equal-time agent label ties by span ID without changing event order", () => {
    const first = { ...root, span_id: "first", name: "reviewer", start_offset_ms: 1 };
    const second = { ...first, span_id: "second" };
    const details = new Map([
      ["first", detail("first", "Review code", "")],
      ["second", detail("second", "Review tests", "")],
    ]);
    const items = buildConversation([second, first], details, true);
    expect(items.map(({ agentId, agentName }) => ({ agentId, agentName }))).toEqual([
      { agentId: "second", agentName: "reviewer (2)" },
      { agentId: "first", agentName: "reviewer (1)" },
    ]);
  });

  it("removes repeated prefixes and trimmed context, but preserves a genuinely repeated question", () => {
    expect(newConversationMessages([user, call, result], [user, call, result, answer])).toEqual([answer]);
    expect(newConversationMessages([user, call, result], [call, result, answer])).toEqual([answer]);
    expect(newConversationMessages([user, answer], [user, answer, user])).toEqual([user]);
    expect(newConversationMessages([user, call, result], [user])).toEqual([]);
  });

  it("reads a tool exchange once, without repeating the root's final answer", () => {
    const first = { ...root, span_id: "first", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const tool = { ...first, span_id: "tool", name: "lookup", type: "tool", start_offset_ms: 2 } as Span;
    const last = { ...first, span_id: "last", start_offset_ms: 3 };
    const details = new Map([
      ["root", detail("root", [user], [answer])],
      ["first", detail("first", [user], [call])],
      ["tool", { ...detail("tool", { order: 42 }, "Shipped"), output: "Shipped" }],
      ["last", detail("last", [user, call, result], [answer])],
    ]);
    const items = buildConversation([root, first, tool, last], details, true);
    expect(items.flatMap((item) => item.messages)).toEqual([user, { ...call, tool_calls: [] }, answer]);
    expect(items.find((item) => item.span.span_id === "tool")).toMatchObject({
      toolCall: { name: "lookup", args: { order: 42 } },
      toolResult: "Shipped",
    });
  });

  it("uses normalized message content and structured tool arguments from the gateway", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm" } as Span;
    const normalized: SpanDetail = {
      span_id: "model",
      input: "unparsed input",
      output: "unparsed output",
      attributes: {},
      input_ui: { kind: "messages", messages: [{ role: "user", content: user.content }] },
      output_ui: {
        kind: "messages",
        messages: [
          { role: "assistant", content: "Checking", tool_calls: [{ name: "lookup", arguments: '{"order":42}' }] },
        ],
      },
    };
    const details = new Map([
      ["root", detail("root", [], [])],
      ["model", normalized],
    ]);
    const items = buildConversation([root, model], details, true);
    expect(items[0].messages).toEqual([
      user,
      { role: "assistant", content: "Checking", tool_calls: [{ name: "lookup", args: { order: 42 } }] },
    ]);
  });

  it("keeps parallel agent histories separate and excludes framework scaffolding", () => {
    const agent = { ...root, span_id: "agent", parent_span_id: "root" };
    const otherAgent = { ...agent, span_id: "other" };
    const first = { ...root, span_id: "first", parent_span_id: "agent", type: "llm", start_offset_ms: 1 } as Span;
    const second = { ...first, span_id: "second", parent_span_id: "other", start_offset_ms: 2 };
    const framework = { ...first, span_id: "framework", type: "framework" } as Span;
    const spans = [root, agent, otherAgent, first, second, framework];
    expect(conversationSteps(spans).map((span) => span.span_id)).toEqual(["root", "agent", "other", "first", "second"]);
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [], [])],
      ["other", detail("other", [], [])],
      ["first", detail("first", [user], [answer])],
      ["second", detail("second", [user], [answer])],
    ]);
    expect(buildConversation(spans, details, true).flatMap((item) => item.messages)).toEqual([
      user,
      answer,
      user,
      answer,
    ]);
  });

  it("defers the root's final answer until all sections have been loaded", () => {
    const details = new Map([["root", detail("root", [user], [answer])]]);
    expect(buildConversation([root], details, false).flatMap((item) => item.messages)).toEqual([user]);
    expect(buildConversation([root], details, true).flatMap((item) => item.messages)).toEqual([user, answer]);
  });

  it("pairs typed tool arguments without coercing strings that resemble JSON", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const tool = { ...model, span_id: "tool", name: "lookup", type: "tool", start_offset_ms: 2 } as Span;
    const args = { order: 42, active: true, filters: { tags: ["paid"] }, empty: null, label: "42", text: "true" };
    const typedCall = { ...call, tool_calls: [{ name: "lookup", args }] };
    const toolDetail: SpanDetail = {
      ...detail("tool", args, "Shipped"),
      input_ui: {
        kind: "fields",
        fields: Object.entries(args).map(([key, value]) => ({
          key,
          value: typeof value === "string" ? value : JSON.stringify(value),
        })),
      },
    };
    const details = new Map([
      ["root", detail("root", [user], [])],
      ["model", detail("model", [user], [typedCall])],
      ["tool", toolDetail],
    ]);
    const items = buildConversation([root, model, tool], details, true);
    expect(items.flatMap((item) => item.messages.flatMap((message) => message.tool_calls ?? []))).toEqual([]);
    expect(items.find((item) => item.toolCall)?.toolCall?.args).toEqual(args);

    const differentType = { ...typedCall, tool_calls: [{ name: "lookup", args: { ...args, order: "42" } }] };
    details.set("model", detail("model", [user], [differentType]));
    const unmatched = buildConversation([root, model, tool], details, true);
    expect(unmatched.flatMap((item) => item.messages.flatMap((message) => message.tool_calls ?? []))).toEqual(
      differentType.tool_calls,
    );
  });

  it("loads child agents and places their own answers after their tools", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const tool = {
      ...agent,
      span_id: "tool",
      parent_span_id: "agent",
      type: "tool",
      start_offset_ms: 2,
      duration_ms: 2,
    } as Span;
    const next = { ...tool, span_id: "next", parent_span_id: "root", type: "llm", start_offset_ms: 30 } as Span;
    const final = { role: "assistant", content: "Finished the whole run" };
    const details = new Map([
      ["root", detail("root", [], [final])],
      ["agent", detail("agent", [user], [answer])],
      ["tool", detail("tool", { order: 42 }, "Shipped")],
      ["next", detail("next", [], [final])],
    ]);
    const spans = [parent, agent, tool, next];
    expect(conversationSteps(spans)).toContain(agent);
    const items = buildConversation(spans, details, true);
    expect(items.map((item) => item.id)).toEqual(["agent", "tool", "agent-output", "next"]);
    expect(items.flatMap((item) => item.messages)).toEqual([user, answer, final]);

    details.delete("next");
    expect(buildConversation(spans, details, false).flatMap((item) => item.messages)).toEqual([user, answer]);
    details.delete("tool");
    expect(buildConversation(spans, details, false).flatMap((item) => item.messages)).toEqual([user]);
  });

  it("does not repeat a child agent answer already recorded by its model", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const model = {
      ...agent,
      span_id: "model",
      parent_span_id: "agent",
      type: "llm",
      start_offset_ms: 2,
      duration_ms: 2,
    } as Span;
    const spans = [parent, agent, model];
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [user], [answer])],
      ["model", detail("model", [user], [answer])],
    ]);
    expect(buildConversation(spans, details, true).flatMap((item) => item.messages)).toEqual([user, answer]);
  });

  it("stops at a missing step even if later details and completion are supplied", () => {
    const first = { ...root, span_id: "first", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const missing = { ...first, span_id: "missing", start_offset_ms: 2 };
    const last = { ...first, span_id: "last", start_offset_ms: 3 };
    const details = new Map([
      ["root", detail("root", [user], [answer])],
      ["first", detail("first", [user], [call])],
      ["last", detail("last", [user, call, result], [answer])],
    ]);
    expect(buildConversation([root, first, missing, last], details, true).flatMap((item) => item.messages)).toEqual([
      user,
      call,
    ]);
  });

  it("keeps nested and parallel agent answers in their own histories", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const nested = { ...agent, span_id: "nested", parent_span_id: "agent", start_offset_ms: 2, duration_ms: 5 };
    const other = { ...agent, span_id: "other", start_offset_ms: 4, duration_ms: 30 };
    const tool = {
      ...nested,
      span_id: "tool",
      parent_span_id: "nested",
      type: "tool",
      start_offset_ms: 3,
      duration_ms: 1,
    } as Span;
    const otherTool = { ...tool, span_id: "other-tool", parent_span_id: "other", start_offset_ms: 5 };
    const summary = { role: "assistant", content: "Nested work complete" };
    const details = new Map([
      ["root", detail("root", [], [])],
      ["agent", detail("agent", [], [summary])],
      ["nested", detail("nested", [], [answer])],
      ["other", detail("other", [], [answer])],
      ["tool", detail("tool", {}, "First result")],
      ["other-tool", detail("other-tool", {}, "Second result")],
    ]);
    const items = buildConversation([parent, agent, nested, tool, other, otherTool], details, true);
    expect(items.map((item) => item.id)).toEqual([
      "tool",
      "other-tool",
      "nested-output",
      "agent-output",
      "other-output",
    ]);
    expect(items.flatMap((item) => item.messages)).toEqual([answer, summary, answer]);
  });

  it("renders a tool-only trace as one exchange", () => {
    const tool = { ...root, type: "tool" } as Span;
    const items = buildConversation([tool], new Map([["root", detail("root", { order: 42 }, "Shipped")]]), true);
    expect(items).toHaveLength(1);
    expect(items[0].toolResult).toBe("Shipped");
    expect(items[0].messages).toEqual([]);
  });

  it("removes forwarded child answers from ancestors while preserving identical answers in sibling branches", () => {
    const parent = { ...root, start_offset_ms: 0, duration_ms: 100 };
    const agent = { ...parent, span_id: "agent", parent_span_id: "root", start_offset_ms: 1, duration_ms: 20 };
    const nested = { ...agent, span_id: "nested", parent_span_id: "agent", start_offset_ms: 2, duration_ms: 5 };
    const other = { ...agent, span_id: "other", start_offset_ms: 4, duration_ms: 30 };
    const spans = [parent, agent, nested, other];
    const details = new Map(spans.map((span) => [span.span_id, detail(span.span_id, [], [answer])]));
    const items = buildConversation(spans, details, true);
    expect(items.map((item) => item.id)).toEqual(["nested-output", "other-output"]);
    expect(items.flatMap((item) => item.messages)).toEqual([answer, answer]);
  });
});

describe("recorded tool summaries", () => {
  it("pairs repeated named calls within their own branch and preserves unmatched calls and custom text", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const first = { ...model, span_id: "first", name: "terminal", type: "tool", start_offset_ms: 2 } as Span;
    const second = { ...first, span_id: "second", start_offset_ms: 3 };
    const child = { ...root, span_id: "child", parent_span_id: "root", start_offset_ms: 4 };
    const childTool = { ...first, span_id: "child-tool", parent_span_id: "child", start_offset_ms: 5 };
    const saved = "SAVED TASK RESUMED: Continue the unfinished task exactly as recorded";
    const summary = JSON.stringify([{ content: "Checking", tool_names: ["terminal", "terminal", "read_file"] }]);
    const details = new Map([
      ["root", detail("root", [{ role: "user", content: saved }], [])],
      [
        "model",
        { ...detail("model", [], []), output: summary, output_ui: { kind: "text", text: summary } } as SpanDetail,
      ],
      ["first", detail("first", { command: "pwd" }, { output: "/workspace", exit_code: 0 })],
      ["second", detail("second", { command: "pwd" }, { output: "/workspace", exit_code: 0 })],
      ["child", detail("child", [{ role: "user", content: saved }], [])],
      ["child-tool", detail("child-tool", { path: "README.md" }, "content")],
    ]);
    const items = buildConversation([root, model, first, second, child, childTool], details, true);
    expect(items.flatMap((item) => item.messages).filter((message) => message.content === saved)).toHaveLength(2);
    expect(items.flatMap((item) => item.messages.flatMap((message) => message.tool_calls ?? []))).toEqual([
      { name: "read_file", args: undefined },
    ]);
    expect(items.filter((item) => item.toolCall).map((item) => item.id)).toEqual(["first", "second", "child-tool"]);
    expect(JSON.parse(items.find((item) => item.id === "first")!.toolResult!)).toEqual({
      output: "/workspace",
      exit_code: 0,
    });
  });

  it("deduplicates native OpenAI function calls without requiring message text", () => {
    const model = { ...root, span_id: "model", parent_span_id: "root", type: "llm", start_offset_ms: 1 } as Span;
    const tool = { ...model, span_id: "tool", name: "read_file", type: "tool", start_offset_ms: 2 } as Span;
    const output = [
      {
        role: "assistant",
        content: null,
        tool_calls: [{ type: "function", function: { name: "read_file", arguments: '{"path":"README.md"}' } }],
      },
    ];
    const details = new Map([
      ["root", detail("root", [], [])],
      ["model", detail("model", [], output)],
      ["tool", detail("tool", { path: "README.md" }, "file content")],
    ]);
    expect(
      buildConversation([root, model, tool], details, true).filter((item) => item.toolCall || item.messages.length),
    ).toHaveLength(1);
  });
});

describe("coding sessions", () => {
  it("preserves identical user messages and assistant replies in transcript events", () => {
    const spans = [
      root,
      ...[1, 2, 3, 4].map((n) => ({
        ...root,
        span_id: `m${n}`,
        parent_span_id: root.span_id,
        type: "chain" as const,
        start_offset_ms: n,
      })),
    ];
    const details = new Map([
      [root.span_id, { ...detail(root.span_id, [user], []), attributes: { "lens.capture.messages_separate": "true" } }],
      ...spans.slice(1).map(
        (span, i) =>
          [
            span.span_id,
            {
              ...detail(span.span_id, i % 2 === 0 ? [user] : [], i % 2 === 1 ? [answer] : []),
              attributes: { "lens.capture.source": "session_transcript" },
            },
          ] as const,
      ),
    ]);
    expect(buildConversation(spans, details, true).flatMap((item) => item.messages)).toEqual([
      user,
      answer,
      user,
      answer,
    ]);
  });

  it("uses one actor across resumed turns and keeps child branches together", () => {
    const resumed = { ...root, span_id: "resumed", start_offset_ms: 10 };
    const child = { ...root, span_id: "child", parent_span_id: root.span_id, start_offset_ms: 2, name: "reader" };
    const nested = { ...child, span_id: "nested", parent_span_id: "child", start_offset_ms: 3, name: "checker" };
    const details = new Map(
      [root, resumed, child, nested].map((span) => [
        span.span_id,
        {
          ...detail(span.span_id, span.name, "Done"),
          attributes: { "gen_ai.agent.id": span === root || span === resumed ? "main-session" : span.span_id },
        },
      ]),
    );
    const items = buildConversation([root, resumed, child, nested], details, true);
    expect(new Set(items.filter((item) => !item.parentBranchId).map((item) => item.agentId))).toEqual(
      new Set(["main-session"]),
    );
    const groups = groupConversation(items, [root, resumed, child, nested]);
    const branch = groups.find((group) => group.kind === "branch");
    expect(branch).toMatchObject({ kind: "branch", id: "child", name: "reader" });
    if (branch?.kind === "branch")
      expect(branch.children.some((group) => group.kind === "branch" && group.id === "nested")).toBe(true);
  });

  it("shows child conversations when their parent has no recorded messages", () => {
    const child = { ...root, span_id: "child", parent_span_id: root.span_id, start_offset_ms: 1, name: "reader" };
    const details = new Map([
      [root.span_id, detail(root.span_id, [], [])],
      [child.span_id, detail(child.span_id, "Read the file", "File contents")],
    ]);
    const groups = groupConversation(buildConversation([root, child], details, true), [root, child]);
    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({ kind: "branch", id: "child", name: "reader" });
    if (groups[0].kind === "branch") {
      expect(groups[0].children.flatMap((group) => (group.kind === "item" ? group.item.messages : []))).toEqual([
        { role: "user", content: "Read the file" },
        { role: "assistant", content: "File contents" },
      ]);
    }
  });

  it("retains silent intermediate agents in nested branches", () => {
    const child = { ...root, span_id: "child", parent_span_id: root.span_id, start_offset_ms: 1, name: "reader" };
    const nested = { ...child, span_id: "nested", parent_span_id: child.span_id, start_offset_ms: 2, name: "checker" };
    const spans = [root, child, nested];
    const details = new Map([
      [root.span_id, detail(root.span_id, [user], [])],
      [child.span_id, detail(child.span_id, [], [])],
      [nested.span_id, detail(nested.span_id, "Check the result", "Checked")],
    ]);
    const expectedBranch = {
      kind: "branch",
      id: child.span_id,
      name: "reader",
      children: [expect.objectContaining({ kind: "branch", id: nested.span_id, name: "checker" })],
    };
    expect(groupConversation(buildConversation(spans, details, true), spans)).toEqual([
      expect.objectContaining({ kind: "item" }),
      expect.objectContaining(expectedBranch),
    ]);
  });

  it("folds native Claude Agent descendants and omits auxiliary suggestions", () => {
    const agent = {
      ...root,
      span_id: "agent-tool",
      parent_span_id: root.span_id,
      type: "tool" as const,
      name: "Agent",
      framework: "claude-code",
      start_offset_ms: 1,
    };
    const execution = {
      ...agent,
      span_id: "execution",
      parent_span_id: agent.span_id,
      type: "framework" as const,
      name: "claude_code.tool.execution",
    };
    const response = {
      ...root,
      span_id: "response",
      parent_span_id: execution.span_id,
      type: "chain" as const,
      start_offset_ms: 3,
    };
    const suggestion = { ...response, span_id: "suggestion", parent_span_id: root.span_id, type: "llm" as const };
    const details = new Map([
      [root.span_id, { ...detail(root.span_id, [user], []), attributes: { "gen_ai.agent.id": "session" } }],
      [
        agent.span_id,
        {
          ...detail(agent.span_id, { prompt: "Read", description: "Reader" }, "Started"),
          attributes: { "gen_ai.agent.id": "session" },
        },
      ],
      [
        response.span_id,
        { ...detail(response.span_id, [], "Child reply"), attributes: { "event.name": "assistant_response" } },
      ],
      [
        suggestion.span_id,
        {
          ...detail(suggestion.span_id, [], "HIDDEN SUGGESTION"),
          attributes: { query_source_safe: "prompt_suggestion" },
        },
      ],
    ]);
    const items = buildConversation([root, agent, execution, response, suggestion], details, true);
    expect(items.flatMap((item) => item.messages).map((message) => message.content)).not.toContain("HIDDEN SUGGESTION");
    expect(groupConversation(items, [root, agent, execution, response, suggestion])).toContainEqual(
      expect.objectContaining({ kind: "branch", id: "agent-tool", name: "Reader" }),
    );
    expect(items.find((item) => item.span.span_id === response.span_id)?.agentId).toBe("agent-tool");
  });

  it("fills native tool arguments and missing results from logs without duplicating calls or replacing recorded output", () => {
    const tool = {
      ...root,
      span_id: "tool",
      parent_span_id: root.span_id,
      type: "tool" as const,
      name: "Bash",
      start_offset_ms: 1,
    };
    const log = {
      ...tool,
      span_id: "log",
      type: "framework" as const,
      framework: "claude-code",
      name: "claude_code.tool_result",
      start_offset_ms: 2,
    };
    const body = { ...log, span_id: "body", name: "claude_code.api_request_body", start_offset_ms: 3 };
    const toolDetail = {
      ...detail(tool.span_id, { command: "exit 3" }, ""),
      output: "",
      attributes: { "gen_ai.tool.call.id": "call-1" },
    };
    const details = new Map([
      [root.span_id, detail(root.span_id, [user], [])],
      [tool.span_id, toolDetail],
      [
        log.span_id,
        {
          ...detail(log.span_id, { command: "exit 3", description: "Expected failure" }, ""),
          attributes: { tool_use_id: "call-1" },
        },
      ],
      [
        body.span_id,
        detail(body.span_id, "", {
          tool_results: [
            { id: "call-1", content: "Expected stdout" },
            { id: "unrelated", content: "Other stdout" },
          ],
        }),
      ],
    ]);
    const spans = [root, tool, log, body];
    const items = buildConversation(spans, details, true);
    expect(items.filter((item) => item.toolCall)).toHaveLength(1);
    expect(items.find((item) => item.toolCall)).toMatchObject({
      toolCall: { args: { command: "exit 3", description: "Expected failure" } },
      toolResult: "Expected stdout",
    });
    details.set(tool.span_id, { ...toolDetail, output: "Recorded output" });
    expect(buildConversation(spans, details, true).find((item) => item.toolCall)?.toolResult).toBe("Recorded output");
    expect(
      conversationWarnings(
        new Map([
          [
            body.span_id,
            {
              ...detail(body.span_id, "", { warning: "Export truncated" }),
              attributes: { "event.name": "api_request_body" },
            },
          ],
        ]),
        true,
      ),
    ).toEqual(["Export truncated"]);
  });

  it.each([
    { recorded: "repl_main_thread", source: "repl_main_thread" },
    { recorded: "agent", source: "agent:builtin:general-purpose" },
    { recorded: "agent.builtin:general-purpose", source: "agent:builtin:general-purpose" },
    { recorded: "agent.custom:reader", source: "agent:custom:reader" },
  ])("positions $source commentary before tools with native source $recorded", ({ recorded, source }) => {
    const llm = {
      ...root,
      span_id: "llm",
      parent_span_id: root.span_id,
      type: "llm" as const,
      start_offset_ms: 1,
      duration_ms: 10,
      model: "model-a",
    };
    const tool = { ...llm, span_id: "tool", name: "Read", type: "tool" as const, start_offset_ms: 8, duration_ms: 1 };
    const response = { ...llm, span_id: "reply", type: "chain" as const, start_offset_ms: 11, duration_ms: 0 };
    const details = new Map([
      [root.span_id, detail(root.span_id, [], [])],
      [
        llm.span_id,
        {
          ...detail(llm.span_id, [], []),
          attributes: { query_source_safe: recorded, first_content_ms: "2" },
        },
      ],
      [tool.span_id, detail(tool.span_id, {}, "File contents")],
      [
        response.span_id,
        {
          ...detail(response.span_id, [], "Checking"),
          attributes: { "event.name": "assistant_response", query_source: source },
        },
      ],
    ]);
    expect(buildConversation([root, llm, tool, response], details, true)[0].time).toBe(3);
    expect(buildConversation([root, llm, tool, response], details, true).map((item) => item.span.span_id)).toEqual([
      "reply",
      "tool",
    ]);
  });

  it("warns about old incomplete native captures and clears the warning when reply logs arrive", () => {
    const details = new Map([
      ["llm", { ...detail("llm", [], []), output: "", attributes: { "span.type": "llm_request" } }],
    ]);
    expect(conversationWarnings(details, false)).toEqual([]);
    expect(conversationWarnings(details, true)).toHaveLength(1);
    details.set("reply", { ...detail("reply", [], "Hello"), attributes: { "event.name": "assistant_response" } });
    expect(conversationWarnings(details, true)).toEqual([]);
  });
});
