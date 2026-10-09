import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
import research from "../../__fixtures__/research_trace.json";
import { useOpenTraceRouting } from "../../routing";
import type { SpanDetail, Trace } from "../../types";
import { RunView } from "../run/RunView";

vi.mock("../../../../networking", () => ({
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test",
}));
import { agentTraceCall, agentTraceSpanCall } from "../../../../networking";

function RoutedRunView(props: Omit<ComponentProps<typeof RunView>, "selection">) {
  const { selection } = useOpenTraceRouting();
  return <RunView {...props} selection={selection} />;
}

const root = { ...research.spans[0], span_id: "root", parent_span_id: null, type: "agent", duration_ms: 900 };
const llm = { ...root, span_id: "llm", name: "chat", type: "llm", parent_span_id: "root", start_offset_ms: 10 };
const tool = { ...root, span_id: "tool", name: "read_file", type: "tool", parent_span_id: "root", start_offset_ms: 50 };
const trace = { ...research, spans: [root, llm, tool] } as Trace;
const details: Record<string, SpanDetail> = {
  root: {
    span_id: "root",
    input: '[{"role":"user","content":"Read the release notes"}]',
    output: '[{"role":"assistant","content":"The release is ready"}]',
    attributes: {},
  },
  llm: {
    span_id: "llm",
    input: '[{"role":"user","content":"Read the release notes"}]',
    output: JSON.stringify([
      { role: "assistant", content: "Checking the changelog", tool_calls: [{ name: "read_file", args: {} }] },
    ]),
    attributes: {},
  },
  tool: { span_id: "tool", input: '{"path":"CHANGELOG.md"}', output: "All checks passed", attributes: {} },
};

describe("TraceThread", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceCall).mockReset().mockResolvedValue(trace);
    vi.mocked(agentTraceSpanCall)
      .mockReset()
      .mockImplementation(async (_token, _trace, id) => details[id]);
  });

  it("folds the agent's work under the prompt and shows only the final reply until expanded", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });

    expect(await within(thread).findByText("Read the release notes")).toBeVisible();
    expect(await within(thread).findByText("The release is ready")).toBeVisible();
    const worked = within(thread).getByRole("button", { name: /^Worked/, expanded: false });
    expect(within(worked).getByLabelText("1 model calls")).toBeVisible();
    expect(within(worked).getByLabelText("1 tool calls")).toBeVisible();
    expect(within(thread).queryByText("Checking the changelog")).not.toBeInTheDocument();

    await user.click(worked);
    expect(within(thread).getByText("Checking the changelog")).toBeVisible();
    await user.click(within(thread).getByRole("button", { name: "Inspect step read_file" }));
    expect(screen.getByRole("tab", { name: "Steps", selected: true })).toBeVisible();
    expect(screen.getByRole("treeitem", { selected: true })).toHaveAttribute("data-row-id", "tool");
  });

  it("retries a step that failed to load and then shows the reply", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceSpanCall)
      .mockRejectedValueOnce(new Error("boom"))
      .mockImplementation(async (_token, _trace, id) => details[id]);
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    const alert = await within(thread).findByRole("alert");
    expect(alert).toHaveTextContent("Could not load some steps of this thread.");
    await user.click(within(alert).getByRole("button", { name: "Retry" }));
    expect(await within(thread).findByText("The release is ready")).toBeVisible();
    expect(within(thread).queryByRole("alert")).not.toBeInTheDocument();
  });

  it("offers only Steps and Thread, and an old conversation link opens on Steps", async () => {
    window.history.replaceState(null, "", "/?view=conversation");
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    const views = await screen.findByRole("tablist", { name: "Trace view" });
    expect(
      within(views)
        .getAllByRole("tab")
        .map((tab) => tab.textContent),
    ).toEqual(["Steps", "Thread"]);
    expect(within(views).getByRole("tab", { name: "Steps" })).toHaveAttribute("aria-selected", "true");
    window.history.replaceState(null, "", "/");
  });

  it.each([false, true])("refreshes the reply in place and keeps it if the refresh fails (%s)", async (failed) => {
    const user = userEvent.setup();
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    expect(await within(thread).findByText("The release is ready")).toBeVisible();
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (failed) throw new Error("content refresh failed");
      return id === "root"
        ? { ...details.root, output: '[{"role":"assistant","content":"Updated answer"}]' }
        : details[id];
    });
    await user.click(screen.getByRole("button", { name: "Refresh run" }));
    if (failed) {
      expect(await within(thread).findByRole("button", { name: "Retry" })).toBeVisible();
      expect(within(thread).getByText("The release is ready")).toBeVisible();
    } else {
      expect(await within(thread).findByText("Updated answer")).toBeVisible();
      expect(within(thread).queryByText("The release is ready")).not.toBeInTheDocument();
    }
  });

  it("keeps the whole thread on screen when a refresh of the run fails", async () => {
    const user = userEvent.setup();
    const first = { ...trace, spans: [root, llm], next_cursor: "next-page" } as Trace;
    const second = { ...trace, spans: [tool], next_cursor: null } as Trace;
    vi.mocked(agentTraceCall).mockImplementation(async (_token, _trace, _ref, cursor) => (cursor ? second : first));
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    expect(await within(thread).findByText("End of thread")).toBeVisible();
    expect(vi.mocked(agentTraceCall).mock.calls.map((call) => call[3])).toEqual([null, "next-page"]);
    vi.mocked(agentTraceCall).mockRejectedValue(new Error("refresh unavailable"));
    await user.click(screen.getByRole("button", { name: "Refresh run" }));
    expect(await screen.findByText(/Previously received steps are still shown/)).toBeVisible();
    expect(within(thread).getByText("Read the release notes")).toBeVisible();
    expect(within(thread).getByText("The release is ready")).toBeVisible();
  });

  it("shows a failed shell command's output inside the Worked bar and in the step details", async () => {
    const user = userEvent.setup();
    const command = "npm test -- checkout";
    const terminal = { ...tool, name: "terminal", status: "error", error: null };
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [root, llm, terminal] } as Trace);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
      id === "tool"
        ? {
            span_id: "tool",
            input: JSON.stringify({ command }),
            output: JSON.stringify({ output: "FAIL checkout.test.ts", exit_code: 1 }),
            input_ui: { kind: "fields", fields: [{ key: "command", value: command }] },
            output_ui: {
              kind: "fields",
              fields: [
                { key: "output", value: "FAIL checkout.test.ts" },
                { key: "exit_code", value: "1" },
              ],
            },
            attributes: {},
          }
        : details[id],
    );
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    const worked = await within(thread).findByRole("button", { name: /^Worked/ });
    expect(within(worked).getByLabelText("A step failed")).toBeVisible();
    await user.click(worked);
    expect(await within(thread).findByText(/FAIL checkout.test.ts/, { selector: "pre" })).toBeVisible();
    expect(within(thread).getByText("exit_code")).toBeVisible();
    await user.click(within(thread).getByRole("button", { name: "Inspect step terminal" }));
    const pane = screen.getByRole("complementary", { name: "Span details" });
    expect(await within(pane).findByText(command, { selector: "pre" })).toBeVisible();
  });

  it("shows an agent failure once even when the run never replied", async () => {
    const user = userEvent.setup();
    const failedRoot = { ...root, status: "error", error: "Agent exceeded its execution limit" };
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [failedRoot, llm, tool] } as Trace);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
      id === "root" ? { ...details.root, output: "" } : details[id],
    );
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    expect(await within(thread).findByText("End of thread")).toBeVisible();
    expect(within(thread).getAllByText("Agent exceeded its execution limit")).toHaveLength(1);
    await user.click(within(thread).getByRole("button", { name: /^Worked/ }));
    expect(within(thread).getAllByText("Agent exceeded its execution limit")).toHaveLength(1);
  });

  it("warns when a Claude Code trace recorded no assistant replies", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [root] } as Trace);
    vi.mocked(agentTraceSpanCall).mockResolvedValue({
      ...details.root,
      output: "",
      attributes: { "span.type": "llm_request" },
    });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    expect(await screen.findByText(/no recorded assistant replies/)).toBeVisible();
    expect(screen.getByText("Read the release notes")).toBeVisible();
  });

  it("nests a failed subagent's work inside the Worked bar and shows its error once", async () => {
    const user = userEvent.setup();
    const child = {
      ...root,
      span_id: "child",
      parent_span_id: "root",
      name: "Investigate release",
      start_offset_ms: 20,
      duration_ms: 10,
      status: "error",
      error: "Investigation timed out",
    };
    const childTool = { ...tool, span_id: "child-tool", parent_span_id: "child", start_offset_ms: 22, duration_ms: 2 };
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [root, llm, child, childTool] } as Trace);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "child") return { ...details.root, span_id: id, input: "Investigate failed checks", output: "" };
      if (id === "child-tool") return { ...details.tool, span_id: id };
      return details[id];
    });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    await user.click(await within(thread).findByRole("button", { name: /^Worked/ }));
    await user.click(within(thread).getByText("Subagent: Investigate release", { exact: true }));
    expect(within(thread).getByText("Investigate failed checks")).toBeVisible();
    expect(within(thread).getAllByText("Investigation timed out")).toHaveLength(1);
  });
});

interface CompletionRequest {
  url: string;
  body: { model: string; messages: { role: string; content: string }[] };
}

const submitted = (turnId: string) => ({
  title: "Release check",
  turns: [
    {
      turn_id: turnId,
      user: "Read the release notes, please",
      summary: "Read the changelog before answering",
      steps: [{ span_id: "tool", label: "Read CHANGELOG.md", status: "ok" }],
      reply: "",
    },
  ],
});

function completion(args: string) {
  const call = { id: "call_1", type: "function", function: { name: "submit_thread", arguments: args } };
  const message = { role: "assistant", content: null, tool_calls: [call] };
  return {
    id: "chatcmpl-1",
    object: "chat.completion",
    created: 0,
    model: "fireworks_ai/deepseek-v4-pro",
    choices: [{ index: 0, finish_reason: "tool_calls", message }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

function stubGateway(status = 200): CompletionRequest[] {
  const requests: CompletionRequest[] = [];
  const respond = async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = { url: String(input), body: JSON.parse(String(init?.body)) as CompletionRequest["body"] };
    requests.push(request);
    if (status !== 200) return new Response(JSON.stringify({ error: { message: "model not found" } }), { status });
    const prompt = request.body.messages.find((message) => message.role === "user")!.content;
    const turnId = (JSON.parse(prompt) as { turns: { turn_id: string }[] }).turns[0].turn_id;
    const body = JSON.stringify(completion(JSON.stringify(submitted(turnId))));
    return new Response(body, { status: 200, headers: { "content-type": "application/json" } });
  };
  vi.stubGlobal("fetch", vi.fn(respond));
  return requests;
}

describe("TraceThread readable view", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceCall).mockReset().mockResolvedValue(trace);
    vi.mocked(agentTraceSpanCall)
      .mockReset()
      .mockImplementation(async (_token, _trace, id) => details[id]);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("asks the gateway's renderer agent for a transcript and shows it with links back to steps", async () => {
    const user = userEvent.setup();
    const requests = stubGateway();
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    expect(await within(thread).findByText("Read the release notes")).toBeVisible();
    expect(requests).toHaveLength(0);

    await user.click(within(thread).getByRole("button", { name: "Readable", pressed: false }));
    expect(await within(thread).findByText("Read the release notes, please")).toBeVisible();
    expect(within(thread).getByText("Read the changelog before answering")).toBeVisible();
    expect(within(thread).getByText("The release is ready")).toBeVisible();
    expect(within(thread).getByText("Rendered by fireworks_ai/deepseek-v4-pro")).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe("http://proxy.test/chat/completions");
    expect(requests[0].body.model).toBe("fireworks_ai/deepseek-v4-pro");
    expect(requests[0].body.messages.at(-1)?.content).toContain("CHANGELOG.md");

    await user.click(within(thread).getByRole("button", { name: "Inspect Read CHANGELOG.md" }));
    expect(screen.getByRole("tab", { name: "Steps", selected: true })).toBeVisible();
    expect(screen.getByRole("treeitem", { selected: true })).toHaveAttribute("data-row-id", "tool");
  });

  it("keeps the recorded thread and explains the failure when the renderer cannot run", async () => {
    const user = userEvent.setup();
    stubGateway(404);
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Thread" }));
    const thread = await screen.findByRole("region", { name: "Trace thread" });
    await user.click(await within(thread).findByRole("button", { name: "Readable" }));
    expect(await within(thread).findByRole("alert")).toHaveTextContent(/^Readable view failed: .*model not found/);
    expect(within(thread).getByText("Read the release notes")).toBeVisible();
    expect(within(thread).getByText("The release is ready")).toBeVisible();
  });
});
