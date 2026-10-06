import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";
import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
import { RunView } from "../run/RunView";
import { useOpenTraceRouting } from "../../routing";
import { TraceConversation } from "./TraceConversation";
import type { SpanDetail, Trace } from "../../types";
import research from "../../__fixtures__/research_trace.json";

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

const root = { ...research.spans[0], span_id: "root", parent_span_id: null };
const tool = { ...root, span_id: "tool", name: "read_file", parent_span_id: "root", type: "tool", start_offset_ms: 1 };
const trace = { ...research, spans: [root, tool] } as Trace;
const rootDetail: SpanDetail = {
  span_id: "root",
  input: '[{"role":"user","content":"Read the release notes"}]',
  output: '[{"role":"assistant","content":"The release is ready"}]',
  attributes: {},
};
const toolDetail: SpanDetail = {
  span_id: "tool",
  input: '{"path":"CHANGELOG.md"}',
  output: "All checks passed",
  attributes: {},
};

describe("TraceConversation", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(agentTraceCall).mockReset().mockResolvedValue(trace);
    vi.mocked(agentTraceSpanCall)
      .mockReset()
      .mockImplementation(async (_token, _trace, id) => (id === "root" ? rootDetail : { ...toolDetail, span_id: id }));
  });

  it.each([false, true])("refreshes unchanged spans without hiding loaded content on failure (%s)", async (failed) => {
    const user = userEvent.setup();
    vi.mocked(agentTraceCall).mockResolvedValue({
      ...trace,
      summary: { ...trace.summary, trace_ref: "resolved-reference" },
    });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("The release is ready")).toBeVisible();
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (failed) throw new Error("content refresh failed");
      return id === "root" ? { ...rootDetail, output: "Updated final answer" } : toolDetail;
    });
    await user.click(screen.getByRole("button", { name: "Refresh run" }));
    if (failed) {
      expect(await screen.findAllByRole("button", { name: "Retry step" })).toHaveLength(2);
      expect(screen.getByText("The release is ready")).toBeVisible();
    } else {
      expect(await screen.findByText("Updated final answer")).toBeVisible();
      expect(screen.queryByText("The release is ready")).not.toBeInTheDocument();
    }
  });

  it("switches to a readable transcript and opens the exact tool step from it", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    const conversationTab = await screen.findByRole("tab", { name: "Conversation", selected: false });
    await user.click(conversationTab);
    expect(conversationTab).toHaveAttribute("aria-selected", "true");
    const conversation = await screen.findByRole("region", { name: "Trace conversation" });
    expect(await within(conversation).findByText("Read the release notes")).toBeVisible();
    expect(await within(conversation).findByRole("button", { name: "Expand read_file tool call" })).toHaveTextContent(
      "CHANGELOG.md",
    );
    await user.click(within(conversation).getByRole("button", { name: "Expand read_file tool call" }));
    expect(await within(conversation).findByText("CHANGELOG.md")).toBeVisible();
    expect(await within(conversation).findByText("All checks passed")).toBeVisible();
    expect(await within(conversation).findByText("The release is ready")).toBeVisible();
    const toolStep = within(conversation).getByRole("region", { name: "Conversation step read_file" });
    await user.click(within(toolStep).getByRole("button", { name: "Inspect step read_file" }));
    expect(screen.getByRole("tab", { name: "Steps", selected: true })).toBeVisible();
    expect(screen.getByRole("treeitem", { selected: true })).toHaveAttribute("data-row-id", "tool");
    expect(screen.getByRole("heading", { name: "read_file" })).toBeVisible();
  });

  it("waits for an in-flight refresh before requesting another conversation page", async () => {
    const user = userEvent.setup();
    const first = { ...trace, spans: [root], next_cursor: "old-page" };
    const refreshed = { ...first, next_cursor: "fresh-page" };
    const second = { ...trace, spans: [tool], next_cursor: null };
    const pending = Promise.withResolvers<Trace>();
    vi.mocked(agentTraceCall)
      .mockResolvedValueOnce(first)
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValue(second);
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("Read the release notes")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Refresh run" }));
    const more = screen.getByRole("button", { name: "Load next 20 entries" });
    expect(more).toBeDisabled();
    await user.click(more);
    expect(agentTraceCall).toHaveBeenCalledTimes(2);
    await act(async () => pending.resolve(refreshed));
    await waitFor(() => expect(more).toBeEnabled());
    await user.click(more);
    expect(await screen.findByRole("button", { name: "Expand read_file tool call" })).toBeVisible();
    expect(vi.mocked(agentTraceCall).mock.calls.map((call) => call[3])).toEqual([null, null, "fresh-page"]);
  });

  it("can load the next conversation page after a refresh fails without invalidating loaded content", async () => {
    const user = userEvent.setup();
    const first = { ...trace, spans: [root], next_cursor: "next-page" };
    const second = { ...trace, spans: [tool], next_cursor: null };
    vi.mocked(agentTraceCall)
      .mockResolvedValueOnce(first)
      .mockRejectedValueOnce(new Error("refresh unavailable"))
      .mockResolvedValue(second);
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("Read the release notes")).toBeVisible();
    const contentReads = vi.mocked(agentTraceSpanCall).mock.calls.length;
    vi.mocked(agentTraceSpanCall).mockRejectedValue(new Error("content unavailable"));
    await user.click(screen.getByRole("button", { name: "Refresh run" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Previously received steps are still shown");
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(contentReads);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
      id === "root" ? rootDetail : toolDetail,
    );
    const more = screen.getByRole("button", { name: "Load next 20 entries" });
    expect(more).toBeEnabled();
    await user.click(more);
    expect(await screen.findByRole("button", { name: "Expand read_file tool call" })).toBeVisible();
    expect(vi.mocked(agentTraceCall).mock.calls.map((call) => call[3])).toEqual([null, null, "next-page"]);
    expect(screen.queryByText(/Could not load more conversation entries/)).not.toBeInTheDocument();
  });

  it("keeps refresh busy until content finishes when live updates pause", async () => {
    const user = userEvent.setup();
    const pending = Promise.withResolvers<SpanDetail>();
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, next_cursor: "next-page" });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("The release is ready")).toBeVisible();
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
      id === "root" ? pending.promise : toolDetail,
    );
    const refresh = screen.getByRole("button", { name: "Refresh run" });
    await user.click(refresh);
    await waitFor(() => expect(testQueryClient.isFetching({ queryKey: ["agentTrace"] })).toBe(0));
    expect(refresh).toBeDisabled();
    const more = screen.getByRole("button", { name: "Load next 20 entries" });
    expect(more).toBeDisabled();
    await user.click(refresh);
    await user.click(screen.getByRole("button", { name: "Live updates" }));
    await act(async () => pending.resolve({ ...rootDetail, output: "Updated final answer" }));
    expect(await screen.findByText("Updated final answer")).toBeVisible();
    await waitFor(() => expect(refresh).toBeEnabled());
    expect(more).toBeEnabled();
    expect(agentTraceCall).toHaveBeenCalledTimes(2);
  });

  it("renders a failed shell exchange in both views and preserves its raw result", async () => {
    const user = userEvent.setup();
    const command = "npm test -- checkout\nprintf 'finished\\n'";
    const output = JSON.stringify({ output: "PASS cart.test.ts\nFAIL checkout.test.ts", exit_code: 1, error: null });
    const failedTool = { ...tool, name: "terminal", status: "error", error: null };
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [root, failedTool] } as Trace);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
      id === "root"
        ? rootDetail
        : {
            ...toolDetail,
            input: JSON.stringify({ command, workdir: "/workspace" }),
            output,
            input_ui: {
              kind: "fields",
              fields: [
                { key: "command", value: command },
                { key: "workdir", value: "/workspace" },
              ],
            },
            output_ui: {
              kind: "fields",
              fields: [
                { key: "output", value: "PASS cart.test.ts\nFAIL checkout.test.ts" },
                { key: "exit_code", value: "1" },
              ],
            },
          },
    );
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    const conversation = await screen.findByRole("region", { name: "Trace conversation" });
    expect(await within(conversation).findByText(/npm test -- checkout/, { selector: "pre" })).toHaveTextContent(
      "printf 'finished\\n'",
    );
    expect(within(conversation).getByText(/PASS cart.test.ts/, { selector: "pre" })).toHaveTextContent(
      "FAIL checkout.test.ts",
    );
    expect(within(conversation).getByText("exit_code")).toBeVisible();
    await user.click(within(conversation).getByRole("button", { name: "Inspect step terminal" }));
    const details = screen.getByRole("complementary", { name: "Span details" });
    expect(await within(details).findByText(/npm test -- checkout/, { selector: "pre" })).toBeVisible();
    const result = within(details).getByRole("region", { name: "Output", exact: true });
    expect(within(result).getByText("exit_code")).toBeVisible();
    await user.click(within(result).getByRole("radio", { name: "Raw" }));
    expect(within(result).getByText(/"exit_code": 1/, { selector: "pre" })).toHaveTextContent('"error": null');
  });

  it("loads twenty span details initially and pages conversation entries on demand", async () => {
    const user = userEvent.setup();
    const spans = [
      root,
      ...Array.from({ length: 30 }, (_, index) => ({ ...tool, span_id: `tool-${index}`, start_offset_ms: index + 1 })),
    ];
    const long = { ...trace, spans } as Trace;
    renderWithProviders(<TraceConversation trace={long} accessToken="test" onOpenStep={vi.fn()} />);
    const more = await screen.findByRole("button", { name: "Load next 20 entries" });
    await waitFor(() => expect(more).toBeEnabled());
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(20);
    expect(screen.queryByText("The release is ready")).not.toBeInTheDocument();
    await user.click(more);
    expect(await screen.findByText("The release is ready")).toBeVisible();
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(31);
    expect(screen.queryByRole("button", { name: /Load next/ })).not.toBeInTheDocument();
  });

  it("shows a loaded agent reply while later trace pages remain available", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceCall).mockResolvedValue({ ...trace, spans: [root], next_cursor: "next-page" });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("The release is ready")).toBeVisible();
    expect(screen.getByRole("button", { name: "Load more steps" })).toBeVisible();
    expect(screen.queryByText("End of conversation")).not.toBeInTheDocument();
  });

  it("adds twenty visible entries across hidden subagents, supplemental spans, and server pages", async () => {
    const user = userEvent.setup();
    const child = { ...root, span_id: "child", parent_span_id: "root", name: "Reviewer", start_offset_ms: 1 };
    const childSpans = Array.from({ length: 50 }, (_, index) => [
      {
        ...tool,
        span_id: `child-${index}`,
        parent_span_id: "child",
        name: `child tool ${index}`,
        start_offset_ms: 2 + index * 2,
      },
      {
        ...tool,
        span_id: `supplement-${index}`,
        parent_span_id: "child",
        name: "claude_code.tool_result",
        framework: "claude-code",
        type: "event",
        start_offset_ms: 3 + index * 2,
      },
    ]).flat();
    const parentSpans = Array.from({ length: 45 }, (_, index) => [
      { ...tool, span_id: `parent-${index}`, name: `parent tool ${index}`, start_offset_ms: 102 + index * 2 },
      { ...tool, span_id: `empty-${index}`, type: "llm", start_offset_ms: 103 + index * 2 },
    ]).flat();
    const spans = [root, child, ...childSpans, ...parentSpans] as Trace["spans"];
    const first: Trace = { ...trace, spans: spans.slice(0, 120), next_cursor: "next-page" };
    const last: Trace = { ...trace, spans: spans.slice(120), next_cursor: null };
    vi.mocked(agentTraceCall).mockImplementation(async (_token, _trace, _ref, cursor) => (cursor ? last : first));
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "root" || id === "child")
        return { ...rootDetail, span_id: id, input: id === "root" ? "Main task" : "Child task", output: "" };
      if (id.startsWith("empty-") || id.startsWith("supplement-"))
        return { span_id: id, input: "", output: "", attributes: {} };
      return { ...toolDetail, span_id: id };
    });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    expect(await screen.findByText("2 entries shown")).toBeVisible();
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(20);
    await user.click(screen.getByRole("button", { name: "Load next 20 entries", exact: true }));
    expect(await screen.findByText("22 entries shown")).toBeVisible();
    expect(screen.getAllByRole("region", { name: /Conversation step parent tool/ })).toHaveLength(20);
    expect(
      screen.queryByRole("button", { name: "Expand parent tool 20 tool call", exact: true }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Conversation step child tool 0", exact: true })).not.toBeVisible();
    expect(agentTraceCall).toHaveBeenCalledWith("test", trace.summary.trace_id, undefined, "next-page");

    await user.click(screen.getByText("Subagent: Reviewer", { exact: true }));
    expect(screen.getAllByRole("region", { name: /Conversation step child tool/ })).toHaveLength(19);
    await user.click(screen.getByRole("button", { name: "Load next 20 entries in Reviewer", exact: true }));
    expect(screen.getAllByRole("region", { name: /Conversation step child tool/ })).toHaveLength(39);
    expect(screen.getAllByRole("region", { name: /Conversation step parent tool/ })).toHaveLength(20);
    expect(screen.queryByText("End of conversation")).not.toBeInTheDocument();
  });

  it.each([
    { duration: 10, nextCursor: null },
    { duration: 1000, nextCursor: null },
    { duration: 10, nextCursor: "unrelated-page" },
  ])("hides an exhausted subagent control with unrelated work remaining (%j)", async ({ duration, nextCursor }) => {
    const user = userEvent.setup();
    const child = {
      ...root,
      span_id: "child",
      parent_span_id: "root",
      name: "Reviewer",
      start_offset_ms: 1,
      duration_ms: duration,
    };
    const spans = [
      root,
      child,
      { ...tool, span_id: "child-tool", parent_span_id: "child", start_offset_ms: 2, duration_ms: 1 },
      ...Array.from({ length: 40 }, (_, index) => ({
        ...tool,
        span_id: `parent-${index}`,
        start_offset_ms: 100 + index,
        duration_ms: 1,
      })),
    ] as Trace["spans"];
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "child") return { ...rootDetail, span_id: id, input: "Review task", output: "Review finished" };
      return id === "root" ? rootDetail : { ...toolDetail, span_id: id };
    });
    const loadMore = vi.fn();
    renderWithProviders(
      <TraceConversation
        trace={{ ...trace, spans, next_cursor: nextCursor }}
        accessToken="test"
        onOpenStep={vi.fn()}
        paging={{ loading: false, failed: false, loadMore }}
      />,
    );
    const more = await screen.findByRole("button", { name: "Load next 20 entries", exact: true });
    await waitFor(() => expect(more).toBeEnabled());
    await user.click(screen.getByText("Subagent: Reviewer", { exact: true }));
    expect(screen.getByText("Review finished")).toBeVisible();
    expect(screen.queryByRole("button", { name: /entries in Reviewer/ })).not.toBeInTheDocument();
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(20);
    expect(loadMore).not.toHaveBeenCalled();
  });

  it.each([false, true])("stops loading at the selected subagent's end (server paging: %s)", async (paged) => {
    const user = userEvent.setup();
    const child = {
      ...root,
      span_id: "child",
      parent_span_id: "root",
      name: "Reviewer",
      start_offset_ms: 1,
      duration_ms: 60,
    };
    const spans = [
      root,
      child,
      ...Array.from({ length: 25 }, (_, index) => ({
        ...tool,
        span_id: `child-${index}`,
        parent_span_id: "child",
        name: `child tool ${index}`,
        start_offset_ms: 2 + index,
        duration_ms: 1,
      })),
      ...Array.from({ length: 40 }, (_, index) => ({
        ...tool,
        span_id: `parent-${index}`,
        name: `parent tool ${index}`,
        start_offset_ms: 100 + index,
        duration_ms: 1,
      })),
    ] as Trace["spans"];
    const first: Trace = {
      ...trace,
      spans: paged ? spans.slice(0, 20) : spans,
      next_cursor: paged ? "child-page" : null,
    };
    const next: Trace = { ...trace, spans: spans.slice(20, 40), next_cursor: "unrelated-page" };
    vi.mocked(agentTraceCall).mockImplementation(async (_token, _trace, _ref, cursor) => (cursor ? next : first));
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "child") return { ...rootDetail, span_id: id, input: "Review task", output: "Review finished" };
      return id === "root" ? rootDetail : { ...toolDetail, span_id: id };
    });
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    await user.click(await screen.findByText("Subagent: Reviewer", { exact: true }));
    const more = await screen.findByRole("button", { name: "Load next 20 entries in Reviewer" });
    await waitFor(() => expect(more).toBeEnabled());
    await user.click(more);
    expect(await screen.findByRole("button", { name: "Expand child tool 24 tool call" })).toBeVisible();
    await waitFor(() => expect(screen.queryByRole("button", { name: /entries in Reviewer/ })).not.toBeInTheDocument());
    expect(screen.getByText("Review finished")).toBeVisible();
    expect(screen.getAllByRole("region", { name: /Conversation step child tool/ })).toHaveLength(25);
    expect(screen.getByRole("button", { name: "Load next 20 entries", exact: true })).toBeEnabled();
    expect(screen.queryByText("Loading conversation…")).not.toBeInTheDocument();
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(40);
    expect(agentTraceCall).toHaveBeenCalledTimes(paged ? 2 : 1);
    expect(agentTraceCall).not.toHaveBeenCalledWith("test", trace.summary.trace_id, undefined, "unrelated-page");
  });

  it("stops at a failed server page and resumes the requested entries after retry", async () => {
    const user = userEvent.setup();
    const first: Trace = { ...trace, next_cursor: "next-page" };
    const last: Trace = {
      ...trace,
      spans: [{ ...tool, span_id: "later", name: "later tool", start_offset_ms: 2 }],
      next_cursor: null,
    };
    vi.mocked(agentTraceCall)
      .mockResolvedValueOnce(first)
      .mockRejectedValueOnce(new Error("temporarily unavailable"))
      .mockResolvedValue(last);
    renderWithProviders(
      <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
    );
    await user.click(await screen.findByRole("tab", { name: "Conversation" }));
    const more = await screen.findByRole("button", { name: "Load next 20 entries", exact: true });
    await waitFor(() => expect(more).toBeEnabled());
    await user.click(more);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load more conversation entries");
    expect(more).toBeDisabled();
    expect(agentTraceCall).toHaveBeenCalledTimes(2);
    expect(screen.getByText("Read the release notes")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Retry", exact: true }));
    expect(await screen.findByRole("button", { name: "Expand later tool tool call" })).toBeVisible();
    expect(await screen.findByText("End of conversation")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each([false, true])(
    "checks for missing replies after the final trace page and details load (reply: %s)",
    async (hasReply) => {
      const user = userEvent.setup();
      const laterDetail = Promise.withResolvers<SpanDetail>();
      const summary = { ...trace.summary, span_count: 2 };
      const first: Trace = { ...trace, summary, spans: [root], next_cursor: "last-page" };
      const last: Trace = {
        ...trace,
        summary,
        spans: [{ ...tool, span_id: "later", name: "later response", type: "llm" }],
        next_cursor: null,
      };
      vi.mocked(agentTraceCall).mockImplementation(async (_token, _trace, _ref, cursor) => (cursor ? last : first));
      vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
        id === "root" ? { ...rootDetail, output: "", attributes: { "span.type": "llm_request" } } : laterDetail.promise,
      );
      renderWithProviders(
        <RoutedRunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />,
      );
      await user.click(await screen.findByRole("tab", { name: "Conversation" }));
      expect(await screen.findByText("Read the release notes")).toBeVisible();
      expect(screen.queryByText(/no recorded assistant replies/)).not.toBeInTheDocument();
      expect(screen.queryByText("End of conversation")).not.toBeInTheDocument();
      await user.click(screen.getByRole("button", { name: "Load more steps" }));
      expect(await screen.findByText("Loading conversation…")).toBeVisible();
      expect(screen.queryByText(/no recorded assistant replies/)).not.toBeInTheDocument();
      expect(screen.queryByText("End of conversation")).not.toBeInTheDocument();
      const resolved: SpanDetail = {
        span_id: "later",
        input: "",
        output: hasReply ? rootDetail.output : "",
        attributes: hasReply ? { "event.name": "assistant_response" } : {},
      };
      await act(async () => laterDetail.resolve(resolved));
      expect(await screen.findByText("End of conversation")).toBeVisible();
      if (hasReply) {
        expect(screen.getByText("The release is ready")).toBeVisible();
        expect(screen.queryByText(/no recorded assistant replies/)).not.toBeInTheDocument();
      } else {
        expect(screen.getByText(/no recorded assistant replies/)).toBeVisible();
      }
    },
  );

  it("shows distinct agent invocation labels together with each step's time", async () => {
    const first = { ...root, name: "reviewer", start_offset_ms: 1000 };
    const second = { ...first, span_id: "second", start_offset_ms: 2000 };
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => ({
      ...rootDetail,
      span_id: id,
      input: id === "root" ? "Review code" : "Review tests",
      output: "",
    }));
    renderWithProviders(
      <TraceConversation
        trace={{ ...trace, spans: [first, second] } as Trace}
        accessToken="test"
        onOpenStep={vi.fn()}
      />,
    );
    expect(await screen.findByText("reviewer (1)")).toBeVisible();
    expect(screen.getByText("reviewer (2)")).toBeVisible();
    const steps = screen.getAllByRole("region", { name: "Conversation step reviewer" });
    expect(within(steps[0]).getByText("1.00s")).toBeVisible();
    expect(within(steps[1]).getByText("2.00s")).toBeVisible();
  });

  it.each([rootDetail.output, ""])(
    "keeps the root failure visible while loading, then places it in order (output: %s)",
    async (output) => {
      const user = userEvent.setup();
      const rootFetch = Promise.withResolvers<SpanDetail>();
      const failedRoot = { ...root, status: "error", error: "Agent exceeded its execution limit" };
      const spans = [
        failedRoot,
        ...Array.from({ length: 24 }, (_, index) => ({
          ...tool,
          span_id: `tool-${index}`,
          start_offset_ms: index + 1,
        })),
      ];
      vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) =>
        id === "root" ? rootFetch.promise : { ...toolDetail, span_id: id },
      );
      renderWithProviders(
        <TraceConversation trace={{ ...trace, spans } as Trace} accessToken="test" onOpenStep={vi.fn()} />,
      );
      expect(screen.getAllByText("Agent exceeded its execution limit")).toHaveLength(1);
      await act(async () => rootFetch.resolve({ ...rootDetail, output }));
      const more = screen.getByRole("button", { name: "Load next 20 entries" });
      await waitFor(() => expect(more).toBeEnabled());
      expect(screen.getAllByText("Agent exceeded its execution limit")).toHaveLength(1);
      expect(screen.queryByText("The release is ready")).not.toBeInTheDocument();
      await user.click(more);
      expect(await screen.findByText("End of conversation")).toBeVisible();
      expect(screen.getAllByText("Agent exceeded its execution limit")).toHaveLength(1);
      const rootSteps = screen.getAllByRole("region", { name: `Conversation step ${root.name}` });
      expect(within(rootSteps.at(-1)!).getByText("Agent exceeded its execution limit")).toBeVisible();
    },
  );

  it("shows a missing step explicitly and lets the user retry it", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceSpanCall).mockRejectedValueOnce(new Error("temporarily unavailable"));
    renderWithProviders(<TraceConversation trace={trace} accessToken="test" onOpenStep={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Retry this step to continue the conversation");
    await user.click(screen.getByRole("button", { name: "Retry step" }));
    expect(await screen.findByText("Read the release notes")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("pauses a requested page at a failed prefetched detail until it is retried", async () => {
    const user = userEvent.setup();
    const spans = [
      root,
      ...Array.from({ length: 45 }, (_, index) => ({
        ...tool,
        span_id: `tool-${index}`,
        name: `check ${index}`,
        start_offset_ms: index + 1,
      })),
    ] as Trace["spans"];
    const retry = vi.fn().mockRejectedValueOnce(new Error("unavailable")).mockResolvedValue(toolDetail);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "tool-21") return retry();
      return id === "root" ? rootDetail : { ...toolDetail, span_id: id };
    });
    renderWithProviders(<TraceConversation trace={{ ...trace, spans }} accessToken="test" onOpenStep={vi.fn()} />);
    const more = screen.getByRole("button", { name: "Load next 20 entries", exact: true });
    await waitFor(() => expect(more).toBeEnabled());
    await user.click(more);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load check 21");
    expect(retry).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Expand check 22 tool call" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry step" }));
    expect(await screen.findByText("40 entries shown")).toBeVisible();
    expect(screen.getAllByRole("region", { name: /Conversation step check/ })).toHaveLength(39);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("holds later turns and the final answer behind a failed step until retry succeeds", async () => {
    const user = userEvent.setup();
    const first = { ...tool, span_id: "first", name: "First response", type: "llm", start_offset_ms: 1 };
    const failedTool = { ...tool, start_offset_ms: 2 };
    const last = { ...first, span_id: "last", name: "Final response", start_offset_ms: 3 };
    const traced = { ...trace, spans: [root, first, failedTool, last] } as Trace;
    const question = { role: "user", content: "Read the release notes" };
    const checking = {
      role: "assistant",
      content: "Checking the release",
      tool_calls: [{ name: "read_file", args: { path: "CHANGELOG.md" } }],
    };
    const firstDetail = { ...rootDetail, span_id: "first", output: JSON.stringify([checking]) };
    const lastDetail = {
      ...rootDetail,
      span_id: "last",
      input: JSON.stringify([question, checking, { role: "tool", content: "All checks passed" }]),
    };
    const toolFetch = vi.fn().mockRejectedValueOnce(new Error("unavailable")).mockResolvedValue(toolDetail);
    vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
      if (id === "tool") return toolFetch();
      if (id === "first") return firstDetail;
      if (id === "last") return lastDetail;
      return rootDetail;
    });
    renderWithProviders(<TraceConversation trace={traced} accessToken="test" onOpenStep={vi.fn()} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("read_file");
    expect(await screen.findByText("Checking the release")).toBeVisible();
    expect(screen.queryByText("The release is ready")).not.toBeInTheDocument();
    expect(screen.queryByText("End of conversation")).not.toBeInTheDocument();
    expect(screen.getByText("2 entries shown")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Retry step" }));
    expect(await screen.findByText("The release is ready")).toBeVisible();
    expect(screen.getAllByText("Checking the release")).toHaveLength(1);
    expect(screen.getAllByText("Read the release notes")).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: "Expand read_file tool call" })).toHaveLength(1);
    expect(screen.getByText("End of conversation")).toBeVisible();
  });

  it.each(["Partial investigation", ""])(
    "shows a failed child agent's error once after its work, with output %j",
    async (output) => {
      const user = userEvent.setup();
      const agent = {
        ...root,
        span_id: "child",
        parent_span_id: "root",
        name: "Investigate release",
        type: "agent",
        start_offset_ms: 1,
        duration_ms: 10,
        status: "error",
        error: "Investigation timed out",
      };
      const childTool = { ...tool, parent_span_id: "child", start_offset_ms: 2, duration_ms: 2 };
      const traced = { ...trace, spans: [root, agent, childTool] } as Trace;
      vi.mocked(agentTraceSpanCall).mockImplementation(async (_token, _trace, id) => {
        if (id === "child") return { ...rootDetail, span_id: id, input: "Investigate failed checks", output };
        return id === "root" ? rootDetail : toolDetail;
      });
      renderWithProviders(<TraceConversation trace={traced} accessToken="test" onOpenStep={vi.fn()} />);

      expect(await screen.findByText("End of conversation")).toBeVisible();
      await user.click(screen.getByText("Subagent: Investigate release", { exact: true }));
      expect(screen.getAllByText("Investigation timed out")).toHaveLength(1);
      const entries = screen.getAllByRole("region", { name: "Conversation step Investigate release" });
      expect(within(entries[0]).getByText("Investigate failed checks")).toBeVisible();
      expect(within(entries[0]).queryByText("Investigation timed out")).not.toBeInTheDocument();
      expect(within(entries[1]).getByText("Investigation timed out")).toBeVisible();
      if (output) expect(within(entries[1]).getByText(output)).toBeVisible();
    },
  );
});
