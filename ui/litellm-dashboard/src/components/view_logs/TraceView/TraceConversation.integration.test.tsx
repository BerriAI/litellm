import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ComponentProps } from "react";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { RunView } from "./TraceDrawer";
import { useOpenTraceRouting } from "./traceRouting";
import { TraceConversation } from "./TraceConversation";
import type { SpanDetail, Trace } from "./traceTypes";
import research from "./__fixtures__/research_trace.json";

vi.mock("../../networking", () => ({
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test",
}));
import { agentTraceCall, agentTraceSpanCall } from "../../networking";

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
    await user.click(await within(conversation).findByRole("button", { name: "Expand read_file tool call" }));
    expect(await within(conversation).findByText("CHANGELOG.md")).toBeVisible();
    expect(await within(conversation).findByText("All checks passed")).toBeVisible();
    expect(await within(conversation).findByText("The release is ready")).toBeVisible();
    const toolStep = within(conversation).getByRole("region", { name: "Conversation step read_file" });
    await user.click(within(toolStep).getByRole("button", { name: "Inspect step read_file" }));
    expect(screen.getByRole("tab", { name: "Steps", selected: true })).toBeVisible();
    expect(screen.getByRole("treeitem", { selected: true })).toHaveAttribute("data-row-id", "tool");
    expect(screen.getByRole("heading", { name: "read_file" })).toBeVisible();
  });

  it("loads only twenty full steps at a time and fetches the remainder on demand", async () => {
    const user = userEvent.setup();
    const spans = [
      root,
      ...Array.from({ length: 30 }, (_, index) => ({ ...tool, span_id: `tool-${index}`, start_offset_ms: index + 1 })),
    ];
    const long = { ...trace, spans } as Trace;
    renderWithProviders(<TraceConversation trace={long} accessToken="test" onOpenStep={vi.fn()} />);
    const more = await screen.findByRole("button", { name: "Load next 11 steps" });
    await waitFor(() => expect(more).toBeEnabled());
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(20);
    expect(screen.queryByText("The release is ready")).not.toBeInTheDocument();
    await user.click(more);
    expect(await screen.findByText("The release is ready")).toBeVisible();
    expect(agentTraceSpanCall).toHaveBeenCalledTimes(31);
    expect(screen.queryByRole("button", { name: /Load next/ })).not.toBeInTheDocument();
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
      const more = screen.getByRole("button", { name: "Load next 5 steps" });
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
    expect(screen.getByText("2 of 4 steps loaded")).toBeVisible();

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
      expect(screen.getAllByText("Investigation timed out")).toHaveLength(1);
      const entries = screen.getAllByRole("region", { name: "Conversation step Investigate release" });
      expect(within(entries[0]).getByText("Investigate failed checks")).toBeVisible();
      expect(within(entries[0]).queryByText("Investigation timed out")).not.toBeInTheDocument();
      expect(within(entries[1]).getByText("Investigation timed out")).toBeVisible();
      if (output) expect(within(entries[1]).getByText(output)).toBeVisible();
    },
  );
});
