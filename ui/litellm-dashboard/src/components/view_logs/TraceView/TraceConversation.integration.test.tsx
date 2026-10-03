import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import { RunView } from "./TraceDrawer";
import { TraceConversation } from "./TraceConversation";
import type { SpanDetail, Trace } from "./traceTypes";
import research from "./__fixtures__/research_trace.json";

vi.mock("../../networking", () => ({
  agentTraceCall: vi.fn(),
  agentTraceSpanCall: vi.fn(),
  getProxyBaseUrl: () => "http://proxy.test",
}));
import { agentTraceCall, agentTraceSpanCall } from "../../networking";

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
    renderWithProviders(<RunView traceId={trace.summary.trace_id} accessToken="test" onBack={vi.fn()} embedded />);
    await user.click(await screen.findByRole("button", { name: "Trace view: Steps" }));
    await user.click(screen.getByRole("menuitemradio", { name: "Conversation" }));
    const conversation = await screen.findByRole("region", { name: "Trace conversation" });
    expect(await within(conversation).findByText("Read the release notes")).toBeVisible();
    await user.click(await within(conversation).findByRole("button", { name: "Expand read_file tool call" }));
    expect(await within(conversation).findByText("CHANGELOG.md")).toBeVisible();
    expect(await within(conversation).findByText("All checks passed")).toBeVisible();
    expect(await within(conversation).findByText("The release is ready")).toBeVisible();
    const toolStep = within(conversation).getByRole("region", { name: "Conversation step read_file" });
    await user.click(within(toolStep).getByRole("button", { name: "Inspect step read_file" }));
    expect(screen.getByRole("button", { name: "Trace view: Steps" })).toBeVisible();
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

  it("shows a missing step explicitly and lets the user retry it", async () => {
    const user = userEvent.setup();
    vi.mocked(agentTraceSpanCall).mockRejectedValueOnce(new Error("temporarily unavailable"));
    renderWithProviders(<TraceConversation trace={trace} accessToken="test" onOpenStep={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("missing from the conversation");
    await user.click(screen.getByRole("button", { name: "Retry step" }));
    expect(await screen.findByText("Read the release notes")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
