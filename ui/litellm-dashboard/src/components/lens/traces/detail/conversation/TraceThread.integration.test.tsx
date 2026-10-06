import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
});
