import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import type { RunSelection } from "@/components/view_logs/TraceView/traceRouting";

import { renderWithLens } from "@/../tests/lens-test-utils";
import { Inspector } from "@/components/shared/Inspector";

import type { OwnedFinding } from "../model/inbox";
import type { Finding, Lens } from "../model/types";
import { FindingPanel, ownedFindingKey } from "./FindingDetails";

vi.mock("@/components/view_logs/TraceView/TraceDrawer", () => ({
  RunView: ({ traceId, selection }: { traceId: string; selection: RunSelection }) => (
    <div data-testid="run-view">
      {traceId} at {selection.spanId}
    </div>
  ),
}));

const lens = {
  id: "lens",
  settings: { name: "Support reviews", agent_name: "support_agent" },
  jobs: [],
  findings: [],
} as unknown as Lens;

const finding: Finding = {
  id: "finding",
  check_id: "check",
  title: "Repeated retries",
  description: "The tool kept failing",
  kind: "issue",
  priority: "high",
  status: "open",
  revision: 1,
  first_seen: "2026-09-30T10:00:00Z",
  last_seen: "2026-09-30T10:00:00Z",
  reason: "Saved feedback",
  suggestion: "",
  limitation: "",
  occurrences: [],
  evidence: [],
};

const owned = (current: Finding | null): OwnedFinding | null => (current ? { lens, finding: current } : null);

function Harness({ current, onReview }: { current: Finding | null; onReview: FindingPanelReview }) {
  return (
    <Inspector.Root
      items={[]}
      itemKey={ownedFindingKey}
      selected={owned(current)}
      onSelectedChange={vi.fn()}
      noun="finding"
      storageKey="test.finding"
    >
      <FindingPanel readOnly={false} busy={false} onReview={onReview} />
    </Inspector.Root>
  );
}
type FindingPanelReview = Parameters<typeof FindingPanel>[0]["onReview"];

afterEach(() => vi.restoreAllMocks());

it("keeps a feedback draft during refreshes, sends it with status changes, and restores saved feedback on reopening", async () => {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: query.includes("reduce"),
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }) as unknown as MediaQueryList,
  );
  const user = userEvent.setup();
  const onReview = vi.fn();
  const view = renderWithLens(<Harness current={finding} onReview={onReview} />);
  const reason = () => screen.getByRole("textbox", { name: "What should Lens remember?" });
  expect(screen.getByRole("complementary", { name: "Finding details" })).toHaveTextContent("support_agent");
  expect(reason()).toHaveValue("Saved feedback");
  fireEvent.change(reason(), { target: { value: "Draft feedback" } });
  await user.click(screen.getByRole("button", { name: "Mark resolved" }));
  expect(onReview).toHaveBeenLastCalledWith({ lens, finding }, "resolved", "Draft feedback");

  const refreshed: Finding = { ...finding, reason: "Server feedback", status: "resolved" };
  view.rerender(<Harness current={refreshed} onReview={onReview} />);
  expect(reason()).toHaveValue("Draft feedback");
  await user.click(screen.getByRole("button", { name: "Reopen" }));
  expect(onReview).toHaveBeenLastCalledWith({ lens, finding: refreshed }, "open", "Draft feedback");
  await user.click(screen.getByRole("button", { name: "This is expected" }));
  expect(onReview).toHaveBeenLastCalledWith({ lens, finding: refreshed }, "dismissed", "Draft feedback");

  view.rerender(<Harness current={null} onReview={onReview} />);
  expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  view.rerender(<Harness current={refreshed} onReview={onReview} />);
  expect(reason()).toHaveValue("Server feedback");
  const other: Finding = { ...finding, id: "other", reason: "Other feedback" };
  view.rerender(<Harness current={other} onReview={onReview} />);
  expect(reason()).toHaveValue("Other feedback");
});

it("stacks a quote's original step over the finding and keeps the feedback draft on the way back", async () => {
  const user = userEvent.setup();
  const traceOf = (id: string) => btoa(JSON.stringify(["traces", "", id]));
  const quoted: Finding = {
    ...finding,
    evidence: [
      { execution_id: traceOf("trace-1"), span_id: "step-a", quote: "first quote", role: "support" },
      { execution_id: traceOf("trace-2"), span_id: "step-b", quote: "second quote", role: "support" },
    ],
  };
  const onUrlUpdate = vi.fn();
  renderWithLens(<Harness current={quoted} onReview={vi.fn()} />, { onUrlUpdate });
  const panel = screen.getByRole("complementary", { name: "Finding details" });
  const reason = () => within(panel).getByRole("textbox", { name: "What should Lens remember?", hidden: true });
  fireEvent.change(reason(), { target: { value: "Draft feedback" } });
  for (const summary of within(panel).getAllByText(/quote$/)) await user.click(summary);
  await user.click(within(panel).getAllByRole("button", { name: "Open original step" })[0]);
  expect(await within(panel).findByTestId("run-view")).toHaveTextContent("trace-1 at step-a");
  expect(reason()).not.toBeVisible();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

  await user.click(within(panel).getByRole("button", { name: "Back to finding" }));
  expect(within(panel).queryByTestId("run-view")).not.toBeInTheDocument();
  expect(reason()).toBeVisible();
  expect(reason()).toHaveValue("Draft feedback");

  await user.click(within(panel).getAllByRole("button", { name: "Open original step" })[1]);
  expect(await within(panel).findByTestId("run-view")).toHaveTextContent("trace-2 at step-b");
  const url = new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
  expect(url.get("evidence")).toBe(traceOf("trace-2"));
  expect(url.get("evidence_span")).toBe("step-b");
});
