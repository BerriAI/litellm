import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { Inspector } from "@/components/shared/Inspector";

import type { OwnedFinding } from "../model/inbox";
import type { Finding, Lens } from "../model/types";
import { FindingPanel, ownedFindingKey } from "./FindingDetails";

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
  const view = renderWithProviders(<Harness current={finding} onReview={onReview} />);
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
