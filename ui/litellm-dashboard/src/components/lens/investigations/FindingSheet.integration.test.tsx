import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import type { Finding } from "../model/types";
import { FindingSheet } from "./FindingSheet";

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

it("keeps a feedback draft during refreshes, sends it with status changes, and restores saved feedback on reopening", async () => {
  const user = userEvent.setup();
  const changeFinding = vi.fn().mockResolvedValue(undefined);
  const props = { sampledRuns: [], readOnly: false, busy: false, onClose: vi.fn(), onEvidence: vi.fn(), changeFinding };
  const view = renderWithProviders(<FindingSheet {...props} finding={finding} />);
  const reason = () => screen.getByRole("textbox", { name: "What should Lens remember?" });
  expect(reason()).toHaveValue("Saved feedback");
  fireEvent.change(reason(), { target: { value: "Draft feedback" } });
  await user.click(screen.getByRole("button", { name: "Mark resolved" }));
  expect(changeFinding).toHaveBeenLastCalledWith("resolved", "Draft feedback");

  const refreshed: Finding = { ...finding, reason: "Server feedback", status: "resolved" };
  view.rerender(<FindingSheet {...props} finding={refreshed} />);
  expect(reason()).toHaveValue("Draft feedback");
  await user.click(screen.getByRole("button", { name: "Reopen" }));
  expect(changeFinding).toHaveBeenLastCalledWith("open", "Draft feedback");
  await user.click(screen.getByRole("button", { name: "This is expected" }));
  expect(changeFinding).toHaveBeenLastCalledWith("dismissed", "Draft feedback");

  view.rerender(<FindingSheet {...props} />);
  view.rerender(<FindingSheet {...props} finding={refreshed} />);
  expect(reason()).toHaveValue("Server feedback");
  const other: Finding = { ...finding, id: "other", reason: "Other feedback" };
  view.rerender(<FindingSheet {...props} finding={other} />);
  expect(reason()).toHaveValue("Other feedback");
});
