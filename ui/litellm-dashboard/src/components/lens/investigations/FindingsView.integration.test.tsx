import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithLens, stubGateway } from "@/../tests/lens-test-utils";
import { testQueryClient } from "@/../tests/test-utils";
import { createLensDemoData } from "../data/demo/fixtures";
import { findingKey, inboxRows } from "../model/inbox";
import type { Lens } from "../model/types";
import { FindingsView } from "./FindingsView";

let proxy = stubGateway();
const data = createLensDemoData();
const support = data.lenses[0];
const twin: Lens = { ...support, id: "twin", settings: { ...support.settings, name: "Second support review" } };
const lenses = [support, twin, data.lenses[1]];
const issue = support.findings.find((finding) => finding.kind === "issue")!;

beforeEach(() => {
  testQueryClient.clear();
  proxy = stubGateway();
  proxy.get.mockImplementation(async (path) =>
    path === "/lens" ? { lenses, workers: [], tracing_enabled: true } : { data: [] },
  );
});

it("deduplicates findings across investigations and applies feedback to every source", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithLens(<FindingsView />, { searchParams: "?tab=findings", onUrlUpdate });
  const rows = await screen.findAllByRole("row", { name: issue.title });
  expect(rows).toHaveLength(1);
  expect(within(rows[0]).getByRole("cell", { name: "2", exact: true })).toBeVisible();
  await user.click(rows[0]);
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  fireEvent.change(within(panel).getByRole("textbox"), { target: { value: "A handoff now handles failures" } });
  await user.click(within(panel).getByRole("button", { name: "Mark resolved" }));
  await waitFor(() => expect(proxy.patch).toHaveBeenCalledTimes(2));
  expect(proxy.patch.mock.calls.map(([path, request]) => [path, request.body])).toEqual([
    [`/lens/support/findings/${issue.id}`, { status: "resolved", reason: "A handoff now handles failures" }],
    [`/lens/twin/findings/${issue.id}`, { status: "resolved", reason: "A handoff now handles failures" }],
  ]);
  await waitFor(() => expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).has("issue")).toBe(false));
});

it("restores inbox filters from a link and keeps filter changes in the URL", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithLens(<FindingsView />, {
    searchParams: "?tab=findings&inbox_agent=support_agent&priority=high",
    onUrlUpdate,
  });
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
  expect(screen.queryByRole("row", { name: data.lenses[1].findings[0].title })).not.toBeInTheDocument();
  await user.click(screen.getByRole("combobox", { name: "Filter by priority" }));
  await user.click(screen.getByRole("option", { name: "Low", exact: true }));
  expect(await screen.findByText("No findings match these filters.")).toBeVisible();
  await waitFor(() =>
    expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).get("priority")).toBe("low"),
  );
});

it("reopens an inbox finding from a link and navigates between findings with the keyboard", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  const rows = inboxRows(lenses);
  renderWithLens(<FindingsView readOnly />, {
    searchParams: `?tab=findings&issue=${encodeURIComponent(rows[0].key)}`,
    onUrlUpdate,
  });
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(within(panel).getByRole("heading", { name: rows[0].title })).toBeVisible();
  expect(within(panel).queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
  await user.keyboard("j");
  expect(await within(panel).findByRole("heading", { name: rows[1].title })).toBeVisible();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).has("issue")).toBe(false));
});

it("retains feedback and the open finding when a grouped review fails", async () => {
  const user = userEvent.setup();
  proxy.patch.mockImplementation(async (path) => {
    if (path.startsWith("/lens/twin/")) throw new Error("Review could not be saved");
  });
  renderWithLens(<FindingsView />, { searchParams: "?tab=findings" });
  await user.click(await screen.findByRole("row", { name: issue.title }));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "Keep this feedback" } });
  await user.click(screen.getByRole("button", { name: "Mark resolved" }));
  expect(await screen.findByText("Review could not be saved")).toBeVisible();
  expect(screen.getByRole("textbox")).toHaveValue("Keep this feedback");
  expect(screen.getByRole("button", { name: "Mark resolved" })).toBeEnabled();
});

it("reviews only the selected check when two findings have the same title", async () => {
  const other = { ...issue, id: "different-check", check_id: "different-check" };
  proxy.get.mockImplementation(async (path) =>
    path === "/lens"
      ? { lenses: [{ ...support, findings: [issue, other] }], workers: [], tracing_enabled: true }
      : { data: [] },
  );
  const user = userEvent.setup();
  renderWithLens(<FindingsView />, { searchParams: "?tab=findings" });
  const rows = await screen.findAllByRole("row", { name: issue.title });
  expect(rows).toHaveLength(2);
  await user.click(rows[0]);
  await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
  await waitFor(() => expect(proxy.patch).toHaveBeenCalledTimes(1));
  expect(proxy.patch.mock.calls[0][0]).toBe(`/lens/support/findings/${issue.id}`);
});

it("opens a grouped finding from a link to any of its owning investigations", async () => {
  renderWithLens(<FindingsView readOnly />, {
    searchParams: `?tab=findings&issue=${encodeURIComponent(findingKey(twin, issue))}`,
  });
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(within(panel).getByRole("heading", { name: issue.title })).toBeVisible();
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "true");
});
