import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithLens, stubGateway } from "@/../tests/lens-test-utils";
import { testQueryClient } from "@/../tests/test-utils";
import { createLensDemoData } from "../data/demo/fixtures";
import { findingKey, inboxRows } from "../model/inbox";
import type { Lens } from "../model/types";
import { InvestigationsView } from "./InvestigationsView";

let proxy = stubGateway();
const data = createLensDemoData();
const support = data.lenses[0];
const twin: Lens = { ...support, id: "twin", settings: { ...support.settings, name: "Second support review" } };
const lenses = [support, twin, data.lenses[1]];
const issue = support.findings.find((finding) => finding.kind === "issue")!;

beforeEach(() => {
  testQueryClient.clear();
  proxy = stubGateway();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens") return { lenses, workers: [], tracing_enabled: true };
    if (path.endsWith("/runs")) return [];
    if (path.endsWith("/reviews")) return { reviews: [], reviewed: 0 };
    return { data: [] };
  });
});

it("deduplicates findings across investigations and applies feedback to every source", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithLens(<InvestigationsView />, { searchParams: "?tab=investigations", onUrlUpdate });
  const rows = await screen.findAllByRole("row", { name: issue.title });
  expect(rows).toHaveLength(1);
  expect(screen.getByRole("columnheader", { name: "Investigation" })).toBeVisible();
  const investigations = `${support.settings.name}, ${twin.settings.name}`;
  expect(within(rows[0]).getByTitle(investigations)).toBeVisible();
  expect(within(rows[0]).getByText(/2 runs/)).toBeVisible();
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
  renderWithLens(<InvestigationsView />, {
    searchParams: "?tab=findings&inbox_agent=support_agent&priority=high",
    onUrlUpdate,
  });
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
  expect(screen.queryByRole("row", { name: data.lenses[1].findings[0].title })).not.toBeInTheDocument();
  await user.click(screen.getByRole("combobox", { name: "Filter by priority" }));
  await user.click(screen.getByRole("option", { name: "Low" }));
  expect(await screen.findByText("No findings match these filters.")).toBeVisible();
  await waitFor(() =>
    expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).get("priority")).toBe("low"),
  );
});

it("reopens an inbox finding from a link and walks the combined rows with the keyboard", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  const rows = inboxRows(lenses);
  renderWithLens(<InvestigationsView readOnly />, {
    searchParams: `?tab=findings&issue=${encodeURIComponent(rows[0].key)}`,
    onUrlUpdate,
  });
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(within(panel).getByRole("heading", { name: rows[0].title })).toBeVisible();
  expect(within(panel).queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
  await user.keyboard("j");
  expect(await screen.findByRole("complementary", { name: "Investigation details" })).toHaveTextContent(
    twin.settings.name,
  );
  await user.keyboard("j");
  expect(await screen.findByRole("heading", { name: data.lenses[1].settings.name })).toBeVisible();
  await user.keyboard("j");
  expect(await within(panel).findByRole("heading", { name: rows[1].title })).toBeVisible();
  await user.keyboard("{Escape}");
  await waitFor(() => expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).has("issue")).toBe(false));
});

it("keeps finding rows collapsed across filter changes and expands them on request", async () => {
  const user = userEvent.setup();
  renderWithLens(<InvestigationsView readOnly />, { searchParams: "?tab=investigations" });
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
  await user.click(screen.getByRole("button", { name: `Hide findings for ${support.settings.name}` }));
  expect(screen.queryByRole("row", { name: issue.title })).not.toBeInTheDocument();
  await user.click(screen.getByRole("combobox", { name: "Filter by priority" }));
  await user.click(screen.getByRole("option", { name: "High" }));
  const expand = await screen.findByRole("button", { name: `Show findings for ${support.settings.name}` });
  expect(expand).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByRole("row", { name: issue.title })).not.toBeInTheDocument();
  await user.click(expand);
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
});

it("keeps grouped findings available when search leaves only a secondary owner", async () => {
  const user = userEvent.setup();
  renderWithLens(<InvestigationsView readOnly />, { searchParams: "?tab=investigations" });
  expect(await screen.findByRole("row", { name: issue.title })).toBeVisible();
  await user.type(screen.getByRole("combobox", { name: "Search investigations" }), twin.settings.name);
  await waitFor(() => expect(screen.queryByRole("row", { name: support.settings.name })).not.toBeInTheDocument());
  expect(await screen.findByRole("row", { name: twin.settings.name })).toBeVisible();
  expect(screen.getByRole("button", { name: `Hide findings for ${twin.settings.name}` })).toHaveAttribute(
    "aria-expanded",
    "true",
  );
  expect(screen.getByRole("row", { name: issue.title })).toBeVisible();
  await user.click(screen.getByRole("row", { name: issue.title }));
  const panel = within(await screen.findByRole("complementary", { name: "Finding details" }));
  expect(panel.getByRole("heading", { name: issue.title })).toBeVisible();
});

it("retains feedback and the open finding when a grouped review fails", async () => {
  const user = userEvent.setup();
  proxy.patch.mockImplementation(async (path) => {
    if (path.startsWith("/lens/twin/")) throw new Error("Review could not be saved");
  });
  renderWithLens(<InvestigationsView />, { searchParams: "?tab=investigations" });
  await user.click(await screen.findByRole("row", { name: issue.title }));
  const panel = within(screen.getByRole("complementary", { name: "Finding details" }));
  fireEvent.change(panel.getByRole("textbox"), { target: { value: "Keep this feedback" } });
  await user.click(screen.getByRole("button", { name: "Mark resolved" }));
  expect(await screen.findByText("Review could not be saved")).toBeVisible();
  expect(panel.getByRole("textbox")).toHaveValue("Keep this feedback");
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
  renderWithLens(<InvestigationsView />, { searchParams: "?tab=investigations" });
  const rows = await screen.findAllByRole("row", { name: issue.title });
  expect(rows).toHaveLength(2);
  expect(within(rows[0]).getByTitle(support.settings.name)).toBeVisible();
  await user.click(rows[0]);
  await user.click(await screen.findByRole("button", { name: "Mark resolved" }));
  await waitFor(() => expect(proxy.patch).toHaveBeenCalledTimes(1));
  expect(proxy.patch.mock.calls[0][0]).toBe(`/lens/support/findings/${issue.id}`);
});

it("opens grouped evidence from its owning investigation and returns to the finding", async () => {
  const requestId = btoa(JSON.stringify(["requests", "", "request-2"]));
  const requestFinding = {
    ...issue,
    occurrences: [requestId],
    evidence: [{ ...issue.evidence[0], execution_id: requestId }],
  };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens")
      return { lenses: [support, { ...twin, findings: [requestFinding] }], workers: [], tracing_enabled: true };
    if (path.includes("/executions/"))
      return { parts: [{ span_id: "request", content: "Saved request body", truncated: false }] };
    return { data: [] };
  });
  const user = userEvent.setup();
  renderWithLens(<InvestigationsView />, { searchParams: "?tab=investigations" });
  await user.click(await screen.findByRole("row", { name: issue.title }));
  await user.click(screen.getByText("request-2"));
  await user.click(screen.getByRole("button", { name: "Open request" }));
  expect(await screen.findByText("Saved request body")).toBeVisible();
  expect(proxy.get).toHaveBeenCalledWith(expect.stringContaining("/lens/twin/executions/"), expect.anything());
  await user.click(screen.getByRole("button", { name: "Back to finding" }));
  expect(await screen.findByRole("heading", { name: issue.title })).toBeVisible();
});

it("opens a grouped finding from a link to any of its owning investigations", async () => {
  renderWithLens(<InvestigationsView readOnly />, {
    searchParams: `?tab=findings&issue=${encodeURIComponent(findingKey(twin, issue))}`,
  });
  const panel = await screen.findByRole("complementary", { name: "Finding details" });
  expect(within(panel).getByRole("heading", { name: issue.title })).toBeVisible();
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "true");
});

it("keeps grouped links stable across investigation sorting and different source finding IDs", async () => {
  const first = { ...issue, priority: "medium" as const };
  const second = { ...issue, id: "other-owner-issue", priority: "high" as const };
  proxy.get.mockImplementation(async (path) =>
    path === "/lens"
      ? {
          lenses: [
            { ...support, findings: [first] },
            { ...twin, findings: [second], created_at: new Date(Date.parse(support.created_at) + 1000).toISOString() },
          ],
          workers: [],
          tracing_enabled: true,
        }
      : { data: [] },
  );
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithLens(<InvestigationsView />, { searchParams: "?tab=investigations", onUrlUpdate });
  await user.click(await screen.findByRole("row", { name: issue.title }));
  const panel = within(await screen.findByRole("complementary", { name: "Finding details" }));
  expect(panel.getByRole("heading", { name: issue.title })).toBeVisible();
  expect(screen.getByRole("row", { name: issue.title })).toHaveAttribute("aria-selected", "true");
  await waitFor(() =>
    expect(new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString).get("issue")).toBe(
      findingKey(support, first),
    ),
  );
  await user.click(panel.getByRole("button", { name: "Mark resolved" }));
  await waitFor(() => expect(proxy.patch).toHaveBeenCalledTimes(2));
  expect(proxy.patch.mock.calls.map(([path]) => path)).toEqual([
    `/lens/support/findings/${first.id}`,
    `/lens/twin/findings/${second.id}`,
  ]);
});
