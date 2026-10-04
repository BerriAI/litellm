import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { createLensDemoData } from "./demo/createLensDemo";

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
const expectUrl = (onUrlUpdate: ReturnType<typeof vi.fn>, check: (params: URLSearchParams) => void) =>
  waitFor(() => check(lastUrl(onUrlUpdate)));

const network = vi.fn<typeof fetch>();
beforeEach(() => {
  testQueryClient.clear();
  vi.stubGlobal("fetch", network);
  network.mockReset();
  network.mockImplementation(async (input) => {
    const path = new URL(String(input), "http://localhost").pathname;
    if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
    if (path === "/lens") return Response.json({ lenses: [], workers: [], tracing_enabled: false });
    return Response.json({ data: [], traces: false, requests: false });
  });
});

describe("Lens interactive demo", () => {
  it("opens without tracing, keeps the sample session in the URL, and restores the live view without mixing data", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Internal User" readOnly={false} />, {
      onUrlUpdate,
    });
    expect(await screen.findByRole("heading", { name: "The gateway that helps your agents improve" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Explore with sample data" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.getByRole("switch", { name: "Demo data" })).toBeChecked();
    await expectUrl(onUrlUpdate, (url) => expect(url.get("demo")).toBe("true"));
    expect(screen.queryByRole("button", { name: "Set up tracing" })).not.toBeInTheDocument();
    network.mockClear();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(screen.getByText("Where is order #1042?")).toBeVisible();
    const search = screen.getByRole("combobox", { name: "Search runs" });
    await user.type(search, "headphones");
    await waitFor(() => expect(screen.queryByText("Where is order #1042?")).not.toBeInTheDocument());
    expect(screen.getByText("Can I return my headphones?")).toBeVisible();
    await user.clear(search);
    await user.type(search, "agent:support_agent status:error");
    const table = screen.getByRole("table", { name: "Agent runs" });
    await waitFor(() => expect(within(table).getAllByRole("row")).toHaveLength(4));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("q")).toBe("agent:support_agent status:error"));
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("row", { name: /Support quality/ })).toBeVisible();
    expect(screen.queryByRole("button", { name: "New investigation" })).not.toBeInTheDocument();
    expect(network).not.toHaveBeenCalled();
    await user.click(screen.getByRole("switch", { name: "Demo data" }));
    expect(await screen.findByText(/Investigations require proxy administrator access/)).toBeVisible();
    expect(screen.queryByText("Can I return my headphones?")).not.toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Demo data" })).not.toBeChecked();
    await expectUrl(onUrlUpdate, (url) => expect([...url.entries()]).toEqual([["tab", "investigations"]]));
  });

  it("opens the sample session from ?demo=true and keeps the open run and step in the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Internal User" readOnly={false} />, {
      searchParams: "?demo=true",
      onUrlUpdate,
    });
    expect(await screen.findByRole("switch", { name: "Demo data" })).toBeChecked();
    await user.click(await screen.findByText("Where is order #1042?"));
    const drawer = await screen.findByRole("complementary", { name: "Trace details" });
    await expectUrl(onUrlUpdate, (url) => expect(url.get("trace")).toBeTruthy());
    const openRun = lastUrl(onUrlUpdate);
    expect(openRun.get("demo")).toBe("true");
    const run = createLensDemoData().runs.find(({ trace }) => trace.summary.trace_id === openRun.get("trace"));
    const steps = await within(drawer).findAllByRole("treeitem");
    const step = steps.find((row) => run?.trace.spans.some((span) => span.span_id === row.getAttribute("data-row-id")));
    await user.click(step!);
    await expectUrl(onUrlUpdate, (url) => expect(url.get("span")).toBe(step!.getAttribute("data-row-id")));
    expect(lastUrl(onUrlUpdate).get("trace")).toBe(openRun.get("trace"));
    await user.click(within(drawer).getByRole("tab", { name: "Attributes" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("span_tab")).toBe("attributes"));
    await user.click(within(drawer).getByRole("tab", { name: "Conversation" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("view")).toBe("conversation"));
    expect(network).not.toHaveBeenCalled();
    await user.click(screen.getByRole("switch", { name: "Demo data" }));
    expect(await screen.findByRole("heading", { name: "The gateway that helps your agents improve" })).toBeVisible();
    await expectUrl(onUrlUpdate, (url) => expect([...url.keys()]).toEqual([]));
  });

  it("drops the live run selection when entering the sample session", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    network.mockResolvedValue(Response.json({ detail: "Tracing is not enabled" }, { status: 501 }));
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Internal User" readOnly={false} />, {
      searchParams: "?tab=traces&setup=lens&trace=live-trace&span=live-span&lens=live-lens",
      onUrlUpdate,
    });
    await user.click(await screen.findByRole("switch", { name: "Demo data" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    await expectUrl(onUrlUpdate, (url) =>
      expect([...url.entries()]).toEqual([
        ["tab", "traces"],
        ["demo", "true"],
      ]),
    );
    expect(screen.queryByText(/Could not load trace/)).not.toBeInTheDocument();
  });

  it("reopens a shared sample link on the same run, step and section", async () => {
    const data = createLensDemoData();
    const run = data.runs[0].trace;
    const step = run.spans[run.spans.length - 1];
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Internal User" readOnly={false} />, {
      searchParams: `?demo=true&trace=${run.summary.trace_id}&span=${step.span_id}&span_tab=request`,
    });
    const drawer = await screen.findByRole("complementary", { name: "Trace details" });
    expect(await within(drawer).findByRole("treeitem", { selected: true })).toHaveAttribute(
      "data-row-id",
      step.span_id,
    );
    expect(within(drawer).getByRole("tab", { name: "Request" })).toHaveAttribute("aria-selected", "true");
  });

  it("connects findings and history to their original trace without live requests", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
      onUrlUpdate,
    });
    await user.click(await screen.findByRole("button", { name: "Explore with sample data" }));
    network.mockClear();
    await user.click(await screen.findByRole("row", { name: /Repeated lookups leave customers without an answer/ }));
    const finding = screen.getByRole("dialog");
    expect(within(finding).getByText(/The support agent retries/)).toBeVisible();
    const summaries = within(finding).getAllByText("support_agent", { exact: true });
    await user.click(summaries[0]);
    await user.click(within(finding).getAllByRole("button", { name: /Open original step/ })[0]);
    expect(await screen.findByRole("complementary", { name: "Span details" })).toHaveTextContent(
      "I will check that for you.",
    );
    await user.click(screen.getByRole("button", { name: "Copy for agent" }));
    expect(await navigator.clipboard.readText()).toContain("I will check that for you.");
    expect(await navigator.clipboard.readText()).not.toContain("Authorization");
    await user.click(screen.getByRole("tab", { name: "Attributes" }));
    expect(await screen.findByText("gen_ai.agent.name")).toBeVisible();
    await user.click(
      within(screen.getByRole("dialog", { name: "Original run" })).getByRole("button", { name: "Close" }),
    );
    await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }));
    expect(await screen.findByRole("table", { name: "Investigations" })).toBeVisible();
    expect(network).not.toHaveBeenCalled();
    await expectUrl(onUrlUpdate, (url) => expect(url.get("demo")).toBe("true"));
    await expectUrl(onUrlUpdate, (url) => expect(url.has("span")).toBe(false));
    await user.click(screen.getByRole("switch", { name: "Demo data" }));
    expect(await screen.findByRole("heading", { name: "The gateway that helps your agents improve" })).toBeVisible();
  });

  it("has no demo entry for existing investigations, populated traces, or connecting another agent", async () => {
    const user = userEvent.setup();
    const data = createLensDemoData();
    const saved = data.lenses[0];
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: true });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ data: data.runs.map((run) => run.trace.summary) });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: `?tab=investigations&lens=${saved.id}`,
    });
    expect(await screen.findByRole("heading", { name: saved.settings.name })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Traces" }));
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set up tracing" })).not.toBeInTheDocument();
  });

  it("offers sample data in the main panel only for the active tab that needs setup", async () => {
    const user = userEvent.setup();
    const saved = createLensDemoData().lenses[0];
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: false });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: false, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
    expect(await screen.findByRole("button", { name: "Explore with sample data" })).toBeVisible();
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    await user.click(tabs.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("row", { name: new RegExp(saved.settings.name) })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Explore with sample data" })).not.toBeInTheDocument();
    await user.click(tabs.getByRole("tab", { name: "Traces" }));
    await user.click(await screen.findByRole("button", { name: "Explore with sample data" }));
    expect(await screen.findByRole("table", { name: "Agent runs" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
  });

  it("marks the Investigations tab while a scan runs and clears it once the scan finishes", async () => {
    const saved = createLensDemoData().lenses[0];
    const withJob = (status: (typeof saved.jobs)[number]["status"]) => ({
      ...saved,
      jobs: [{ ...saved.jobs[0], status }, ...saved.jobs.slice(1)],
    });
    const lenses = vi.fn(() => [withJob("running")]);
    network.mockImplementation(async (input) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/lens") return Response.json({ lenses: lenses(), workers: [], tracing_enabled: false });
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: false, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
    const tab = within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Investigations" });
    await waitFor(() => expect(tab).toHaveAccessibleDescription("An investigation is running"));
    lenses.mockReturnValue([withJob("completed")]);
    await testQueryClient.refetchQueries({ queryKey: ["lens", "list"] });
    await waitFor(() => expect(tab).toHaveAccessibleDescription(""));
  });
});
