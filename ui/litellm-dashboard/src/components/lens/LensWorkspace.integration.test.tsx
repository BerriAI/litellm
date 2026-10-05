import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { dismissLensIntro, readRequest, requestPath } from "@/../tests/lens-test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { lensKeys } from "./data/queries";
import { createLensDemoData } from "./data/demo/fixtures";

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? ""));
const expectUrl = (onUrlUpdate: ReturnType<typeof vi.fn>, check: (params: URLSearchParams) => void) =>
  waitFor(() => check(lastUrl(onUrlUpdate)));

const network = vi.fn<typeof fetch>();
beforeEach(() => {
  testQueryClient.clear();
  window.localStorage.clear();
  window.sessionStorage.clear();
  dismissLensIntro();
  vi.stubGlobal("fetch", network);
  network.mockReset();
  network.mockImplementation(async (input) => {
    const path = requestPath(input);
    if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
    if (path === "/lens") return Response.json({ lenses: [], workers: [], tracing_enabled: false });
    return Response.json({ data: [], traces: false, requests: false });
  });
});

describe("Lens interactive demo", () => {
  it("opens without tracing, keeps the sample session in the URL, and restores the live view without mixing data", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    window.localStorage.clear();
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
    expect(await screen.findByRole("heading", { name: "Enable tracing" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
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
    await user.click(await screen.findByRole("button", { name: "Explore with sample data" }));
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
    window.localStorage.clear();
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
      onUrlUpdate,
    });
    await user.click(await screen.findByRole("button", { name: "Explore with sample data" }));
    network.mockClear();
    await user.click(await screen.findByRole("row", { name: /Repeated lookups leave customers without an answer/ }));
    const finding = screen.getByRole("complementary", { name: "Finding details" });
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
    expect(within(finding).getByRole("button", { name: "Back to finding" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(within(finding).getByRole("button", { name: "Back to finding" }));
    expect(within(finding).getByText(/The support agent retries/)).toBeVisible();
    await user.click(within(finding).getByRole("button", { name: "Close finding (Esc)" }));
    expect(await screen.findByRole("table", { name: "Investigations" })).toBeVisible();
    expect(network).not.toHaveBeenCalled();
    await expectUrl(onUrlUpdate, (url) => expect(url.get("demo")).toBe("true"));
    await expectUrl(onUrlUpdate, (url) => expect(url.has("span")).toBe(false));
    await user.click(screen.getByRole("switch", { name: "Demo data" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("demo")).toBeNull());
    expect(screen.getByRole("switch", { name: "Demo data" })).not.toBeChecked();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("has no demo entry for existing investigations, populated traces, or connecting another agent", async () => {
    const user = userEvent.setup();
    const data = createLensDemoData();
    const saved = data.lenses[0];
    network.mockImplementation(async (input) => {
      const path = requestPath(input);
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

  it("shows the header preview only for the active tab that still needs setup", async () => {
    const user = userEvent.setup();
    const saved = createLensDemoData().lenses[0];
    network.mockImplementation(async (input) => {
      const path = requestPath(input);
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: false });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: false, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
    expect(await screen.findByRole("button", { name: "Preview sample" })).toBeVisible();
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    await user.click(tabs.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("row", { name: new RegExp(saved.settings.name) })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Preview sample" })).not.toBeInTheDocument();
    await user.click(tabs.getByRole("tab", { name: "Traces" }));
    await user.click(await screen.findByRole("button", { name: "Preview sample" }));
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
      const path = requestPath(input);
      if (path === "/lens") return Response.json({ lenses: lenses(), workers: [], tracing_enabled: false });
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: false, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
    const tab = within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Investigations" });
    await waitFor(() => expect(tab).toHaveAccessibleDescription("An investigation is running"));
    lenses.mockReturnValue([withJob("completed")]);
    await testQueryClient.refetchQueries({ queryKey: lensKeys.lists() });
    await waitFor(() => expect(tab).toHaveAccessibleDescription(""));
  });

  it("marks the Investigations tab while the inline editor is open and clears it on back", async () => {
    const user = userEvent.setup();
    const saved = createLensDemoData().lenses[0];
    network.mockImplementation(async (input) => {
      const path = requestPath(input);
      if (path === "/lens") return Response.json({ lenses: [saved], workers: [], tracing_enabled: true });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/lens/agents") return Response.json([]);
      if (path.startsWith("/lens/preview")) return Response.json({ eligible: 0, selected: 0, executions: [] });
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations&dialog=new",
    });
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    expect(await screen.findByRole("region", { name: "New investigation" })).toBeVisible();
    const tab = tabs.getByRole("tab", { name: /^Investigations/ });
    expect(tab).toHaveAttribute("aria-selected", "true");
    expect(within(tab).getByText("New")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Back to investigations" }));
    expect(await screen.findByRole("row", { name: new RegExp(saved.settings.name) })).toBeVisible();
    expect(within(tab).queryByText("New")).not.toBeInTheDocument();
  });

  it("adds a quiet Settings tab that manages the worker inline and reflects its health", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    const saved = createLensDemoData().lenses[0];
    const worker = {
      id: "worker",
      name: "Worker",
      revoked: false,
      analysis_key_id: "a".repeat(64),
      scope: saved.scope,
      last_seen: new Date().toISOString(),
    };
    const workers = vi.fn(() => [worker]);
    network.mockImplementation(async (input) => {
      const path = requestPath(input);
      if (path === "/lens") return Response.json({ lenses: [saved], workers: workers(), tracing_enabled: true });
      if (path.endsWith("/runs")) return Response.json(saved.jobs);
      if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, { onUrlUpdate });
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    const settings = await tabs.findByRole("tab", { name: "Settings" });
    expect(settings).toHaveAttribute("title", "Worker connected");
    await user.click(settings);
    await expectUrl(onUrlUpdate, (url) => expect(url.get("tab")).toBe("settings"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    const panel = within(screen.getByRole("region", { name: "Settings" }));
    expect(panel.getByRole("status")).toHaveTextContent("Tracing enabled");
    expect(panel.getByRole("heading", { name: "Analysis worker" })).toBeVisible();
    expect(panel.getByRole("heading", { name: worker.name })).toBeVisible();
    expect(panel.getByText("Connected")).toBeVisible();
    await user.click(panel.getByRole("button", { name: "Edit access" }));
    expect(panel.getByRole("heading", { name: "Analysis access" })).toBeVisible();
    await user.click(panel.getByRole("button", { name: "Cancel" }));
    expect(panel.getByRole("heading", { name: "Analysis worker" })).toBeVisible();
    workers.mockReturnValue([{ ...worker, revoked: true }]);
    await testQueryClient.refetchQueries({ queryKey: lensKeys.lists() });
    await waitFor(() => expect(settings).toHaveAttribute("title", "Connect worker"));
    expect(panel.getByRole("heading", { name: "Connect a worker" })).toBeVisible();
    expect(panel.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    await user.click(panel.getByRole("button", { name: "Connect an agent" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("tab")).toBe("traces"));
    expect(tabs.getByRole("tab", { name: "Traces" })).toHaveAttribute("aria-selected", "true");
  });

  it("sends the first-time guide's Connect worker into the Settings tab", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    network.mockImplementation(async (input) => {
      const path = requestPath(input);
      if (path === "/lens") return Response.json({ lenses: [], workers: [], tracing_enabled: true });
      if (path === "/v1/traces") return Response.json({ data: [{}] });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
      onUrlUpdate,
    });
    const guide = within(await screen.findByRole("region", { name: "Get Lens running" }));
    await user.click(await guide.findByRole("button", { name: "Connect worker" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("tab")).toBe("settings"));
    const panel = within(await screen.findByRole("region", { name: "Settings" }));
    expect(panel.getByRole("heading", { name: "Connect a worker" })).toBeVisible();
    expect(panel.getByRole("button", { name: "Get install command" })).toBeVisible();
  });

  it("keeps a pending worker install across tab switches and offers the first investigation once it connects", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    const token = "b".repeat(64);
    const worker = {
      id: "worker",
      name: "Lens worker",
      revoked: false,
      analysis_key_id: token,
      scope: { all_teams: true, api_key_hash: "", team_id: "" },
      last_seen: "1970-01-01T00:00:00Z",
    };
    const workers = vi.fn((): (typeof worker)[] => []);
    network.mockImplementation(async (input, init) => {
      const { path, method } = await readRequest(input, init);
      if (path === "/lens") return Response.json({ lenses: [], workers: workers(), tracing_enabled: true });
      if (path === "/lens/workers/register" && method === "POST") {
        workers.mockReturnValue([worker]);
        return Response.json({ token: "lens-test-token", image: "lens-worker:v1", worker });
      }
      if (path === "/key/list") return Response.json({ keys: [{ token, key_alias: "Analysis" }], total_pages: 1 });
      if (path === "/key/info") return Response.json({ info: { models: [], max_budget: null } });
      if (path === "/lens/agents") return Response.json([]);
      if (path.startsWith("/lens/preview")) return Response.json({ eligible: 0, selected: 0, executions: [] });
      if (path === "/v1/traces") return Response.json({ data: [createLensDemoData().runs[0].trace.summary] });
      return Response.json({ data: [], traces: true, requests: false });
    });
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=settings",
      onUrlUpdate,
    });
    const panel = within(await screen.findByRole("region", { name: "Settings" }));
    await user.click(panel.getByText("Advanced options"));
    await user.click(panel.getByRole("switch", { name: "Use an existing virtual key" }));
    await user.click(panel.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(panel.getByRole("button", { name: "Get install command" }));
    expect(await panel.findByText("Waiting for your worker to connect…")).toBeInTheDocument();
    const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
    await user.click(tabs.getByRole("tab", { name: "Traces" }));
    await waitFor(() => expect(panel.getByText("Waiting for your worker to connect…")).not.toBeVisible());
    await user.click(tabs.getByRole("tab", { name: "Settings" }));
    expect(panel.getByText("Waiting for your worker to connect…")).toBeVisible();
    expect(panel.getByLabelText("Docker command preview")).toHaveTextContent("LENS_WORKER_TOKEN=lens-test-token");
    workers.mockReturnValue([{ ...worker, last_seen: new Date().toISOString() }]);
    await testQueryClient.refetchQueries({ queryKey: lensKeys.lists() });
    expect(await panel.findByRole("heading", { name: "Worker connected" })).toBeVisible();
    await user.click(panel.getByRole("button", { name: "New investigation" }));
    await expectUrl(onUrlUpdate, (url) => expect(url.get("tab")).toBe("investigations"));
    expect(lastUrl(onUrlUpdate).get("dialog")).toBe("new");
    expect(await screen.findByRole("region", { name: "New investigation" })).toBeVisible();
  });

  it("hides the Settings tab for read-only sessions", async () => {
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly />);
    expect(await screen.findByRole("tablist", { name: "Lens" })).toBeVisible();
    await waitFor(() => expect(network).toHaveBeenCalled());
    expect(screen.queryByRole("tab", { name: "Settings" })).not.toBeInTheDocument();
  });

  it("sends read-only sessions following a Settings link to the default tab", async () => {
    renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly />, {
      searchParams: "?tab=settings",
    });
    expect(await screen.findByRole("tab", { name: "Traces", selected: true })).toBeVisible();
  });

  it("turns the worker health dot off once heartbeats expire even when polling returns unchanged data", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const saved = createLensDemoData().lenses[0];
      const worker = {
        id: "worker",
        name: "Worker",
        revoked: false,
        analysis_key_id: "a".repeat(64),
        scope: saved.scope,
        last_seen: new Date().toISOString(),
      };
      network.mockImplementation(async (input) => {
        const path = requestPath(input);
        if (path === "/lens") return Response.json({ lenses: [saved], workers: [worker], tracing_enabled: true });
        if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
        return Response.json({ data: [], traces: true, requests: false });
      });
      renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />);
      const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
      const settings = await tabs.findByRole("tab", { name: "Settings" });
      expect(settings).toHaveAttribute("title", "Worker connected");
      await vi.advanceTimersByTimeAsync(130000);
      await waitFor(() => expect(settings).toHaveAttribute("title", "Connect worker"));
    } finally {
      vi.useRealTimers();
    }
  });

  it("polls /lens every 2s while Settings waits for a worker, then drops to the 10s cadence once it connects", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      const saved = createLensDemoData().lenses[0];
      const worker = {
        id: "worker",
        name: "Worker",
        revoked: false,
        analysis_key_id: "a".repeat(64),
        scope: saved.scope,
        last_seen: new Date(Date.now() - 600_000).toISOString(),
      };
      const workers = vi.fn(() => [worker]);
      const listCalls = () => network.mock.calls.filter(([input]) => requestPath(input) === "/lens").length;
      network.mockImplementation(async (input) => {
        const path = requestPath(input);
        if (path === "/lens") return Response.json({ lenses: [saved], workers: workers(), tracing_enabled: true });
        if (path === "/v1/traces") return Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
        return Response.json({ data: [], traces: true, requests: false });
      });
      renderWithProviders(<LensWorkspace accessToken="live-token" userRole="Admin" readOnly={false} />, {
        searchParams: "?tab=settings",
      });
      expect(await screen.findByRole("region", { name: "Settings" })).toBeVisible();
      const initial = listCalls();
      await vi.advanceTimersByTimeAsync(2000);
      await waitFor(() => expect(listCalls()).toBe(initial + 1));
      workers.mockReturnValue([{ ...worker, last_seen: new Date().toISOString() }]);
      await vi.advanceTimersByTimeAsync(2000);
      await waitFor(() => expect(listCalls()).toBe(initial + 2));
      await vi.advanceTimersByTimeAsync(2000);
      expect(listCalls()).toBe(initial + 2);
      await vi.advanceTimersByTimeAsync(8000);
      await waitFor(() => expect(listCalls()).toBe(initial + 3));
    } finally {
      vi.useRealTimers();
    }
  });
});
