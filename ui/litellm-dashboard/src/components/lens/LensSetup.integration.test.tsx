import { act, fireEvent, screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { readRequest, requestPath } from "@/../tests/lens-test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { createLensDemoData } from "./data/demo/fixtures";
import type { LensList } from "./model/types";

const network = vi.fn<typeof fetch>();
const list = vi.fn<() => Promise<LensList>>();
const data = createLensDemoData();
const worker = () => ({
  id: "setup-worker",
  analysis_key_id: "a".repeat(64),
  revoked: false,
  last_seen: new Date().toISOString(),
  scope: data.lenses[0].scope,
});

function serve({ enabled = false, traces = false, requests = false, connected = false } = {}) {
  list.mockResolvedValue({ lenses: [], workers: connected ? [worker()] : [], tracing_enabled: enabled });
  network.mockImplementation(async (input, init) => {
    const { path, method, body, query } = await readRequest(input, init);
    if (path === "/v1/traces")
      return enabled
        ? Response.json({ data: traces ? [data.runs[0].trace.summary] : [] })
        : Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
    if (path === "/lens/activity/available") return Response.json({ traces, requests });
    if (path === "/lens" && method === "POST") {
      const saved = { ...data.lenses[0], settings: { ...data.lenses[0].settings, ...(body as object) } };
      list.mockResolvedValue({ lenses: [saved], workers: [worker()], tracing_enabled: true });
      return Response.json(saved);
    }
    if (path === "/lens") return Response.json(await list());
    if (path === "/key/generate") return Response.json({ token_id: worker().analysis_key_id });
    if (path === "/lens/workers/register") {
      list.mockResolvedValue({ lenses: [], workers: [worker()], tracing_enabled: true });
      return Response.json({ worker: worker(), token: "test-worker-token", image: "test-worker-image" });
    }
    if (path === "/models") return Response.json({ data: [{ id: "analysis" }] });
    if (path === "/model_group/info")
      return Response.json({ data: [{ model_group: "analysis", providers: ["OpenAI"], mode: "chat" }] });
    if (path === "/key/info") return Response.json({ info: { models: ["analysis"], max_budget: 100 } });
    if (path === "/lens/agents") return Response.json(["support_agent"]);
    if (path === "/lens/preview/sample") return Response.json({ eligible: 1, selected: 1, executions: [] });
    if (path.endsWith("/reviews"))
      return Response.json({
        reviews: data.lenses[0].jobs[0].reviews.slice(Number(query.get("after") ?? 0)),
        reviewed: data.lenses[0].jobs[0].reviewed,
      });
    if (path.endsWith("/runs")) return Response.json(data.lenses[0].jobs);
    return Response.json({ data: [] });
  });
}

beforeEach(() => {
  testQueryClient.clear();
  window.localStorage.clear();
  window.sessionStorage.clear();
  network.mockReset();
  list.mockReset();
  vi.stubGlobal("fetch", network);
  Element.prototype.scrollIntoView = vi.fn();
  serve();
});

const renderWorkspace = (options?: Parameters<typeof renderWithProviders>[1], userRole = "Admin") =>
  renderWithProviders(<LensWorkspace accessToken="setup-token" userRole={userRole} readOnly={false} />, options);

const setupParam = (onUrlUpdate: ReturnType<typeof vi.fn>) =>
  new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString ?? "")).get("setup");

async function connectWorkerFromSettings(user: ReturnType<typeof userEvent.setup>) {
  const settings = within(await screen.findByRole("region", { name: "Settings" }));
  expect(settings.getByRole("heading", { name: "Connect a worker" })).toBeVisible();
  await user.click(settings.getByRole("combobox", { name: "Analysis model" }));
  await user.click(await screen.findByRole("option", { name: "analysis" }));
  await user.click(settings.getByRole("button", { name: "Get install command" }));
  expect(await settings.findByRole("heading", { name: "Worker connected" })).toBeVisible();
  await user.click(settings.getByRole("button", { name: "New investigation" }));
  expect(await screen.findByRole("region", { name: "New investigation" })).toBeVisible();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
}

describe("Lens introduction", () => {
  it("replaces both empty tabs with the introduction inside the page, including after a reload", async () => {
    window.localStorage.setItem("lens.intro.dismissed", "true");
    window.sessionStorage.setItem("lens.intro.seen", "true");
    const user = userEvent.setup();
    const first = renderWorkspace();
    const intro = await screen.findByRole("region", { name: "Get started with Lens" });
    expect(
      within(screen.getByRole("tabpanel", { name: "Traces" })).getByRole("region", {
        name: "Get started with Lens",
      }),
    ).toBe(intro);
    expect(within(intro).getByRole("heading", { name: "Before you start" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Enable tracing" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(
      within(screen.getByRole("tabpanel", { name: "Investigations" })).getByRole("region", {
        name: "Get started with Lens",
      }),
    ).toBeVisible();
    first.unmount();
    renderWorkspace();
    expect(await screen.findByRole("region", { name: "Get started with Lens" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens explicit setup links in the page and clears setup when navigating to Settings", async () => {
    serve({ enabled: true, traces: true });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWorkspace({ onUrlUpdate, searchParams: "?setup=lens" });
    expect(await screen.findByRole("region", { name: "Get started with Lens" })).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Settings" }));
    expect(await screen.findByRole("region", { name: "Settings" })).toBeVisible();
    expect(screen.queryByRole("region", { name: "Get started with Lens" })).not.toBeInTheDocument();
    expect(screen.queryByRole("switch", { name: "Show the introduction on each new session" })).not.toBeInTheDocument();
    await waitFor(() => expect(setupParam(onUrlUpdate)).toBeNull());
  });

  it("never opens on its own inside the sample session", async () => {
    renderWorkspace({ searchParams: "?demo=true" });
    expect(await screen.findByText("Where is order #1042?")).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});

describe("Lens setup journey", () => {
  it.each(["/lens", "/lens/activity/available"])(
    "keeps recorded traces visible while %s is pending",
    async (pendingPath) => {
      serve({ enabled: true, traces: true });
      const normal = network.getMockImplementation()!;
      network.mockImplementation((input, init) =>
        requestPath(input) === pendingPath ? new Promise<Response>(() => {}) : normal(input, init),
      );
      renderWorkspace();
      expect(await screen.findByRole("table", { name: "Agent runs" })).toBeVisible();
    },
  );

  it.each(["/v1/traces", "/lens/activity/available"])(
    "opens a saved investigation while %s is pending",
    async (pendingPath) => {
      serve();
      list.mockResolvedValue({ lenses: data.lenses, workers: [worker()], tracing_enabled: false });
      const normal = network.getMockImplementation()!;
      network.mockImplementation((input, init) =>
        requestPath(input) === pendingPath ? new Promise<Response>(() => {}) : normal(input, init),
      );
      renderWorkspace({ searchParams: `?lens=${data.lenses[0].id}` });
      expect(await screen.findByRole("heading", { name: data.lenses[0].settings.name })).toBeVisible();
    },
  );

  it("walks a first visit from the introduction through tracing into the Settings tab and the first investigation", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWorkspace({ onUrlUpdate });
    const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
    await user.click(await intro.findByRole("button", { name: "Set up Lens" }));
    await waitFor(() => expect(setupParam(onUrlUpdate)).toBe("lens"));
    serve({ enabled: true });
    await user.click(intro.getByRole("button", { name: "Check setup" }));
    expect(await intro.findByText("Trace storage is connected")).toBeVisible();
    await user.click(intro.getByRole("button", { name: "Continue to your agent" }));
    expect(intro.getByRole("button", { name: "Check for traces" })).toBeVisible();
    serve({ enabled: true, traces: true });
    await user.click(intro.getByRole("button", { name: "Check for traces" }));
    expect(await intro.findByText(/Your first trace is ready/)).toBeVisible();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("table", { name: "Agent runs" })).not.toBeInTheDocument();
    await user.click(intro.getByRole("button", { name: "Continue to worker" }));
    await waitFor(() => expect(setupParam(onUrlUpdate)).toBeNull());
    await connectWorkerFromSettings(user);
  });

  it("resumes setup from the URL and leaves only when the user chooses traces", async () => {
    serve({ enabled: true, traces: true });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWorkspace({ searchParams: "?tab=investigations&setup=lens", onUrlUpdate });
    const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
    expect(await intro.findByRole("button", { name: "Connect worker" })).toBeVisible();
    expect(intro.getByRole("heading", { name: "Get Lens running" })).toBeVisible();
    await user.click(intro.getByRole("button", { name: "View traces" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(await screen.findByRole("table", { name: "Agent runs" })).toBeVisible();
    await waitFor(() => expect(setupParam(onUrlUpdate)).toBeNull());
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    const guide = within(await screen.findByRole("region", { name: "Get Lens running" }));
    expect(guide.getByRole("button", { name: /Connect a worker/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("allows request-only investigations without forcing agent instrumentation", async () => {
    serve({ requests: true });
    const user = userEvent.setup();
    const welcome = renderWorkspace({ searchParams: "?tab=investigations" });
    expect(await screen.findByRole("region", { name: "Get Lens running" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    welcome.unmount();
    renderWorkspace({ searchParams: "?tab=investigations&setup=lens" });
    const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
    expect(await intro.findByRole("heading", { name: "Before you start" })).toBeVisible();
    expect(intro.getByRole("button", { name: "Connect worker" })).toBeEnabled();
    await user.click(intro.getByRole("button", { name: /Enable tracing on the gateway/ }));
    expect(intro.getByRole("button", { name: "Continue with request logs" })).toBeEnabled();
    await user.click(intro.getByRole("button", { name: /Send your first trace/ }));
    await user.click(intro.getByRole("button", { name: "Continue with request logs" }));
    await connectWorkerFromSettings(user);
  });

  it("keeps setup recoverable when checking for a first trace fails", async () => {
    serve({ enabled: true });
    const user = userEvent.setup();
    renderWorkspace({ searchParams: "?setup=lens" });
    const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
    await intro.findByRole("button", { name: "Check for traces" });
    const normal = network.getMockImplementation()!;
    network.mockImplementation(async (input, init) => {
      const path = requestPath(input);
      if (path === "/v1/traces") return Response.json({ detail: "Trace storage unavailable" }, { status: 503 });
      return normal(input, init);
    });
    await user.click(intro.getByRole("button", { name: "Check for traces" }));
    expect(await intro.findByRole("alert")).toHaveTextContent("Could not check setup");
    expect(intro.queryByRole("button", { name: "Continue to worker" })).not.toBeInTheDocument();
    serve({ enabled: true, traces: true });
    await user.click(intro.getByRole("button", { name: "Retry" }));
    expect(await intro.findByRole("button", { name: "Continue to worker" })).toBeEnabled();
    expect(intro.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps request-only users in investigations when an activity refresh fails", async () => {
    serve({ requests: true, connected: true });
    const user = userEvent.setup();
    renderWorkspace({ searchParams: "?tab=investigations" });
    expect(await screen.findByRole("button", { name: "New investigation" })).toBeEnabled();
    const normal = network.getMockImplementation()!;
    network.mockImplementation((input, init) =>
      requestPath(input) === "/lens/activity/available"
        ? Promise.resolve(Response.json({ detail: "Activity unavailable" }, { status: 503 }))
        : normal(input, init),
    );
    await act(() => testQueryClient.refetchQueries());
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not check setup. Activity unavailable");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
    network.mockImplementation(normal);
    await user.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "New investigation" })).toBeEnabled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps administrator-only setup unavailable to trace viewers", async () => {
    serve({ enabled: true, traces: true });
    renderWorkspace({ searchParams: "?setup=lens" }, "Internal User");
    const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
    expect(await intro.findByText(/A gateway administrator can connect a worker/)).toBeVisible();
    expect(intro.getByRole("button", { name: "Connect worker" })).toBeDisabled();
    expect(network.mock.calls.some(([input]) => requestPath(input) === "/lens")).toBe(false);
  });

  it.each(["traces", "requests with trace errors", "requests with pending traces", "traces with activity errors"])(
    "finishes guided setup with %s and opens the saved investigation",
    async (scenario) => {
      const source = scenario.startsWith("requests") ? "requests" : "traces";
      const activity = { enabled: true, traces: source === "traces", requests: source === "requests", connected: true };
      serve(activity);
      const normal = network.getMockImplementation()!;
      const failingPath = scenario === "requests with trace errors" ? "/v1/traces" : "/lens/activity/available";
      network.mockImplementation((input, init) => {
        const path = requestPath(input);
        if (path === "/v1/traces" && scenario === "requests with pending traces")
          return new Promise<Response>(() => {});
        if (scenario.endsWith("errors") && path === failingPath)
          return Promise.resolve(Response.json({ detail: "Activity unavailable" }, { status: 503 }));
        return normal(input, init);
      });
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn();
      renderWorkspace({ searchParams: "?setup=lens", onUrlUpdate });
      const intro = within(await screen.findByRole("region", { name: "Get started with Lens" }));
      await user.click(await intro.findByRole("button", { name: "New investigation" }));
      const editor = within(await screen.findByRole("region", { name: "New investigation" }));
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      fireEvent.change(editor.getByRole("textbox", { name: "Investigation name" }), {
        target: { value: "My first review" },
      });
      await user.click(editor.getByRole("button", { name: "Continue" }));
      await user.click(editor.getByRole("button", { name: "Continue" }));
      await waitFor(() => expect(editor.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
      await user.click(editor.getByRole("button", { name: "Run and monitor" }));
      expect(await screen.findByRole("heading", { name: "My first review" })).toBeVisible();
      expect(
        within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Investigations" }),
      ).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(setupParam(onUrlUpdate)).toBeNull());
      const requests = await Promise.all(network.mock.calls.map(([input, init]) => readRequest(input, init)));
      const create = requests.find((request) => request.path === "/lens" && request.method === "POST");
      expect(create).toBeDefined();
      expect(create?.body).toEqual(expect.objectContaining({ name: "My first review", source }));
    },
  );
});
