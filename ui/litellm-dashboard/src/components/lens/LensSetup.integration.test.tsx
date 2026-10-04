import { act, fireEvent, screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { LensWorkspace } from "./LensWorkspace";
import { createLensDemoData } from "./demo/createLensDemo";
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
    const path = new URL(String(input), "http://localhost").pathname;
    if (path === "/v1/traces")
      return enabled
        ? Response.json({ data: traces ? [data.runs[0].trace.summary] : [] })
        : Response.json({ detail: "Tracing is not enabled" }, { status: 501 });
    if (path === "/lens/activity/available") return Response.json({ traces, requests });
    if (path === "/lens" && init?.method === "POST") {
      const saved = { ...data.lenses[0], settings: { ...data.lenses[0].settings, ...JSON.parse(String(init.body)) } };
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
    if (path.endsWith("/runs")) return Response.json(data.lenses[0].jobs);
    return Response.json({ data: [] });
  });
}

beforeEach(() => {
  testQueryClient.clear();
  network.mockReset();
  list.mockReset();
  vi.stubGlobal("fetch", network);
  Element.prototype.scrollIntoView = vi.fn();
  serve();
});

describe("Lens setup journey", () => {
  it.each(["/lens", "/lens/activity/available"])(
    "keeps recorded traces visible while %s is pending",
    async (pendingPath) => {
      serve({ enabled: true, traces: true });
      const normal = network.getMockImplementation()!;
      network.mockImplementation((input, init) =>
        new URL(String(input), "http://localhost").pathname === pendingPath
          ? new Promise<Response>(() => {})
          : normal(input, init),
      );
      renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />);
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
        new URL(String(input), "http://localhost").pathname === pendingPath
          ? new Promise<Response>(() => {})
          : normal(input, init),
      );
      renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
        searchParams: `?lens=${data.lenses[0].id}`,
      });
      expect(await screen.findByRole("heading", { name: data.lenses[0].settings.name })).toBeVisible();
    },
  );

  it("shares the introduction across tabs and stays in setup after the first trace", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, { onUrlUpdate });
    expect(await screen.findByRole("heading", { name: "Before you start" })).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(screen.getByRole("heading", { name: "Before you start" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Set up Lens" }));
    await waitFor(() => expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("setup")).toBe("lens"));
    serve({ enabled: true });
    await user.click(screen.getByRole("button", { name: "Check setup" }));
    expect(await screen.findByText("Trace storage is connected")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Continue to your agent" }));
    expect(screen.getByRole("button", { name: "Check for traces" })).toBeVisible();
    serve({ enabled: true, traces: true });
    await user.click(screen.getByRole("button", { name: "Check for traces" }));
    expect(await screen.findByText(/Your first trace is ready/)).toBeVisible();
    expect(screen.queryByRole("table", { name: "Agent runs" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Continue to worker" }));
    const connection = within(await screen.findByRole("dialog", { name: "Connect a worker" }));
    await user.click(connection.getByRole("combobox", { name: "Analysis model" }));
    await user.click(await screen.findByRole("option", { name: "analysis" }));
    await user.click(connection.getByRole("button", { name: "Get install command" }));
    const connected = within(await screen.findByRole("dialog", { name: "Worker connected" }));
    await user.click(connected.getByRole("button", { name: "New investigation" }));
    expect(await screen.findByRole("dialog", { name: "Which activity should we investigate?" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "Get Lens running", hidden: true })).toBeInTheDocument();
  });

  it("resumes setup after a reload and leaves only when the user chooses traces", async () => {
    serve({ enabled: true, traces: true });
    const user = userEvent.setup();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations&setup=lens",
    });
    expect(await screen.findByRole("button", { name: "Connect worker" })).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Traces" }));
    expect(screen.getByRole("heading", { name: "Get Lens running" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "View traces" }));
    expect(await screen.findByRole("table", { name: "Agent runs" })).toBeVisible();
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(await screen.findByRole("heading", { name: "Run your first investigation" })).toBeVisible();
    expect(screen.queryByRole("link", { name: "Set up traces" })).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Get Lens running" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Set up Lens" }));
    expect(await screen.findByRole("heading", { name: "Get Lens running" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeVisible();
  });

  it("allows request-only investigations without forcing agent instrumentation", async () => {
    serve({ requests: true });
    const user = userEvent.setup();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
    });
    expect(await screen.findByText("Request logs received")).toBeVisible();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();
    expect(screen.queryByRole("heading", { name: "Before you start" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Set up Lens" }));
    expect(await screen.findByRole("heading", { name: "Before you start" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: /Enable tracing on the gateway/ }));
    expect(screen.getByRole("button", { name: "Continue with request logs" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: /Send your first trace/ }));
    await user.click(screen.getByRole("button", { name: "Continue with request logs" }));
    const connection = within(await screen.findByRole("dialog", { name: "Connect a worker" }));
    await user.click(connection.getByRole("combobox", { name: "Analysis model" }));
    await user.click(await screen.findByRole("option", { name: "analysis" }));
    await user.click(connection.getByRole("button", { name: "Get install command" }));
    const connected = within(await screen.findByRole("dialog", { name: "Worker connected" }));
    await user.click(connected.getByRole("button", { name: "New investigation" }));
    expect(await screen.findByRole("dialog", { name: "Which activity should we investigate?" })).toBeVisible();
  });

  it("keeps setup recoverable when checking for a first trace fails", async () => {
    serve({ enabled: true });
    const user = userEvent.setup();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?setup=lens",
    });
    await screen.findByRole("button", { name: "Check for traces" });
    const normal = network.getMockImplementation()!;
    network.mockImplementation(async (input, init) => {
      const path = new URL(String(input), "http://localhost").pathname;
      if (path === "/v1/traces") return Response.json({ detail: "Trace storage unavailable" }, { status: 503 });
      return normal(input, init);
    });
    await user.click(screen.getByRole("button", { name: "Check for traces" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not check setup");
    expect(screen.queryByRole("button", { name: "Continue to worker" })).not.toBeInTheDocument();
    serve({ enabled: true, traces: true });
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Continue to worker" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps request-only users in investigations when an activity refresh fails", async () => {
    serve({ requests: true, connected: true });
    const user = userEvent.setup();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
      searchParams: "?tab=investigations",
    });
    expect(await screen.findByText("Request logs received")).toBeVisible();
    const normal = network.getMockImplementation()!;
    network.mockImplementation((input, init) =>
      new URL(String(input), "http://localhost").pathname === "/lens/activity/available"
        ? Promise.resolve(Response.json({ detail: "Activity unavailable" }, { status: 503 }))
        : normal(input, init),
    );
    await act(() => testQueryClient.refetchQueries());
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not check recorded activity");
    expect(screen.queryByRole("heading", { name: "Get Lens running" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New investigation" })).toBeDisabled();
    network.mockImplementation(normal);
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Request logs received")).toBeVisible();
    expect(screen.getByRole("button", { name: "New investigation" })).toBeEnabled();
  });

  it("keeps administrator-only setup unavailable to trace viewers", async () => {
    serve({ enabled: true, traces: true });
    const user = userEvent.setup();
    renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Internal User" readOnly={false} />, {
      searchParams: "?setup=lens",
    });
    expect(await screen.findByText(/A gateway administrator can connect a worker/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeDisabled();
    await user.click(screen.getByRole("tab", { name: "Investigations" }));
    expect(screen.getByRole("button", { name: "Connect worker" })).toBeDisabled();
    expect(network.mock.calls.some(([input]) => new URL(String(input), "http://localhost").pathname === "/lens")).toBe(
      false,
    );
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
        const path = new URL(String(input), "http://localhost").pathname;
        if (path === "/v1/traces" && scenario === "requests with pending traces")
          return new Promise<Response>(() => {});
        if (scenario.endsWith("errors") && path === failingPath)
          return Promise.resolve(Response.json({ detail: "Activity unavailable" }, { status: 503 }));
        return normal(input, init);
      });
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn();
      renderWithProviders(<LensWorkspace accessToken="setup-token" userRole="Admin" readOnly={false} />, {
        searchParams: "?setup=lens",
        onUrlUpdate,
      });
      await user.click(await screen.findByRole("button", { name: "New investigation" }));
      const dialog = within(screen.getByRole("dialog"));
      fireEvent.change(dialog.getByRole("textbox", { name: "Investigation name" }), {
        target: { value: "My first review" },
      });
      await user.click(dialog.getByRole("button", { name: "Continue" }));
      await user.click(dialog.getByRole("button", { name: "Continue" }));
      await waitFor(() => expect(dialog.getByRole("button", { name: "Run and monitor" })).toBeEnabled());
      await user.click(dialog.getByRole("button", { name: "Run and monitor" }));
      expect(await screen.findByRole("heading", { name: "My first review" })).toBeVisible();
      expect(screen.queryByRole("heading", { name: "Get Lens running" })).not.toBeInTheDocument();
      expect(
        within(screen.getByRole("tablist", { name: "Lens" })).getByRole("tab", { name: "Investigations" }),
      ).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("setup")).toBeNull());
      const create = network.mock.calls.find(
        ([input, init]) => new URL(String(input), "http://localhost").pathname === "/lens" && init?.method === "POST",
      );
      expect(create).toBeDefined();
      expect(JSON.parse(String(create?.[1]?.body))).toEqual(
        expect.objectContaining({ name: "My first review", source }),
      );
    },
  );
});
