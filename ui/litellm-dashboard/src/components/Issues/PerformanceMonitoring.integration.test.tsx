import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, renderWithProviders, screen, testQueryClient } from "../../../tests/test-utils";
import PerformanceMonitoring from "./PerformanceMonitoring";

let state = { enabled: false, incidents_used: 0, incident_limit: 5 as number | null };
let failed = false;
const writes: unknown[] = [];
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

describe("Performance monitoring activation", () => {
  beforeEach(() => {
    testQueryClient.clear();
    state = { enabled: false, incidents_used: 0, incident_limit: 5 };
    failed = false;
    writes.length = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const path = new URL(request.url).pathname;
      if (path === "/management/v1/users/me") return json({ user_id: "example-admin", user_role: "proxy_admin" });
      if (path !== "/alerts/monitoring") return json({ detail: "Unexpected request" }, 404);
      if (failed) return json({ detail: "Unavailable" }, 503);
      if (request.method === "PATCH") {
        const body = await request.json();
        writes.push(body);
        state = { ...state, ...body };
      }
      return json(state);
    });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("shows remaining allowance when paused after detecting incidents", async () => {
    state = { enabled: false, incidents_used: 2, incident_limit: 5 };
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" />);
    expect(await screen.findByText(/3 included incidents remaining/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Enable performance alerts" })).toBeEnabled();
  });

  it("updates an open page when monitoring exhausts its allowance", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    state = { enabled: true, incidents_used: 4, incident_limit: 5 };
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" />);
    expect(await screen.findByText(/4 of 5 included incidents detected/)).toBeVisible();
    state = { enabled: true, incidents_used: 5, incident_limit: 5 };
    await act(() => vi.advanceTimersByTimeAsync(5 * 60_000));
    expect(await screen.findByText("Performance monitoring paused")).toBeVisible();
    expect(screen.getByRole("link", { name: "Upgrade to Enterprise" })).toBeVisible();
  });

  it("enables defaults in one action and reads back saved state", async () => {
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" />);
    fireEvent.click(await screen.findByRole("button", { name: "Enable performance alerts" }));
    expect(await screen.findByText("Performance alerts enabled")).toBeVisible();
    expect(writes).toEqual([{ enabled: true }]);
    expect(screen.getByText(/0 of 5 included incidents/)).toBeVisible();
  });

  it("keeps exhausted history accessible and does not offer a reset", async () => {
    state = { enabled: true, incidents_used: 5, incident_limit: 5 };
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" />);
    expect(await screen.findByText("Performance monitoring paused")).toBeVisible();
    expect(screen.getByText(/Saved alerts remain available/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Upgrade to Enterprise" })).toHaveAttribute(
      "href",
      "https://www.litellm.ai/enterprise",
    );
    expect(screen.queryByRole("button", { name: "Enable performance alerts" })).not.toBeInTheDocument();
    expect(writes).toEqual([]);
  });

  it("requires an admin without hiding model deprecations", async () => {
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" readOnly />);
    expect(await screen.findByText("An admin can enable monitoring.")).toBeVisible();
    expect(screen.getByText("Model-deprecation alerts are unaffected.")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Enable performance alerts" })).not.toBeInTheDocument();
  });

  it("shows a retryable error rather than treating failure as disabled", async () => {
    failed = true;
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load Performance monitoring.");
    expect(screen.queryByRole("button", { name: "Enable performance alerts" })).not.toBeInTheDocument();
    failed = false;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("button", { name: "Enable performance alerts" })).toBeVisible();
  });

  it("pauses Enterprise monitoring while retaining its allowance state", async () => {
    state = { enabled: true, incidents_used: 5, incident_limit: null };
    renderWithProviders(<PerformanceMonitoring accessToken="test-session" settings />);
    fireEvent.click(await screen.findByRole("button", { name: "Pause monitoring" }));
    expect(await screen.findByRole("button", { name: "Enable performance alerts" })).toBeVisible();
    expect(writes).toEqual([{ enabled: false }]);
    expect(state.incidents_used).toBe(5);
  });
});
