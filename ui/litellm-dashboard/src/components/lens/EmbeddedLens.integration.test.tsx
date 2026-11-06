import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { registerAuthTokenGetter } from "@/lib/http/runtime";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { EmbeddedLens } from "./EmbeddedLens";

const network = vi.fn<typeof fetch>();

beforeEach(() => {
  testQueryClient.clear();
  network.mockReset();
  window.localStorage.clear();
  window.sessionStorage.clear();
  registerAuthTokenGetter(() => "gateway-session");
  setGlobalLitellmHeaderName("X-Gateway-Key");
  switchToWorkerUrl("https://gateway.test/prefix");
  vi.stubGlobal("fetch", network);
});

afterEach(() => {
  switchToWorkerUrl(null);
  setGlobalLitellmHeaderName();
  vi.unstubAllGlobals();
});

it("renders the published Lens settings and uses the gateway session, prefix, and public contract", async () => {
  const user = userEvent.setup();
  network.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(String(input), init);
    expect(request.url).toMatch(/^https:\/\/gateway\.test\/prefix\/lens/);
    expect(request.headers.get("X-Gateway-Key")).toBe("Bearer gateway-session");
    expect(request.headers.get("X-Lens-Contract")).toBe("1");
    const path = new URL(request.url).pathname;
    if (path === "/prefix/lens") return Response.json({ lenses: [], workers: [], tracing_enabled: true });
    if (path === "/prefix/lens/signals") return Response.json({ model: "", threshold: 0.5, signals: [] });
    if (path === "/prefix/lens/model_group/info") return Response.json({ data: [] });
    if (path === "/prefix/lens/service")
      return Response.json({ configured: true, connected: true, status: { storage_ready: true } });
    throw new Error(`Unexpected Lens request ${path}`);
  });
  renderWithProviders(<EmbeddedLens accessToken="gateway-session" userRole="Admin" readOnly={false} />, {
    searchParams: "?tab=settings",
  });
  expect(await screen.findByRole("heading", { name: "Add an analysis provider" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Configure analysis models" })).toHaveAttribute(
    "href",
    "https://github.com/BerriAI/lens/blob/main/docs/analysis.md",
  );
  expect(screen.getByText("Tracing enabled")).toBeVisible();
  const analysis = within(screen.getByRole("region", { name: "Analysis", exact: true }));
  const copyPrompt = analysis.getByRole("button", { name: "Set it up for me" });
  await user.click(copyPrompt);
  const prompt = await navigator.clipboard.readText();
  expect(prompt).toContain("I opened Lens from an existing LiteLLM admin dashboard.");
  expect(prompt).toContain('LiteLLM API base: "https://gateway.test/prefix"');
  expect(prompt).toContain("Configure Lens investigations");
  expect(prompt).not.toContain("standalone Lens app");
  expect(prompt).not.toContain("gateway-session");
  expect(copyPrompt).toHaveTextContent("Prompt copied");
  expect(analysis.getByRole("button", { name: "Check configuration" })).toBeVisible();
  expect(network).toHaveBeenCalled();
});

it("opens the packaged demo trace and preserves its shareable selection in the gateway URL", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn();
  renderWithProviders(<EmbeddedLens accessToken="gateway-session" userRole="Admin" readOnly={false} />, {
    searchParams: "?demo=true",
    onUrlUpdate,
  });
  await user.click(await screen.findByText("Where is order #1042?"));
  const details = await screen.findByRole("complementary", { name: "Trace details" });
  expect(await within(details).findByRole("tab", { name: "Thread" })).toBeVisible();
  const query = new URLSearchParams(String(onUrlUpdate.mock.lastCall?.[0].queryString));
  expect(query.get("demo")).toBe("true");
  expect(query.get("trace")).toBeTruthy();
  expect(network).not.toHaveBeenCalled();
});

it("opens spend-log traces without asking to install a Lens service", async () => {
  network.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(String(input), init);
    const path = new URL(request.url).pathname;
    if (path === "/prefix/lens/service")
      return Response.json({ configured: false, connected: false, status: { storage_ready: false } });
    if (path === "/prefix/v1/traces") return Response.json({ data: [], next_cursor: null });
    if (path === "/prefix/v1/traces/agents") return Response.json({ agents: [] });
    throw new Error(`Unexpected Lens request ${path}`);
  });
  renderWithProviders(<EmbeddedLens accessToken="gateway-session" userRole="Admin" readOnly={false} />);
  expect(await screen.findByRole("heading", { name: "Your agent runs will appear here" })).toBeVisible();
  expect(screen.getByText(/x-litellm-agent-name/)).toBeVisible();
  expect(screen.queryByRole("region", { name: "Get Lens running" })).not.toBeInTheDocument();
  const tabs = within(screen.getByRole("tablist", { name: "Lens" }));
  expect(tabs.getAllByRole("tab").map((tab) => tab.textContent)).toEqual(["Agents", "Traces"]);
});
