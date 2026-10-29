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
      return Response.json({ configured: false, connected: false, status: { storage_ready: false } });
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

it("keeps manual installation on the unconfigured gateway landing and copies its observed setup context", async () => {
  const user = userEvent.setup();
  network.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(String(input), init);
    const path = new URL(request.url).pathname;
    if (path === "/prefix/lens/service") {
      const connection = { url: "", configured: false, connected: false, status: { storage_ready: false } };
      return Response.json(connection);
    }
    if (path === "/prefix/lens")
      return Response.json(
        { detail: "Configure LITELLM_LENS_URL and LENS_GATEWAY_SECRET for the Lens service" },
        { status: 503 },
      );
    if (path === "/prefix/v1/traces" || path === "/prefix/v1/traces/agents")
      return Response.json(
        { detail: "Agent tracing is not enabled. Configure the Lens service and LITELLM_LENS_URL." },
        { status: 501 },
      );
    throw new Error(`Unexpected Lens request ${path}`);
  });
  renderWithProviders(<EmbeddedLens accessToken="gateway-session" userRole="Admin" readOnly={false} />);

  const setup = within(await screen.findByRole("region", { name: "Get Lens running" }));
  expect(setup.getByRole("button", { name: /Install Lens/ })).toHaveAttribute("aria-expanded", "true");
  expect(await setup.findByRole("link", { name: "Helm setup" })).toHaveAttribute(
    "href",
    "https://docs.litellm.ai/docs/proxy/lens/deployment/kubernetes#existing-deployment",
  );
  expect(setup.getByRole("link", { name: "Docker setup" })).toHaveAttribute(
    "href",
    "https://docs.litellm.ai/docs/proxy/lens/deployment/docker-compose",
  );
  expect(setup.getByRole("button", { name: "Check setup" })).toBeEnabled();

  await user.click(setup.getByRole("button", { name: "Set it up for me" }));
  const prompt = await navigator.clipboard.readText();
  expect(prompt).toContain("existing LiteLLM admin dashboard");
  expect(prompt).toContain('LiteLLM API base: "https://gateway.test/prefix"');
  expect(prompt).toContain("Lens connection is not configured");
  expect(prompt).toContain("Check for an existing Lens service before installing another");
  expect(prompt).not.toContain("gateway-session");
});
