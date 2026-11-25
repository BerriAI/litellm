import { act, fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, useTheme } from "next-themes";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { EmbeddedLens } from "./EmbeddedLens";
import { GatewayLensNavigation, GatewayLensNavigationProvider } from "./GatewayLensNavigation";
import { SidebarProvider } from "@/components/ui/sidebar";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { setServerRootPath } from "@/lib/serverRootPath";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";

const network = vi.fn<typeof fetch>();
const accessToken = "sk-dashboard-session";
const token = `header.${btoa(JSON.stringify({ key: "sk-dashboard-session" }))}.signature`;
const connection = (configured = true, connected = true, browserUrl = "https://lens.example.test") =>
  Response.json({ configured, connected, browser_url: browserUrl });
const ui = () => new Response("<html></html>", { headers: { "X-Lens-UI-Auth": "bearer-v1" } });
const ready = () => network.mockResolvedValueOnce(ui()).mockResolvedValueOnce(connection());

beforeEach(() => {
  network.mockReset();
  vi.stubGlobal("fetch", network);
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
  clearTokenCookies();
  window.sessionStorage.setItem("token", token);
});

afterEach(() => {
  vi.restoreAllMocks();
  clearTokenCookies();
  switchToWorkerUrl(null);
  setGlobalLitellmHeaderName("Authorization");
  setServerRootPath("/");
  vi.unstubAllGlobals();
});

const render = (key = accessToken, headerName = "Authorization") =>
  renderWithProviders(<EmbeddedLens accessToken={key} headerName={headerName} />);

it("frames the connected Lens at the tab in the dashboard URL", async () => {
  ready();
  render();
  expect(await screen.findByTitle("Lens")).toHaveAttribute(
    "src",
    `${window.location.origin}/lens/ui/?tab=traces&gateway-auth=bearer-v1`,
  );
});

it("checks the public capability before service status using the supplied dashboard bearer", async () => {
  const otherToken = `header.${btoa(JSON.stringify({ key: "sk-other-session" }))}.signature`;
  document.cookie = `token=${otherToken}; Path=/`;
  setGlobalLitellmHeaderName("X-Gateway-Key");
  const requests: Request[] = [];
  network.mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    requests.push(request);
    if (new URL(request.url).pathname === "/lens/service") return connection();
    return ui();
  });
  render(accessToken, "X-Gateway-Key");

  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(requests.map((request) => new URL(request.url).pathname)).toEqual(["/lens/ui/", "/lens/service"]);
  expect(requests.every((request) => request.method === "GET")).toBe(true);
  expect(requests[0].headers.get("X-Gateway-Key")).toBeNull();
  expect(requests[0].headers.get("Authorization")).toBeNull();
  expect(requests[1].headers.get("X-Gateway-Key")).toBe(`Bearer ${accessToken}`);
  expect(requests[1].headers.get("Authorization")).toBeNull();
  expect(getCookie("token")).toBe(otherToken);
});

it("renders the connected frame's Lens navigation in the gateway sidebar and keeps the theme in sync", async () => {
  ready();
  const user = userEvent.setup();
  function ToggleTheme() {
    const { setTheme } = useTheme();
    return <button onClick={() => setTheme("dark")}>Dark</button>;
  }
  const { unmount } = renderWithProviders(
    <ThemeProvider defaultTheme="light" storageKey="gateway-lens-test">
      <ToggleTheme />
      <SidebarProvider>
        <GatewayLensNavigationProvider>
          <aside aria-label="Gateway navigation">
            <GatewayLensNavigation collapsed={false} />
          </aside>
          <EmbeddedLens accessToken={accessToken} headerName="Authorization" />
        </GatewayLensNavigationProvider>
      </SidebarProvider>
    </ThemeProvider>,
  );
  const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
  const child = iframe.contentWindow!;
  const postMessage = vi.spyOn(child, "postMessage").mockImplementation(() => {});
  const message = {
    type: "navigation",
    navigation: {
      activeTab: "traces",
      tabs: [
        { value: "traces", label: "Traces" },
        { value: "findings", label: "Findings" },
      ],
      currentAgent: "support_agent",
      agents: [
        { value: "support_agent", label: "support_agent" },
        { value: "release_agent", label: "release_agent" },
      ],
      agentLabel: "Agent",
      bookCallLabel: "Book a call",
    },
  };
  const send = (data: object, source: Window = child, origin = window.location.origin) => {
    act(() => {
      window.dispatchEvent(
        new MessageEvent("message", { source, origin, data: { channel: "litellm-lens:v1", ...data } }),
      );
    });
  };
  const sidebar = within(screen.getByRole("complementary", { name: "Gateway navigation" }));
  try {
    send({ type: "ready" });
    expect(postMessage).toHaveBeenCalledWith(
      { channel: "litellm-lens:v1", command: { type: "theme", theme: "light" } },
      window.location.origin,
    );
    send(message, window);
    send(message, child, "https://other.example");
    expect(sidebar.queryByRole("button", { name: "Findings" })).not.toBeInTheDocument();

    send(message);
    expect(sidebar.getByRole("button", { name: "Traces" })).toHaveAttribute("aria-current", "page");
    expect(sidebar.getByRole("link", { name: "Lens documentation" })).toBeVisible();
    await user.click(sidebar.getByRole("button", { name: "Findings" }));
    expect(postMessage).toHaveBeenLastCalledWith(
      { channel: "litellm-lens:v1", command: { type: "tab", value: "findings" } },
      window.location.origin,
    );
    const agent = sidebar.getByRole("combobox", { name: "Agent: support_agent" });
    await user.click(agent);
    await user.type(await screen.findByRole("combobox", { name: "Agent" }), "release");
    await user.keyboard("{ArrowDown}");
    await user.click(await screen.findByRole("option", { name: "release_agent" }));
    expect(postMessage).toHaveBeenLastCalledWith(
      { channel: "litellm-lens:v1", command: { type: "agent", value: "release_agent" } },
      window.location.origin,
    );
    await user.click(screen.getByRole("button", { name: "Dark" }));
    expect(postMessage).toHaveBeenLastCalledWith(
      { channel: "litellm-lens:v1", command: { type: "theme", theme: "dark" } },
      window.location.origin,
    );
  } finally {
    postMessage.mockRestore();
    unmount();
    window.localStorage.removeItem("gateway-lens-test");
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.removeProperty("color-scheme");
  }
});

it("shows the original introduction and linked sample evidence when Lens is not configured", async () => {
  network.mockResolvedValueOnce(new Response(null, { status: 404 })).mockResolvedValueOnce(connection(false, false));
  render();
  expect(await screen.findByRole("heading", { name: "The gateway that helps your agents improve" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Set up Lens" })).toHaveAttribute(
    "href",
    "https://docs.litellm.ai/docs/proxy/lens",
  );
  expect(screen.getByRole("list", { name: "Sample trace timeline" })).toBeVisible();
  const finding = screen.getByRole("button", { name: "Claims numbers it never measured" });
  fireEvent.click(finding);
  expect(finding).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Update Lens to use it here" })).not.toBeInTheDocument();
});

it("opens the workspace after setup succeeds and the user checks the connection again", async () => {
  network
    .mockResolvedValueOnce(new Response(null, { status: 404 }))
    .mockResolvedValueOnce(connection(false, false))
    .mockResolvedValueOnce(ui())
    .mockResolvedValueOnce(connection());
  render();
  fireEvent.click(await screen.findByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(screen.queryByRole("heading", { name: "The gateway that helps your agents improve" })).not.toBeInTheDocument();
});

it("offers the standalone launch only for a connected older Lens with a browser address", async () => {
  network.mockResolvedValueOnce(new Response("", { status: 404 })).mockResolvedValueOnce(connection());
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
    "href",
    `${window.location.origin}/lens/login`,
  );
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
  expect(network).toHaveBeenCalledTimes(2);
});

it("does not offer the localhost launch when the gateway has no valid browser address", async () => {
  network.mockResolvedValueOnce(new Response("", { status: 404 })).mockResolvedValueOnce(connection(true, true, ""));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  expect(screen.getByText(/configure a Lens address that is accessible from your browser/)).toBeVisible();
});

it("shows a connection problem rather than installation or upgrade for a configured disconnected service", async () => {
  network.mockResolvedValueOnce(new Response(null, { status: 404 })).mockResolvedValueOnce(connection(true, false));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
});

it.each([401, 403, 404, 503])("does not mistake a failed connection check (%s) for missing setup", async (status) => {
  network
    .mockResolvedValueOnce(new Response(null, { status: 404 }))
    .mockResolvedValueOnce(Response.json({ detail: "Unavailable" }, { status }));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
});

it.each(["UI probe", "service check"])("can retry a failed %s without reloading the dashboard", async (step) => {
  if (step === "service check") network.mockResolvedValueOnce(ui());
  network
    .mockRejectedValueOnce(new TypeError("network"))
    .mockResolvedValueOnce(ui())
    .mockResolvedValueOnce(connection());
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
});

it("sends the access key only to the ready iframe and leaves browser credentials unchanged", async () => {
  window.history.replaceState(null, "", "/ui/lens/?tab=traces&gateway-auth=cookie&gateway-auth=other");
  const bridgeKey = "sk-bridge-fixture";
  const cookieWrites = vi.spyOn(document, "cookie", "set");
  const reload = vi.fn();
  vi.stubGlobal("location", { ...window.location, reload });
  ready();
  render(bridgeKey, "X-Gateway-Key");
  const iframe = await screen.findByTitle<HTMLIFrameElement>("Lens");
  const child = iframe.contentWindow!;
  const postMessage = vi.spyOn(child, "postMessage").mockImplementation(() => {});
  const send = (source: Window, origin: string, channel = "litellm-lens:v1") => {
    act(() => {
      window.dispatchEvent(new MessageEvent("message", { source, origin, data: { channel, type: "auth-ready" } }));
    });
  };

  send(window, window.location.origin);
  send(child, "https://other.example.test");
  send(child, window.location.origin, "other-channel");
  expect(postMessage).not.toHaveBeenCalled();
  send(child, window.location.origin);
  expect(postMessage).toHaveBeenCalledExactlyOnceWith(
    { channel: "litellm-lens:v1", command: { type: "auth", accessToken: bridgeKey, headerName: "X-Gateway-Key" } },
    window.location.origin,
  );
  const frameUrl = new URL(iframe.src);
  expect(frameUrl.searchParams.get("tab")).toBe("traces");
  expect(frameUrl.searchParams.getAll("gateway-auth")).toEqual(["bearer-v1"]);
  expect(iframe.src).not.toContain(bridgeKey);
  expect(network.mock.calls.map(([input, init]) => new Request(input, init).url).join(" ")).not.toContain(bridgeKey);
  expect(cookieWrites).not.toHaveBeenCalled();
  expect(sessionStorage.getItem("token")).toBe(token);
  for (const storage of [sessionStorage, localStorage]) {
    const values = Array.from({ length: storage.length }, (_, index) => storage.getItem(storage.key(index) ?? ""));
    expect(values.join(" ")).not.toContain(bridgeKey);
  }
  expect(reload).not.toHaveBeenCalled();
});

it.each([
  { accessToken: "sk-replacement-session", headerName: "Authorization" },
  { accessToken, headerName: "X-Alternate-Key" },
])("immediately replaces the iframe when its credentials change to %j", async (credentials) => {
  ready();
  const { rerender } = render();
  const previousFrame = await screen.findByTitle<HTMLIFrameElement>("Lens");
  const previousChild = previousFrame.contentWindow!;
  ready();
  setGlobalLitellmHeaderName(credentials.headerName);
  rerender(<EmbeddedLens {...credentials} />);

  expect(previousFrame).not.toBeInTheDocument();
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
  const currentFrame = await screen.findByTitle<HTMLIFrameElement>("Lens");
  expect(currentFrame).not.toBe(previousFrame);
  const currentChild = currentFrame.contentWindow!;
  const postMessage = vi.spyOn(currentChild, "postMessage").mockImplementation(() => {});
  const message = { channel: "litellm-lens:v1", type: "auth-ready" };
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", { source: previousChild, origin: window.location.origin, data: message }),
    );
  });
  expect(postMessage).not.toHaveBeenCalled();
  act(() => {
    window.dispatchEvent(
      new MessageEvent("message", { source: currentChild, origin: window.location.origin, data: message }),
    );
  });
  expect(postMessage).toHaveBeenCalledExactlyOnceWith(
    { channel: "litellm-lens:v1", command: { type: "auth", ...credentials } },
    window.location.origin,
  );
  const [input, init] = network.mock.calls[3];
  expect(new Request(input, init).headers.get(credentials.headerName)).toBe(`Bearer ${credentials.accessToken}`);
});

it("bounds a stalled capability probe and opens Lens after retry", async () => {
  network.mockImplementationOnce(
    (_input, options) =>
      new Promise<Response>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), {
          once: true,
        });
      }),
  );
  render();

  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" }, { timeout: 7000 })).toBeVisible();
  expect(network).toHaveBeenCalledTimes(1);
  expect(network.mock.calls[0][1]?.signal?.aborted).toBe(true);
  ready();
  fireEvent.click(screen.getByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(network).toHaveBeenCalledTimes(3);
});

it("cancels the capability probe on unmount without starting protected requests", async () => {
  network.mockImplementationOnce(
    (_input, options) =>
      new Promise<Response>((_resolve, reject) => {
        options?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), {
          once: true,
        });
      }),
  );
  const { unmount } = render();
  const signal = network.mock.calls[0][1]?.signal;
  expect(signal?.aborted).toBe(false);
  unmount();
  expect(signal?.aborted).toBe(true);
  await act(async () => {});
  expect(network).toHaveBeenCalledTimes(1);
});

it.each([null, "cookie-v1", "bearer-v2"])(
  "offers standalone Lens when the UI lacks the supported bearer capability (%s)",
  async (capability) => {
    const headers = capability ? { "X-Lens-UI-Auth": capability } : undefined;
    network.mockResolvedValueOnce(new Response("<html></html>", { headers })).mockResolvedValueOnce(connection());
    render();

    expect(await screen.findByRole("link", { name: "Open Lens" })).toHaveAttribute(
      "href",
      `${window.location.origin}/lens/login`,
    );
    expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
    expect(network).toHaveBeenCalledTimes(2);
  },
);

it("does not frame a capability response redirected to another origin", async () => {
  const response = ui();
  Object.defineProperty(response, "url", { value: "https://other.example.test/lens/ui/" });
  network.mockResolvedValueOnce(response).mockResolvedValueOnce(connection());
  render();

  expect(await screen.findByRole("link", { name: "Open Lens" })).toBeVisible();
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
  expect(network).toHaveBeenCalledTimes(2);
});

it.each([404, 200])(
  "preserves the split-origin standalone fallback without handing off credentials (UI %s)",
  async (status) => {
    switchToWorkerUrl("https://api.example.test");
    network
      .mockResolvedValueOnce(new Response(null, { status, headers: { "X-Lens-UI-Auth": "bearer-v1" } }))
      .mockResolvedValueOnce(connection());
    render();

    expect(await screen.findByRole("link", { name: "Open Lens" })).toHaveAttribute(
      "href",
      "https://api.example.test/lens/login",
    );
    expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
    const requests = network.mock.calls.map(([input, init]) => new Request(input, init));
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual(["/lens/ui/", "/lens/service"]);
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(requests[0].headers.get("Authorization")).toBeNull();
    expect(getCookie("token")).toBe(token);
  },
);

it("uses the deployment prefix once for the UI probe, service check and iframe", async () => {
  setServerRootPath("/gateway");
  switchToWorkerUrl(`${window.location.origin}/gateway`);
  ready();
  render();

  expect(await screen.findByTitle("Lens")).toHaveAttribute(
    "src",
    `${window.location.origin}/gateway/lens/ui/?tab=traces&gateway-auth=bearer-v1`,
  );
  expect(
    network.mock.calls.map(([input]) => new URL(input instanceof Request ? input.url : String(input)).pathname),
  ).toEqual(["/gateway/lens/ui/", "/gateway/lens/service"]);
});
