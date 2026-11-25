import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, useTheme } from "next-themes";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { EmbeddedLens } from "./EmbeddedLens";
import { GatewayLensNavigation, GatewayLensNavigationProvider } from "./GatewayLensNavigation";
import { SidebarProvider } from "@/components/ui/sidebar";
import { switchToWorkerUrl } from "@/components/networking";
import { setServerRootPath } from "@/lib/serverRootPath";
import { clearTokenCookies, getCookie } from "@/utils/cookieUtils";

const network = vi.fn<typeof fetch>();
const token = `header.${btoa(JSON.stringify({ key: "sk-dashboard-session" }))}.signature`;
const connection = (configured = true, connected = true, browserUrl = "https://lens.example.test") =>
  Response.json({ configured, connected, browser_url: browserUrl });
const ready = () =>
  network
    .mockResolvedValueOnce(new Response("<html></html>"))
    .mockResolvedValueOnce(Response.json({ token }))
    .mockResolvedValueOnce(connection());

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
  setServerRootPath("/");
  vi.unstubAllGlobals();
});

const render = (dashboardToken: string | null = token) =>
  renderWithProviders(<EmbeddedLens dashboardToken={dashboardToken} />);

it("frames the connected Lens at the tab in the dashboard URL", async () => {
  ready();
  render();
  expect(await screen.findByTitle("Lens")).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
});

it("restores the dashboard's storage fallback before opening Lens", async () => {
  window.sessionStorage.setItem("token", token);
  const requests: Request[] = [];
  network.mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    requests.push(request);
    if (new URL(request.url).pathname === "/lens/service") return connection();
    if (new URL(request.url).pathname === "/lens/session") return Response.json({ token });
    return new Response("<html></html>");
  });
  try {
    render();
    expect(await screen.findByTitle("Lens")).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual([
      "/lens/ui/",
      "/lens/session",
      "/lens/service",
    ]);
    expect(requests[0].headers.get("Authorization")).toBeNull();
    expect(requests[2].headers.get("Authorization")).toBe("Bearer sk-dashboard-session");
    expect(requests[1].method).toBe("POST");
    expect(requests[1].credentials).toBe("include");
    expect(requests[1].redirect).toBe("error");
    expect(requests[1].headers.get("Authorization")).toBeNull();
    expect(await requests[1].json()).toEqual({ token });
  } finally {
    window.sessionStorage.removeItem("token");
  }
});

it.each([401, 403, 503])("does not open Lens when the dashboard session handoff fails (%s)", async (status) => {
  window.sessionStorage.setItem("token", "dashboard-session-fixture");
  network
    .mockResolvedValueOnce(new Response("<html></html>"))
    .mockResolvedValueOnce(Response.json({ detail: "Sign in again" }, { status }));
  try {
    render();
    expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
    expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
    expect(network).toHaveBeenCalledTimes(2);
  } finally {
    window.sessionStorage.removeItem("token");
  }
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
          <EmbeddedLens dashboardToken={token} />
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
    .mockResolvedValueOnce(new Response("<html></html>"))
    .mockResolvedValueOnce(Response.json({ token }))
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

it.each(["UI probe", "session handoff"])("can retry a failed %s without reloading the dashboard", async (step) => {
  if (step === "session handoff") network.mockResolvedValueOnce(new Response("<html></html>"));
  network
    .mockRejectedValueOnce(new TypeError("network"))
    .mockResolvedValueOnce(new Response("<html></html>"))
    .mockResolvedValueOnce(Response.json({ token }))
    .mockResolvedValueOnce(connection());
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
});

it.each(["stale", "canonical"])(
  "reloads once when the root login differs from the mounted dashboard with a %s readable cookie",
  async (readableCookie) => {
    const canonical = `header.${btoa(JSON.stringify({ key: "sk-current-session" }))}.signature`;
    const cookieToken = readableCookie === "stale" ? token : canonical;
    document.cookie = `token=${canonical}; Path=/`;
    document.cookie = `token=${cookieToken}; Path=/ui`;
    expect(getCookie("token")).toBe(cookieToken);
    const cookieWrites = vi.spyOn(document, "cookie", "set");
    const reload = vi.fn();
    vi.stubGlobal("location", { ...window.location, reload });
    const requests: Request[] = [];
    network.mockImplementation(async (input, init) => {
      const request = new Request(input, init);
      requests.push(request);
      const path = new URL(request.url).pathname;
      if (path === "/lens/session") return Response.json({ token: canonical });
      if (path === "/lens/service") return connection();
      return new Response("<html></html>");
    });
    const firstDocument = render(token);

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual(["/lens/ui/", "/lens/session"]);
    expect(await requests[1].json()).toEqual({ token });
    expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
    expect(getCookie("token")).toBe(canonical);
    expect(sessionStorage.getItem("token")).toBe(canonical);
    expect(cookieWrites).toHaveBeenCalledTimes(1);
    expect(cookieWrites).toHaveBeenCalledWith(expect.stringContaining("path=/ui;"));

    firstDocument.unmount();
    render(canonical);

    expect(await screen.findByTitle("Lens")).toBeVisible();
    expect(requests.at(-1)?.headers.get("Authorization")).toBe("Bearer sk-current-session");
    expect(reload).toHaveBeenCalledTimes(1);
  },
);

it("repairs stale client credentials without reloading when the mounted dashboard already has the canonical login", async () => {
  document.cookie = "token=stale-session; Path=/ui";
  const reload = vi.fn();
  vi.stubGlobal("location", { ...window.location, reload });
  ready();
  render(token);

  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(network.mock.calls[1][1]?.body).toBe(JSON.stringify({ token }));
  const [input, init] = network.mock.calls[2];
  expect(new Request(input, init).headers.get("Authorization")).toBe("Bearer sk-dashboard-session");
  expect(getCookie("token")).toBe(token);
  expect(reload).not.toHaveBeenCalled();
});

it("hands off a null token when the root session is not readable and persists the returned login", async () => {
  clearTokenCookies();
  const reload = vi.fn();
  vi.stubGlobal("location", { ...window.location, reload });
  network.mockResolvedValueOnce(new Response("<html></html>")).mockResolvedValueOnce(Response.json({ token }));
  render(null);

  await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
  expect(network.mock.calls[1][1]?.body).toBe(JSON.stringify({ token: null }));
  expect(network.mock.calls[1][1]?.credentials).toBe("include");
  expect(getCookie("token")).toBe(token);
  expect(network).toHaveBeenCalledTimes(2);
});

it("shows a retryable error without reloading when neither cookie nor storage can retain the selected login", async () => {
  const reload = vi.fn();
  vi.stubGlobal("location", { ...window.location, reload });
  vi.spyOn(document, "cookie", "set").mockImplementation(() => {});
  vi.stubGlobal("sessionStorage", {
    getItem: (key: string) => (key === "token" ? token : null),
    setItem: () => {
      throw new DOMException("Storage denied", "SecurityError");
    },
    removeItem: () => {},
  });
  network
    .mockResolvedValueOnce(new Response("<html></html>"))
    .mockResolvedValueOnce(Response.json({ token: "canonical-session" }));
  render();

  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Check connection" })).toBeVisible();
  expect(reload).not.toHaveBeenCalled();
  expect(getCookie("token")).toBe(token);
  expect(network).toHaveBeenCalledTimes(2);
});

it.each([{}, { token: "" }, { token: 42 }])("refuses malformed canonical session data %j", async (body) => {
  network.mockResolvedValueOnce(new Response("<html></html>")).mockResolvedValueOnce(Response.json(body));
  render();

  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(getCookie("token")).toBe(token);
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
  expect(network).toHaveBeenCalledTimes(2);
});

it.each([404, 200])(
  "preserves the split-origin standalone fallback without handing off credentials (UI %s)",
  async (status) => {
    switchToWorkerUrl("https://api.example.test");
    network.mockResolvedValueOnce(new Response(null, { status })).mockResolvedValueOnce(connection());
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

it("uses the deployment prefix once for the UI probe, handoff and iframe", async () => {
  setServerRootPath("/gateway");
  switchToWorkerUrl(`${window.location.origin}/gateway`);
  ready();
  render();

  expect(await screen.findByTitle("Lens")).toHaveAttribute(
    "src",
    `${window.location.origin}/gateway/lens/ui/?tab=traces`,
  );
  expect(
    network.mock.calls.map(([input]) => new URL(input instanceof Request ? input.url : String(input)).pathname),
  ).toEqual(["/gateway/lens/ui/", "/gateway/lens/session", "/gateway/lens/service"]);
});
