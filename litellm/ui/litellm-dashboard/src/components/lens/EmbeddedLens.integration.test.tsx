import { act, fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, useTheme } from "next-themes";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { EmbeddedLens } from "./EmbeddedLens";
import { GatewayLensNavigation, GatewayLensNavigationProvider } from "./GatewayLensNavigation";
import { SidebarProvider } from "@/components/ui/sidebar";

const network = vi.fn<typeof fetch>();
const connection = (configured = true, connected = true, browserUrl = "https://lens.example.test") =>
  Response.json({ configured, connected, browser_url: browserUrl });

beforeEach(() => {
  network.mockReset();
  vi.stubGlobal("fetch", network);
  window.history.replaceState(null, "", "/ui/lens/?tab=traces");
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const render = () => renderWithProviders(<EmbeddedLens />);

it("frames the connected Lens at the tab in the dashboard URL", async () => {
  network.mockResolvedValueOnce(connection()).mockResolvedValueOnce(new Response("<html></html>"));
  render();
  expect(await screen.findByTitle("Lens")).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
});

it("restores the dashboard's storage fallback before opening Lens", async () => {
  const token = `header.${btoa(JSON.stringify({ key: "sk-dashboard-session" }))}.signature`;
  window.sessionStorage.setItem("token", token);
  const requests: Request[] = [];
  network.mockImplementation(async (input, init) => {
    const request = new Request(input, init);
    requests.push(request);
    if (new URL(request.url).pathname === "/lens/service") return connection();
    if (new URL(request.url).pathname === "/lens/session") return new Response(null, { status: 204 });
    return new Response("<html></html>");
  });
  try {
    render();
    expect(await screen.findByTitle("Lens")).toHaveAttribute("src", `${window.location.origin}/lens/ui/?tab=traces`);
    expect(requests.map((request) => new URL(request.url).pathname)).toEqual([
      "/lens/service",
      "/lens/session",
      "/lens/ui/",
    ]);
    expect(requests[0].headers.get("Authorization")).toBe("Bearer sk-dashboard-session");
    expect(requests[1].method).toBe("POST");
    expect(requests[1].credentials).toBe("include");
    expect(requests[1].redirect).toBe("error");
    expect(requests[1].headers.get("Authorization")).toBeNull();
    expect(await requests[1].json()).toEqual({ token });
  } finally {
    window.sessionStorage.removeItem("token");
  }
});

it.each([401, 403, 409])("does not open Lens when the dashboard session is refused (%s)", async (status) => {
  window.sessionStorage.setItem("token", "dashboard-session-fixture");
  network
    .mockResolvedValueOnce(connection())
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
  network.mockResolvedValueOnce(connection()).mockResolvedValueOnce(new Response("<html></html>"));
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
          <EmbeddedLens />
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
  network.mockResolvedValueOnce(connection(false, false));
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
    .mockResolvedValueOnce(connection(false, false))
    .mockResolvedValueOnce(connection())
    .mockResolvedValueOnce(new Response("<html></html>"));
  render();
  fireEvent.click(await screen.findByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
  expect(screen.queryByRole("heading", { name: "The gateway that helps your agents improve" })).not.toBeInTheDocument();
});

it("offers the standalone launch only for a connected older Lens with a browser address", async () => {
  network.mockResolvedValueOnce(connection()).mockResolvedValueOnce(new Response("", { status: 404 }));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
    "href",
    `${window.location.origin}/lens/login`,
  );
  expect(screen.queryByTitle("Lens")).not.toBeInTheDocument();
});

it("does not offer the localhost launch when the gateway has no valid browser address", async () => {
  network.mockResolvedValueOnce(connection(true, true, "")).mockResolvedValueOnce(new Response("", { status: 404 }));
  render();
  expect(await screen.findByRole("heading", { name: "Update Lens to use it here" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  expect(screen.getByText(/configure a Lens address that is accessible from your browser/)).toBeVisible();
});

it("shows a connection problem rather than installation or upgrade for a configured disconnected service", async () => {
  network.mockResolvedValueOnce(connection(true, false));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
});

it.each([401, 403, 404, 503])("does not mistake a failed connection check (%s) for missing setup", async (status) => {
  network.mockResolvedValueOnce(Response.json({ detail: "Unavailable" }, { status }));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Set up Lens" })).not.toBeInTheDocument();
});

it("can retry a failed request without reloading the dashboard", async () => {
  network
    .mockRejectedValueOnce(new TypeError("network"))
    .mockResolvedValueOnce(connection())
    .mockResolvedValueOnce(new Response("<html></html>"));
  render();
  expect(await screen.findByRole("heading", { name: "Could not connect to Lens" })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Check connection" }));
  expect(await screen.findByTitle("Lens")).toBeVisible();
});
