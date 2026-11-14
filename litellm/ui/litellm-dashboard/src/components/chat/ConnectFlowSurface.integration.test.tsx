import { webcrypto } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ConnectFlowSurface from "./ConnectFlowSurface";
import McpOAuthCallbackPage from "@/app/mcp/oauth/callback/page";
import {
  exchangeMcpOAuthToken,
  fetchConnectFlow,
  fetchMCPServers,
  getMCPOAuthUserCredentialStatus,
  listMCPTools,
  registerMcpOAuthClient,
  storeMCPOAuthUserCredential,
  type ConnectFlowStatus,
} from "@/components/networking";
import type { MCPServer } from "@/components/mcp_tools/types";

const query = vi.hoisted(() => ({ params: {} as Record<string, string | null> }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => ({ get: (key: string) => query.params[key] ?? null }),
}));
vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  fetchConnectFlow: vi.fn(),
  fetchMCPServers: vi.fn(),
  listMCPTools: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  registerMcpOAuthClient: vi.fn(),
  exchangeMcpOAuthToken: vi.fn(),
  storeMCPOAuthUserCredential: vi.fn(),
}));

const CONNECT_PAGE = "https://gateway.example.com/ui/connect?connect_flow=flow-handle-123";
const realLocation = Object.getOwnPropertyDescriptor(window, "location");

const browser = { visits: [CONNECT_PAGE] };

const stubBrowserLocation = () => {
  browser.visits = [CONNECT_PAGE];
  const location = {
    origin: "https://gateway.example.com",
    pathname: "/ui/connect",
    get href() {
      return browser.visits[browser.visits.length - 1];
    },
    set href(url: string) {
      browser.visits.push(url);
    },
    replace: (url: string) => {
      browser.visits.push(url);
    },
  };
  Object.defineProperty(window, "location", { configurable: true, value: location });
};

const vendorVisits = () => browser.visits.filter((url) => url.includes("/v1/mcp/server/oauth/"));

const servers = [
  { server_id: "catalog-id", server_name: "catalog", auth_type: "none", connected_app_reachable: true },
  { server_id: "orders-id", server_name: "orders", auth_type: "none", connected_app_reachable: true },
  { server_id: "tracker-id", server_name: "tracker", auth_type: "none", connected_app_reachable: true },
] as MCPServer[];

const vault = {
  server_id: "vault-id",
  server_name: "vault",
  auth_type: "oauth2",
  oauth2_flow: "authorization_code",
  connected_app_reachable: true,
} as MCPServer;

const flow = (kind: ConnectFlowStatus["state"], connected: boolean | null = null): ConnectFlowStatus => ({
  state: kind,
  client_origin: "https://claude.ai",
  server_id: kind === "interactive" || kind === "m2m" ? "s-design" : null,
  server_name: kind === "interactive" || kind === "m2m" ? "design_tool" : null,
  connected,
  servers: null,
});

const deferred = <T,>() => {
  let resolve: (value: T) => void = () => {};
  let reject: (reason: Error) => void = () => {};
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve: (value: T) => resolve(value), reject: (reason: Error) => reject(reason) };
};

const renderSurface = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (
    <QueryClientProvider client={client}>
      <ConnectFlowSurface accessToken="token-123" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>
  );
  const view = render(tree);
  return { unmount: view.unmount, flushPendingRenders: () => view.rerender(tree) };
};

const openConnectPage = (result: ConnectFlowStatus = flow("unscoped")) => {
  vi.mocked(fetchConnectFlow).mockResolvedValue(result);
  return renderSurface();
};

const returnFromVendor = async (page: { unmount: () => void }) => {
  await waitFor(() => expect(window.location.href).toContain("/authorize"));
  const vendorState = new URL(window.location.href).searchParams.get("state");
  page.unmount();
  query.params = { code: "vendor-code", state: vendorState };
  const callback = render(<McpOAuthCallbackPage />);
  await waitFor(() => expect(window.location.href).toContain("mcpOauthReturn=apps"));
  callback.unmount();
  query.params = { connect_flow: "flow-handle-123", mcpOauthReturn: "apps" };
  return renderSurface();
};

const serverSwitch = (name: string) => screen.findByRole("switch", { name: `Include ${name} in this connection` });
const finishButton = () => screen.getByRole("button", { name: /finish connecting/i });
const serverIdsFinishSends = () =>
  screen
    .queryAllByDisplayValue(/./)
    .filter((field) => field.getAttribute("name") === "server")
    .map((field) => field.getAttribute("value"));

beforeEach(() => {
  query.params = { connect_flow: "flow-handle-123" };
  vi.stubGlobal("crypto", webcrypto);
  stubBrowserLocation();
  vi.mocked(fetchMCPServers).mockResolvedValue(servers);
  vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });
  vi.mocked(registerMcpOAuthClient).mockResolvedValue({ client_id: "registered-client" });
  vi.mocked(exchangeMcpOAuthToken).mockResolvedValue({ access_token: "vendor-token" });
  vi.mocked(getMCPOAuthUserCredentialStatus).mockImplementation(async (_token, serverId) => ({
    server_id: serverId,
    has_credential: false,
    is_expired: false,
  }));
  vi.mocked(storeMCPOAuthUserCredential).mockImplementation(async (_token, serverId) => ({
    server_id: serverId,
    has_credential: true,
    is_expired: false,
  }));
});

afterEach(() => {
  query.params = {};
  sessionStorage.clear();
  vi.unstubAllGlobals();
  if (realLocation) Object.defineProperty(window, "location", realLocation);
  vi.resetAllMocks();
});

describe("the connect page as one tree", () => {
  it.each([
    { result: flow("unscoped"), grid: true, finish: true, cancel: false },
    { result: flow("interactive", true), grid: false, finish: true, cancel: true },
    { result: flow("m2m", true), grid: false, finish: true, cancel: true },
    { result: flow("stale"), grid: false, finish: false, cancel: true },
  ])("renders $result.state without widening its action surface", async ({ result, grid, finish, cancel }) => {
    openConnectPage(result);

    await screen.findAllByText(/https:\/\/claude\.ai/);
    expect(screen.queryByRole("tablist") !== null).toBe(grid);
    expect(screen.queryByRole("button", { name: /finish connecting/i }) !== null).toBe(finish);
    expect(screen.queryByRole("button", { name: "Cancel" }) !== null).toBe(cancel);
    expect(vendorVisits()).toEqual([]);
  });

  it.each([flow("interactive", true), flow("m2m", true)])(
    "leaves Finish open on a $state flow, which has no panel to wait for",
    async (result) => {
      openConnectPage(result);

      expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeEnabled();
    },
  );

  it("keeps the grid and Finish hidden until the gateway accepts a handle", () => {
    vi.mocked(fetchConnectFlow).mockReturnValue(new Promise(() => {}));
    renderSurface();

    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toHaveAttribute("value", "deny");
  });

  it("keeps the grid and Finish hidden when flow validation fails", async () => {
    query.params = { connect_flow: "invalid-handle" };
    vi.mocked(fetchConnectFlow).mockRejectedValue(new Error("invalid flow"));
    renderSurface();

    await screen.findByRole("button", { name: "Cancel" });
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
  });

  it("sends the user to the vendor once on an interactive flow that is not signed in", async () => {
    openConnectPage(flow("interactive", false));

    await waitFor(() => expect(window.location.href).toContain("/v1/mcp/server/oauth/s-design/authorize"));
    expect(vendorVisits()).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  it("shows Finish once a login returning from the vendor is saved", async () => {
    let saved = false;
    vi.mocked(fetchConnectFlow).mockImplementation(async () => flow("interactive", saved));
    vi.mocked(storeMCPOAuthUserCredential).mockImplementation(async (_token, serverId) => {
      saved = true;
      return { server_id: serverId, has_credential: true, is_expired: false };
    });

    await returnFromVendor(renderSurface());

    expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeEnabled();
    expect(exchangeMcpOAuthToken).toHaveBeenCalledWith(
      expect.objectContaining({ serverId: "s-design", code: "vendor-code" }),
    );
    expect(storeMCPOAuthUserCredential).toHaveBeenCalledWith(
      "token-123",
      "s-design",
      expect.objectContaining({ access_token: "vendor-token" }),
    );
  });

  it("offers Connect again, without Finish, when a login returning from the vendor cannot be saved", async () => {
    vi.mocked(fetchConnectFlow).mockResolvedValue(flow("interactive", false));
    const save = deferred<never>();
    vi.mocked(storeMCPOAuthUserCredential).mockReturnValue(save.promise);

    await returnFromVendor(renderSurface());
    await screen.findByRole("button", { name: "Connecting…" });
    await act(async () => {
      save.reject(new Error("save failed"));
    });

    expect(await screen.findByRole("button", { name: "Connect" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: /finish connecting/i })).not.toBeInTheDocument();
  });

  it("holds Finish while a login returning from the vendor is being saved", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue([...servers, vault]);
    const exchange = deferred<{ access_token: string }>();
    vi.mocked(exchangeMcpOAuthToken).mockReturnValue(exchange.promise);
    const page = openConnectPage();
    fireEvent.click(await screen.findByText("Connect", { exact: true }));

    await returnFromVendor(page);

    expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeDisabled();
    expect(serverIdsFinishSends()).toEqual([]);

    await act(async () => {
      exchange.resolve({ access_token: "vendor-token" });
    });

    await waitFor(() => expect(finishButton()).toBeEnabled());
    expect(storeMCPOAuthUserCredential).toHaveBeenCalledWith(
      "token-123",
      "vault-id",
      expect.objectContaining({ access_token: "vendor-token" }),
    );
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("sends only the flow while every switch is on, so the session stays as on main", async () => {
    openConnectPage();

    expect(await serverSwitch("catalog")).toBeChecked();
    expect(await serverSwitch("orders")).toBeChecked();
    expect(await serverSwitch("tracker")).toBeChecked();
    expect(finishButton()).toBeEnabled();
    expect(screen.getByDisplayValue("flow-handle-123")).toHaveAttribute("name", "flow");
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("sends the servers still on once one is switched off, and only the flow once it is back on", async () => {
    openConnectPage();
    const orders = await serverSwitch("orders");

    fireEvent.click(orders);

    expect(orders).not.toBeChecked();
    expect(serverIdsFinishSends()).toEqual(["catalog-id", "tracker-id"]);
    expect(finishButton()).toBeEnabled();

    fireEvent.click(orders);

    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("holds Finish and says why once every switch is off", async () => {
    openConnectPage();

    fireEvent.click(await serverSwitch("catalog"));
    fireEvent.click(await serverSwitch("orders"));
    fireEvent.click(await serverSwitch("tracker"));

    expect(finishButton()).toBeDisabled();
    expect(finishButton()).toHaveAccessibleDescription("Turn on at least one server to finish connecting.");
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("keeps a server the client requires on, locked and sent", async () => {
    openConnectPage({ ...flow("unscoped"), required_server_ids: ["tracker-id"] });
    const tracker = await serverSwitch("tracker");

    fireEvent.click(tracker);
    fireEvent.click(await serverSwitch("orders"));

    expect(tracker).toBeChecked();
    expect(screen.getByText("Required by this client")).toBeInTheDocument();
    expect(serverIdsFinishSends()).toEqual(["catalog-id", "tracker-id"]);
  });

  it("keeps every switch on and sends only the flow when the flow cannot be narrowed", async () => {
    openConnectPage({ ...flow("unscoped"), selectable: false });
    const orders = await serverSwitch("orders");

    fireEvent.click(orders);

    expect(orders).toBeChecked();
    expect(screen.getByText(/https:\/\/claude\.ai will get every server you can reach/)).toBeInTheDocument();
    expect(finishButton()).toBeEnabled();
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("keeps a switch off across a reload and holds Finish until the server list is back", async () => {
    const { unmount } = openConnectPage();
    fireEvent.click(await serverSwitch("orders"));
    unmount();
    const list = deferred<MCPServer[]>();
    vi.mocked(fetchMCPServers).mockReturnValue(list.promise);

    openConnectPage();

    expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeDisabled();
    expect(serverIdsFinishSends()).toEqual([]);

    await act(async () => {
      list.resolve(servers);
    });

    expect(await serverSwitch("orders")).not.toBeChecked();
    await waitFor(() => expect(serverIdsFinishSends()).toEqual(["catalog-id", "tracker-id"]));
  });

  it("holds Finish after a reload with a switch off when the server list fails to load", async () => {
    const { unmount } = openConnectPage();
    fireEvent.click(await serverSwitch("orders"));
    unmount();
    vi.mocked(fetchMCPServers).mockRejectedValue(new Error("list failed"));

    const page = openConnectPage();

    expect(await screen.findByText(/Could not load your MCP servers/)).toBeInTheDocument();
    page.flushPendingRenders();
    expect(finishButton()).toBeDisabled();
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it("lets the user finish as on main while the server list is still loading", async () => {
    vi.mocked(fetchMCPServers).mockReturnValue(new Promise(() => {}));
    openConnectPage();

    expect(await screen.findByRole("button", { name: /finish connecting/i })).toBeEnabled();
    expect(serverIdsFinishSends()).toEqual([]);
  });

  it.each([
    {
      when: "fails to load",
      list: () => Promise.reject<MCPServer[]>(new Error("list failed")),
      shown: /Could not load your MCP servers/,
    },
    {
      when: "comes back empty",
      list: () => Promise.resolve<MCPServer[]>([]),
      shown: /No MCP servers are available to this connection yet/,
    },
  ])("lets the user finish as on main when the server list $when", async ({ list, shown }) => {
    vi.mocked(fetchMCPServers).mockImplementation(list);
    const page = openConnectPage();

    expect(await screen.findByText(shown)).toBeInTheDocument();
    page.flushPendingRenders();
    expect(finishButton()).toBeEnabled();
    expect(serverIdsFinishSends()).toEqual([]);
  });
});
