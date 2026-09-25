import "openai/shims/web";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import userEvent from "@testing-library/user-event";
import type { ComponentType } from "react";
import SidebarAccountMenu from "@/components/SidebarAccountMenu/SidebarAccountMenu";
import UserDropdown from "@/components/Navbar/UserDropdown/UserDropdown";
import LiteAdmin from "./LiteAdmin";
import { MAX_INPUT_LENGTH } from "./agent";

const { transport } = vi.hoisted(() => {
  const transport = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", transport);
  return { transport };
});

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

const MANAGEMENT = "https://management.test/proxy";
const INFERENCE = "https://management.test/inference";
const EXTERNAL_INFERENCE = "https://inference.test/proxy";
const NEW_KEY = "sk-created-for-widget-test";
type RecordedRequest = { url: string; headers: Headers; body: Record<string, unknown> };
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
const sockets: BackendSocket[] = [];
class BackendSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => Promise<void>) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: Record<string, unknown>[] = [];
  closed = false;
  constructor(public url: string) {
    sockets.push(this);
    queueMicrotask(() => this.onopen?.());
  }
  send(data: string) {
    this.sent.push(JSON.parse(data));
  }
  close() {
    this.closed = true;
  }
  async emit(event: unknown) {
    if (!this.closed) await this.onmessage?.({ data: JSON.stringify(event) });
  }
}
const keyAction = {
  id: "action-1",
  name: "key_create",
  title: "Create a virtual key",
  arguments: { key_alias: "Widget key", team_id: "team-1", max_budget: 40 },
  destructive: false,
};

function session(role = "proxy_admin", user = "first-admin") {
  const encode = (value: object) =>
    btoa(JSON.stringify(value)).replaceAll("=", "").replaceAll("+", "-").replaceAll("/", "_");
  const claims = {
    key: `sk-session-${user}`,
    user_id: user,
    user_role: role,
    auth_header_name: "X-Gateway-Session",
    exp: Date.now() / 1000 + 3600,
  };
  const token = `${encode({ alg: "none" })}.${encode(claims)}.test`;
  document.cookie = `token=${token}; Path=/`;
  return token;
}

function SessionReady() {
  const { authLoading } = useAuth();
  return <output>{authLoading ? "Session loading" : "Session ready"}</output>;
}

function renderWidget(Menu?: ComponentType<{ onLogout: () => void }>) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const tree = () => (
    <QueryClientProvider client={client}>
      <Toaster />
      <AuthProvider>
        <SessionReady />
        {Menu && <Menu onLogout={() => undefined} />}
        <LiteAdmin />
      </AuthProvider>
    </QueryClientProvider>
  );
  const view = render(tree());
  return { ...view, refresh: () => view.rerender(tree()), client };
}

interface GatewayOptions {
  settings?: { target: string; status: number } | ((request: Request) => Promise<Response>);
}
function gateway(options: GatewayOptions = {}) {
  const requests: RecordedRequest[] = [];
  const settings = options.settings ?? { target: INFERENCE, status: 200 };
  transport.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init);
    requests.push({ url: request.url, headers: request.headers, body: {} });
    const path = new URL(request.url).pathname;
    if (path.endsWith("/litellm-ui-config"))
      return json({ proxy_base_url: MANAGEMENT, server_root_path: "", admin_ui_disabled: false });
    if (path.endsWith("/health/readiness/details")) return json({ status: "healthy" });
    if (path.endsWith("/sso/get/ui_settings")) {
      if (typeof settings === "function") return settings(request);
      return json({ PROXY_BASE_URL: MANAGEMENT, LITELLM_UI_API_DOC_BASE_URL: settings.target }, settings.status);
    }
    if (path.endsWith("/model_group/info"))
      return json({
        data: [
          { model_group: "a-embedding", mode: "embedding" },
          { model_group: "chat-model", mode: "chat" },
        ],
      });
    throw new Error(`Unexpected browser request: ${request.url}`);
  });
  return requests;
}

async function selectModel() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("combobox", { name: "LiteAdmin model" }));
  await user.click(await screen.findByRole("option", { name: "chat-model" }));
}

async function openWidget() {
  fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
  await selectModel();
}

function send(text: string) {
  fireEvent.change(screen.getByPlaceholderText("Ask LiteAdmin…"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

beforeEach(() => {
  transport.mockReset();
  sockets.length = 0;
  vi.stubGlobal("WebSocket", BackendSocket);
  localStorage.clear();
  sessionStorage.clear();
  switchToWorkerUrl(null);
  setGlobalLitellmHeaderName("Authorization");
  session();
});

afterEach(() => {
  toast.dismiss();
  document.cookie = "token=; Max-Age=0; Path=/";
});

describe("LiteAdmin in the gateway", () => {
  it.each([
    ["sidebar", SidebarAccountMenu],
    ["navbar", UserDropdown],
  ] as const)("persists Hide LiteAdmin from the %s account menu", async (_name, Menu) => {
    gateway();
    const user = userEvent.setup();
    const view = renderWidget(Menu);
    await screen.findByRole("button", { name: "LiteAdmin" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const toggle = await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    expect(toggle).toBeChecked();
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();

    view.unmount();
    const restored = renderWidget(Menu);
    await screen.findByText("Session ready");
    await waitFor(() => expect(restored.client.isFetching()).toBe(0));
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const savedToggle = await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" });
    expect(savedToggle).toBeChecked();
    await user.click(savedToggle);
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
  });

  it("isolates Hide LiteAdmin by admin and gateway and reacts to another tab clearing it", async () => {
    gateway();
    const user = userEvent.setup();
    const view = renderWidget(SidebarAccountMenu);
    await screen.findByRole("button", { name: "LiteAdmin" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    await user.click(await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" }));

    session("proxy_admin", "second-admin");
    view.refresh();
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide LiteAdmin" })).not.toBeChecked();
    session();
    view.refresh();
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();

    switchToWorkerUrl("https://other-gateway.test");
    view.refresh();
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide LiteAdmin" })).not.toBeChecked();
    switchToWorkerUrl(MANAGEMENT);
    view.refresh();
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();

    act(() => {
      localStorage.clear();
      window.dispatchEvent(new StorageEvent("storage", { key: null }));
    });
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide LiteAdmin" })).not.toBeChecked();
  });

  it.each([
    ["sidebar", SidebarAccountMenu],
    ["navbar", UserDropdown],
  ] as const)("does not offer Hide LiteAdmin to a view-only admin in the %s menu", async (_name, Menu) => {
    session("proxy_admin_viewer");
    gateway();
    renderWidget(Menu);
    await screen.findByText("Session ready");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    expect(await screen.findByRole("switch", { name: "Toggle hide all prompts" })).toBeInTheDocument();
    expect(screen.queryByRole("switch", { name: "Toggle hide LiteAdmin" })).not.toBeInTheDocument();
  });

  it.each(["proxy_admin_viewer", "internal_user", "internal_user_viewer", "org_admin"])(
    "does not expose operations to %s",
    async (role) => {
      session(role);
      const requests = gateway();
      const { client } = renderWidget();
      await screen.findByText("Session ready");
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
      expect(requests.every((request) => request.url.endsWith("/litellm-ui-config"))).toBe(true);
    },
  );

  it("sends chat to the backend with the selected model and displays its answer", async () => {
    gateway();
    renderWidget();
    await openWidget();
    send("Check the team budget");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    expect(sockets[0].url).toBe("wss://management.test/proxy/liteadmin/chat");
    expect(sockets[0].sent[0]).toEqual({
      access_token: "sk-session-first-admin",
      chat: {
        model: "chat-model",
        inference_base_url: `${INFERENCE}`,
        messages: [{ role: "user", content: "Check the team budget" }],
      },
    });
    await act(async () => {
      await sockets[0].emit({ type: "message", text: "The team budget is $40." });
      await sockets[0].emit({ type: "done" });
    });
    expect(screen.getByText("The team budget is $40.")).toBeInTheDocument();
  });

  it("keeps one action card through review and execution, and excludes its generated key from follow-ups", async () => {
    gateway();
    renderWidget();
    await openWidget();
    send("Create a key");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    act(() => {
      void sockets[0].emit({ type: "approval", action: keyAction });
    });
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    expect(within(review).getByText("Widget key")).toBeInTheDocument();
    expect(sockets[0].sent).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    const confirm = await screen.findByRole("button", { name: "Confirm change" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    await waitFor(() => expect(sockets[0].sent).toHaveLength(2));
    expect(sockets[0].sent[1]).toEqual({ id: keyAction.id, approved: true });
    expect(screen.getByRole("button", { name: "New chat" })).toBeDisabled();
    await act(async () => {
      await sockets[0].emit({ type: "result", action: keyAction, result: { status: "completed", key: NEW_KEY } });
      await sockets[0].emit({ type: "message", text: "Created the key." });
      await sockets[0].emit({ type: "done" });
    });
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy generated key" })).toBeInTheDocument();
    expect(screen.getAllByRole("region", { name: "Create a virtual key" })).toHaveLength(1);
    send("Thanks");
    await waitFor(() => expect(sockets[1]?.sent).toHaveLength(1));
    expect(JSON.stringify(sockets[1].sent)).toContain("Gateway action receipt");
    expect(JSON.stringify(sockets[1].sent)).not.toContain(NEW_KEY);
    expect(localStorage.length).toBe(0);
  });

  it("cancels a pending review by closing its backend connection", async () => {
    gateway();
    renderWidget();
    await openWidget();
    send("Create a key");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    act(() => {
      void sockets[0].emit({ type: "approval", action: keyAction });
    });
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(sockets[0].closed).toBe(true);
    expect(sockets[0].sent).toHaveLength(1);
  });

  it("warns about an interrupted write on account change and rejects a late key", async () => {
    gateway();
    const view = renderWidget();
    await openWidget();
    send("Create a key");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    act(() => {
      void sockets[0].emit({ type: "approval", action: keyAction });
    });
    fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
    await waitFor(() => expect(sockets[0].sent).toHaveLength(2));
    session("proxy_admin", "second-admin");
    view.refresh();
    expect(await screen.findByText(/A submitted change may have completed/)).toBeInTheDocument();
    expect(sockets[0].closed).toBe(true);
    await act(async () =>
      sockets[0].emit({ type: "result", action: keyAction, result: { status: "completed", key: NEW_KEY } }),
    );
    expect(screen.queryByText(NEW_KEY)).not.toBeInTheDocument();
  });

  it("cancels an unsubmitted approval when the backend connection fails", async () => {
    gateway();
    renderWidget();
    await openWidget();
    send("Create a key");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    act(() => {
      void sockets[0].emit({ type: "approval", action: keyAction });
    });
    expect(await screen.findByRole("button", { name: "Confirm change" })).toBeEnabled();
    await act(async () => sockets[0].emit({ type: "error", message: "Connection expired" }));
    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(sockets[0].sent).toHaveLength(1);
  });

  it("requires consent before allowing inference at a different configured origin", async () => {
    gateway({ settings: { target: EXTERNAL_INFERENCE, status: 200 } });
    renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    const approve = await screen.findByRole("button", { name: "Use configured gateway" });
    expect(sockets).toHaveLength(0);
    fireEvent.click(approve);
    await selectModel();
    send("Hello");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    expect(sockets[0].sent[0]).toMatchObject({ chat: { inference_base_url: EXTERNAL_INFERENCE } });
  });

  it("shows image descriptions without loading remote images from an admin answer", async () => {
    gateway();
    renderWidget();
    await openWidget();
    send("Check spend");
    await waitFor(() => expect(sockets[0]?.sent).toHaveLength(1));
    await act(async () => {
      await sockets[0].emit({ type: "message", text: "![Team spend chart](https://image.invalid/private.png)" });
      await sockets[0].emit({ type: "done" });
    });
    expect(screen.getByText("Team spend chart")).toBeInTheDocument();
    expect(within(screen.getByLabelText("LiteAdmin conversation")).queryByRole("img")).not.toBeInTheDocument();
  });

  it("blocks chat when gateway settings cannot be loaded", async () => {
    gateway({ settings: { target: INFERENCE, status: 503 } });
    renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    expect(await screen.findByText("Could not load gateway settings.")).toBeInTheDocument();
    expect(sockets).toHaveLength(0);
  });

  it("leaves an oversized draft editable and keeps it out of the transcript", async () => {
    gateway();
    renderWidget();
    await openWidget();
    const draft = "a".repeat(MAX_INPUT_LENGTH + 1);
    send(draft);
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue(draft);
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("8,000 characters");
    expect(within(screen.getByLabelText("LiteAdmin conversation")).queryByText(draft)).not.toBeInTheDocument();
  });
});
