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
const keyArguments = {
  key_alias: "Widget key",
  team_id: "team-1",
  user_id: null,
  models: null,
  max_budget: 40,
  budget_duration: null,
  rpm_limit: null,
  tpm_limit: null,
  budget_id: null,
  duration: null,
};

type RecordedRequest = { url: string; headers: Headers; body: Record<string, unknown> };
type ModelReply = {
  role: "assistant";
  content: string | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
};

const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
const toolReply = (name: string, args: Record<string, unknown>): ModelReply => ({
  role: "assistant",
  content: null,
  tool_calls: [{ id: "call-test", type: "function", function: { name, arguments: JSON.stringify(args) } }],
});
const answer = (content: string): ModelReply => ({ role: "assistant", content });

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

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
  write?: () => Promise<Response>;
  read?: () => Promise<Response>;
  settings?: { target: string; status: number } | ((request: Request) => Promise<Response>);
}

function gateway(replies: (ModelReply | Promise<ModelReply>)[], options: GatewayOptions = {}) {
  const requests: RecordedRequest[] = [];
  const settings = options.settings ?? { target: INFERENCE, status: 200 };
  transport.mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init);
    const body: Record<string, unknown> = request.method === "GET" ? {} : await request.clone().json();
    requests.push({ url: request.url, headers: request.headers, body });
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
    if (path.endsWith("/chat/completions")) {
      const message = await replies.shift();
      if (!message) return json({ error: { message: "Model unavailable" } }, 503);
      const completion = {
        id: "completion-test",
        object: "chat.completion",
        created: 0,
        model: "chat-model",
        choices: [{ index: 0, message, finish_reason: message.tool_calls ? "tool_calls" : "stop" }],
      };
      return json(completion);
    }
    if (path.endsWith("/team/info"))
      return options.read ? options.read() : json({ team_info: { team_id: "team-1", max_budget: 40 } });
    if (path.endsWith("/team/update"))
      return options.write ? options.write() : json({ team_id: "team-1", max_budget: body.max_budget });
    if (path.endsWith("/key/generate"))
      return options.write ? options.write() : json({ key: NEW_KEY, key_alias: "Widget key" });
    throw new Error(`Unexpected request: ${request.url}`);
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
    gateway([]);
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
    gateway([]);
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
    gateway([]);
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
      const requests = gateway([]);
      const { client } = renderWidget();
      await screen.findByText("Session ready");
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
      expect(requests.every((request) => request.url.endsWith("/litellm-ui-config"))).toBe(true);
    },
  );

  it("uses the chosen model and existing session header for reads", async () => {
    const requests = gateway([toolReply("team_info", { team_id: "team-1" }), answer("The team budget is $40.")]);
    renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    expect(await screen.findByPlaceholderText("Ask LiteAdmin…")).toBeDisabled();
    await selectModel();
    send("Check the team budget");
    expect(await screen.findByText("The team budget is $40.")).toBeInTheDocument();
    const read = requests.find((request) => request.url.includes("/team/info"));
    expect(read?.headers.get("X-Gateway-Session")).toBe("Bearer sk-session-first-admin");
    expect(read?.url).toContain(`${MANAGEMENT}/team/info?`);
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions).toHaveLength(2);
    expect(completions.every((request) => request.url === `${INFERENCE}/chat/completions`)).toBe(true);
    expect(completions[0].headers.get("X-Gateway-Session")).toBe(read?.headers.get("X-Gateway-Session"));
    expect(completions[0].body.model).toBe("chat-model");
  });

  it("shows image descriptions without loading remote images from an admin answer", async () => {
    gateway([answer("Budget summary. ![Team spend chart](https://image.invalid/chart.png?team=private)")]);
    renderWidget();
    await openWidget();
    send("Check team spend");

    expect(await screen.findByText("Team spend chart")).toBeInTheDocument();
    expect(within(screen.getByLabelText("LiteAdmin conversation")).queryByRole("img")).not.toBeInTheDocument();
  });

  it("keeps a single inline action through review, close/reopen and one confirmed write", async () => {
    const response = deferred<Response>();
    const proposed = { ...toolReply("key_create", keyArguments), content: "Here is the requested change." };
    const requests = gateway([proposed, answer("Created the key."), answer("You are welcome.")], {
      write: () => response.promise,
    });
    renderWidget();
    await openWidget();
    send("Create a key for the team");
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    expect(within(review).getByText("Widget key")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    const confirm = await screen.findByRole("button", { name: "Confirm change" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    expect(screen.getByRole("button", { name: "New chat" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Stop request" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    await act(async () => response.resolve(json({ key: NEW_KEY })));
    fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    expect(await screen.findByText("Created the key.")).toBeInTheDocument();
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy generated key" })).toBeInTheDocument();
    expect(screen.getAllByRole("region", { name: "Create a virtual key" })).toHaveLength(1);
    expect(screen.getByLabelText("LiteAdmin conversation")).toHaveTextContent(
      /Here is the requested change\.[\s\S]*Completed[\s\S]*Created the key\./,
    );
    const writes = requests.filter((request) => request.url.endsWith("/key/generate"));
    expect(writes).toHaveLength(1);
    expect(writes[0].body).toEqual({ key_alias: "Widget key", team_id: "team-1", max_budget: 40 });
    send("Thanks");
    await screen.findByText("You are welcome.");
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    const receipt = {
      operation: "key_create",
      status: "completed",
      arguments: { key_alias: "Widget key", team_id: "team-1", max_budget: 40 },
    };
    expect(completions.at(-1)?.body.messages).toContainEqual({
      role: "assistant",
      content: `Gateway action receipt: ${JSON.stringify(receipt)}`,
    });
    expect(JSON.stringify(completions)).not.toContain(NEW_KEY);
    expect(localStorage.length).toBe(0);
  });

  it("carries completed and cancelled changes in order into the next read request", async () => {
    const fields = {
      team_id: "team-1",
      team_alias: null,
      organization_id: null,
      models: null,
      budget_duration: null,
      rpm_limit: null,
      tpm_limit: null,
    };
    const requests = gateway([
      toolReply("team_update", { ...fields, max_budget: 40 }),
      answer("The budget is now $40."),
      toolReply("team_update", { ...fields, max_budget: 41 }),
      toolReply("team_info", { team_id: "team-1" }),
      answer("The budget is still $40."),
    ]);
    renderWidget();
    await openWidget();
    send("Set the team budget to $40");
    fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
    await screen.findByText("The budget is now $40.");
    send("Set the team budget to $41");
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    send("Read the current team budget");
    expect(await screen.findByText("The budget is still $40.")).toBeInTheDocument();
    const completed = {
      operation: "team_update",
      status: "completed",
      arguments: { team_id: "team-1", max_budget: 40 },
    };
    const cancelled = {
      operation: "team_update",
      status: "cancelled",
      arguments: { team_id: "team-1", max_budget: 41 },
    };
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions[3].body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Set the team budget to $40" },
      { role: "assistant", content: `Gateway action receipt: ${JSON.stringify(completed)}` },
      { role: "assistant", content: "The budget is now $40." },
      { role: "user", content: "Set the team budget to $41" },
      { role: "assistant", content: `Gateway action receipt: ${JSON.stringify(cancelled)}` },
      { role: "user", content: "Read the current team budget" },
    ]);
    const writes = requests.filter((request) => request.url.endsWith("/team/update"));
    expect(writes.map((request) => request.body)).toEqual([{ team_id: "team-1", max_budget: 40 }]);
    expect(screen.queryByText(/A submitted change may have completed/)).not.toBeInTheDocument();
  });

  it.each(["account", "target"])("discards an unconfirmed review when the %s changes", async (transition) => {
    const settings = { target: INFERENCE, status: 200 };
    const requests = gateway([toolReply("key_create", keyArguments)], { settings });
    const view = renderWidget();
    await openWidget();
    send("Old request");
    await screen.findByRole("button", { name: "Confirm change" });
    if (transition === "account") {
      session("proxy_admin", "second-admin");
      view.refresh();
    } else {
      settings.target = `${INFERENCE}/updated`;
      await act(async () => view.client.invalidateQueries({ queryKey: ["proxySettings"] }));
    }
    await waitFor(() => expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument());
    expect(screen.queryByText("Old request")).not.toBeInTheDocument();
    expect(screen.queryByText(/A submitted change may have completed/)).not.toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it.each(["account", "target"])(
    "warns about an interrupted write when the %s changes and rejects its late key",
    async (transition) => {
      const response = deferred<Response>();
      const settings = { target: INFERENCE, status: 200 };
      const requests = gateway([toolReply("key_create", keyArguments)], { settings, write: () => response.promise });
      const view = renderWidget();
      await openWidget();
      send("Create a key");
      fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
      await waitFor(() => expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(1));
      if (transition === "account") {
        session("proxy_admin", "second-admin");
        view.refresh();
      } else {
        settings.target = `${INFERENCE}/updated`;
        await act(async () => view.client.invalidateQueries({ queryKey: ["proxySettings"] }));
      }
      expect(await screen.findByText(/A submitted change may have completed/)).toBeInTheDocument();
      await act(async () => response.resolve(json({ key: NEW_KEY })));
      expect(screen.queryByText(NEW_KEY)).not.toBeInTheDocument();
      expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(1);
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1);
    },
  );

  it("requires approval before sending the session to a different configured origin", async () => {
    const settings = { target: EXTERNAL_INFERENCE, status: 200 };
    const requests = gateway([answer("Connected.")], { settings });
    const { client } = renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    const approve = await screen.findByRole("button", { name: "Use configured gateway" });
    expect(requests.some((request) => request.url.startsWith(EXTERNAL_INFERENCE))).toBe(false);
    fireEvent.click(approve);
    await selectModel();
    send("Hello");
    await screen.findByText("Connected.");
    expect(requests.find((request) => request.url.endsWith("/chat/completions"))?.url).toBe(
      `${EXTERNAL_INFERENCE}/chat/completions`,
    );
    settings.target = `${EXTERNAL_INFERENCE}/changed`;
    await act(async () => client.invalidateQueries({ queryKey: ["proxySettings"] }));
    expect(await screen.findByRole("button", { name: "Use configured gateway" })).toBeInTheDocument();
    expect(screen.queryByText("Connected.")).not.toBeInTheDocument();
  });

  it("blocks inference when settings cannot be loaded", async () => {
    const requests = gateway([], { settings: { target: INFERENCE, status: 503 } });
    renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    expect(await screen.findByText("Could not load gateway settings.")).toBeInTheDocument();
    expect(requests.some((request) => request.url.endsWith("/chat/completions"))).toBe(false);
  });

  it("waits for fresh settings and consent after switching workers with the same admin session", async () => {
    const worker = "https://worker.test/proxy";
    const workerInference = "https://worker-inference.test/proxy";
    const workerSettings = deferred<Response>();
    const settingsRequested = deferred<void>();
    const requests = gateway([answer("Original gateway."), answer("Worker gateway.")], {
      settings: async (request) => {
        if (request.url === `${worker}/sso/get/ui_settings`) {
          settingsRequested.resolve();
          return workerSettings.promise;
        }
        return json({ PROXY_BASE_URL: MANAGEMENT, LITELLM_UI_API_DOC_BASE_URL: EXTERNAL_INFERENCE });
      },
    });
    const view = renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    fireEvent.click(await screen.findByRole("button", { name: "Use configured gateway" }));
    await selectModel();
    send("Check the original gateway");
    await screen.findByText("Original gateway.");

    act(() => {
      switchToWorkerUrl(worker);
      view.refresh();
    });
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    await settingsRequested.promise;
    expect(screen.queryByText(EXTERNAL_INFERENCE)).not.toBeInTheDocument();
    expect(await screen.findByLabelText("Loading gateway settings")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Ask LiteAdmin…")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use configured gateway" })).not.toBeInTheDocument();
    expect(screen.queryByText("Original gateway.")).not.toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1);

    const settings = { PROXY_BASE_URL: worker, LITELLM_UI_API_DOC_BASE_URL: workerInference };
    await act(async () => workerSettings.resolve(json(settings)));
    const consent = await screen.findByRole("button", { name: "Use configured gateway" });
    expect(screen.getByText(workerInference)).toBeInTheDocument();
    expect(screen.queryByText(EXTERNAL_INFERENCE)).not.toBeInTheDocument();
    expect(requests.some((request) => request.url.startsWith(workerInference))).toBe(false);
    fireEvent.click(consent);
    await selectModel();
    send("Check the worker gateway");
    await screen.findByText("Worker gateway.");
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions.map((request) => request.url)).toEqual([
      `${EXTERNAL_INFERENCE}/chat/completions`,
      `${workerInference}/chat/completions`,
    ]);
    expect(completions.map((request) => request.headers.get("X-Gateway-Session"))).toEqual([
      "Bearer sk-session-first-admin",
      "Bearer sk-session-first-admin",
    ]);
  });

  it("leaves an oversized draft editable and keeps it out of the transcript", async () => {
    gateway([]);
    renderWidget();
    await openWidget();
    const draft = "a".repeat(MAX_INPUT_LENGTH + 1);
    send(draft);
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toBeEnabled();
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue(draft);
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("8,000 characters");
    expect(within(screen.getByLabelText("LiteAdmin conversation")).queryByText(draft)).not.toBeInTheDocument();
  });

  it("retains the action receipt across later model and settings failures", async () => {
    const settings = { target: INFERENCE, status: 200 };
    const requests = gateway([toolReply("key_create", keyArguments)], { settings });
    const { client } = renderWidget();
    await openWidget();
    send("Create a key");
    const confirm = await screen.findByRole("button", { name: "Confirm change" });
    settings.status = 503;
    fireEvent.click(confirm);
    expect(await screen.findByText("Completed")).toBeInTheDocument();
    expect(await screen.findByText(/Completed changes are shown above/)).toBeInTheDocument();
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/sso/get/ui_settings"))).toHaveLength(1);
    await act(async () => client.invalidateQueries({ queryKey: ["proxySettings"] }));
    expect(requests.filter((request) => request.url.endsWith("/sso/get/ui_settings"))).toHaveLength(2);
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
  });

  it("marks a write failure uncertain and does not retry it", async () => {
    const requests = gateway([toolReply("key_create", keyArguments), answer("Check the key list before retrying.")], {
      write: async () => json({ error: "Private failure detail" }, 502),
    });
    renderWidget();
    await openWidget();
    send("Create a key");
    fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
    expect(await screen.findByText("Check result")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveTextContent(
      /Check.*before trying again/,
    );
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(1);
    send("What happened?");
    await screen.findByText("Check the key list before retrying.");
    const receipt = {
      operation: "key_create",
      status: "unknown",
      arguments: { key_alias: "Widget key", team_id: "team-1", max_budget: 40 },
    };
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions.at(-1)?.body.messages).toContainEqual({
      role: "assistant",
      content: `Gateway action receipt: ${JSON.stringify(receipt)}`,
    });
    expect(JSON.stringify(completions)).not.toContain("Private failure detail");
  });
});
