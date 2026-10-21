import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import userEvent from "@testing-library/user-event";
import { useEffect, type ComponentType } from "react";
import SidebarAccountMenu from "@/components/SidebarAccountMenu/SidebarAccountMenu";
import UserDropdown from "@/components/Navbar/UserDropdown/UserDropdown";
import LiteAdmin, { LiteAdminFrame, LiteAdminHome } from "./LiteAdmin";
import { MAX_INPUT_LENGTH } from "./agent";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";

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

type RecordedRequest = { url: string; headers: Headers; body: Record<string, unknown>; signal: AbortSignal };
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

let signedInUser = { user_id: "first-admin", user_role: "proxy_admin" };

function session(
  role = "proxy_admin",
  user = "first-admin",
  license: { premium_user?: boolean | null | string } = { premium_user: true },
  key = `sk-session-${user}`,
) {
  const encode = (value: object) =>
    btoa(JSON.stringify(value)).replaceAll("=", "").replaceAll("+", "-").replaceAll("/", "_");
  const claims = {
    key,
    user_id: user,
    user_role: role,
    ...license,
    auth_header_name: "X-Gateway-Session",
    exp: Date.now() / 1000 + 3600,
  };
  const token = `${encode({ alg: "none" })}.${encode(claims)}.test`;
  document.cookie = `token=${token}; Path=/`;
  signedInUser = { user_id: user, user_role: role };
  return token;
}

function SessionReady() {
  const { authLoading } = useAuth();
  return <output>{authLoading ? "Session loading" : "Session ready"}</output>;
}

let pageMounts = 0;

function Page() {
  useEffect(() => {
    pageMounts += 1;
  }, []);
  return null;
}

function renderWidget(Menu?: ComponentType<{ onLogout: () => void }>, showHome = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const tree = (home = showHome) => (
    <QueryClientProvider client={client}>
      <Toaster />
      <AuthProvider>
        <SessionReady />
        {Menu && <Menu onLogout={() => undefined} />}
        <LiteAdminFrame>
          <Page />
          {home && <LiteAdminHome />}
          <LiteAdmin />
        </LiteAdminFrame>
      </AuthProvider>
    </QueryClientProvider>
  );
  const view = render(tree());
  return { ...view, refresh: (home = showHome) => view.rerender(tree(home)), client };
}

interface GatewayOptions {
  models?: ModelGroup[];
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
    const recorded = { url: request.url, headers: request.headers, body, signal: request.signal };
    const path = new URL(request.url).pathname;
    if (path.endsWith("/ui/litellm-runtime-config.json")) return json({});
    if (path.endsWith("/management/v1/users/me")) return json({ ...signedInUser, user_email: null });
    requests.push(recorded);
    if (path.endsWith("/litellm-ui-config"))
      return json({ proxy_base_url: MANAGEMENT, server_root_path: "", admin_ui_disabled: false });
    if (path.endsWith("/health/readiness/details")) return json({ status: "healthy" });
    if (path.endsWith("/sso/get/ui_settings")) {
      if (typeof settings === "function") return settings(request);
      return json({ PROXY_BASE_URL: MANAGEMENT, LITELLM_UI_API_DOC_BASE_URL: settings.target }, settings.status);
    }
    if (path.endsWith("/model_group/info"))
      return json({
        data: options.models ?? [
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
    if (path.endsWith("/key/update")) return json(body);
    throw new Error(`Unexpected request: ${request.url}`);
  });
  return requests;
}

async function selectModel(model = "chat-model") {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("combobox", { name: "LiteAdmin model" }));
  await user.click(await screen.findByRole("option", { name: model }));
}

async function openWidget() {
  fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
  await selectModel();
}

function send(text: string) {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

beforeEach(() => {
  pageMounts = 0;
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
  it.each(["keyboard", "click"])("continues one conversation from Home via %s and the dock", async (method) => {
    const requests = gateway([answer("First reply."), answer("Second reply."), answer("Third reply.")]);
    renderWidget(undefined, true);
    const home = await screen.findByRole("region", { name: "Ask LiteAdmin" });
    await selectModel();
    const input = within(home).getByPlaceholderText("What would you like to do?");
    fireEvent.change(input, { target: { value: "First question" } });
    if (method === "keyboard") fireEvent.keyDown(input, { key: "Enter" });
    else fireEvent.click(within(home).getByRole("button", { name: "Send message" }));
    expect(await screen.findByText("First reply.")).toBeInTheDocument();
    expect(home).not.toBeVisible();
    expect(screen.getAllByRole("complementary", { name: "LiteAdmin" })).toHaveLength(1);
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1);
    send("Second question");
    expect(await screen.findByText("Second reply.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    expect(home).toBeVisible();
    send("Third question");
    expect(await screen.findByText("Third reply.")).toBeInTheDocument();
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions).toHaveLength(3);
    expect(completions[2].body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "First question" },
      { role: "assistant", content: "First reply." },
      { role: "user", content: "Second question" },
      { role: "assistant", content: "Second reply." },
      { role: "user", content: "Third question" },
    ]);
  });

  it("prefills each Home suggestion without sending until the edited draft is submitted", async () => {
    const requests = gateway([answer("Budget details received.")]);
    renderWidget(undefined, true);
    await selectModel();
    const input = screen.getByRole("textbox", { name: "Message LiteAdmin" });
    for (const label of ["Create a key", "Add a model", "Set a budget"]) {
      fireEvent.click(screen.getByRole("button", { name: label, exact: true }));
      expect(input).toHaveValue(label);
      expect(input).toHaveFocus();
      expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(0);
    }
    fireEvent.change(input, { target: { value: "Set a budget of $10 for my team" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(0);
    fireEvent.keyDown(input, { key: "Enter" });
    expect(await screen.findByText("Budget details received.")).toBeInTheDocument();
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions).toHaveLength(1);
    expect(completions[0].body.model).toBe("chat-model");
    expect(completions[0].body.messages).toContainEqual({ role: "user", content: "Set a budget of $10 for my team" });
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("preserves the draft and chosen model when opening, closing and leaving Home", async () => {
    const requests = gateway([answer("Draft received.")], {
      models: [
        { model_group: "claude-sonnet-5-5", mode: "chat" },
        { model_group: "chat-model", mode: "chat" },
      ],
    });
    const view = renderWidget(undefined, true);
    await selectModel();
    fireEvent.change(screen.getByPlaceholderText("What would you like to do?"), {
      target: { value: "Keep this draft" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Recent chats" }));
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("Keep this draft");
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    const home = screen.getByRole("region", { name: "Ask LiteAdmin" });
    expect(within(home).getByPlaceholderText("What would you like to do?")).toHaveValue("Keep this draft");
    view.refresh(false);
    expect(home).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("Keep this draft");
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByText("Draft received.")).toBeInTheDocument();
    const completion = requests.find((request) => request.url.endsWith("/chat/completions"));
    expect(completion?.body.model).toBe("chat-model");
    expect(completion?.body.messages).toContainEqual({ role: "user", content: "Keep this draft" });
  });

  it("keeps each tab's history, model and draft independent through the Home composer", async () => {
    const requests = gateway(
      [answer("First answer."), answer("Second answer."), answer("Second follow-up."), answer("First follow-up.")],
      {
        models: [
          { model_group: "chat-model", mode: "chat" },
          { model_group: "other-chat-model", mode: "chat" },
        ],
      },
    );
    renderWidget(undefined, true);
    await selectModel();
    send("First question");
    await screen.findByText("First answer.");
    expect(within(screen.getByRole("tablist", { name: "LiteAdmin chats" })).getAllByRole("tab")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByText("First answer.")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("");
    await selectModel("other-chat-model");
    send("Second question");
    await screen.findByText("Second answer.");
    fireEvent.change(screen.getByPlaceholderText("Ask LiteAdmin…"), { target: { value: "Second draft" } });

    fireEvent.click(screen.getByRole("tab", { name: "Chat 1" }));
    expect(screen.getByText("First answer.")).toBeInTheDocument();
    expect(screen.queryByText("Second answer.")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("");
    expect(screen.getByRole("combobox", { name: "LiteAdmin model" })).toHaveValue("chat-model");
    fireEvent.change(screen.getByPlaceholderText("Ask LiteAdmin…"), { target: { value: "First draft" } });
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    const home = screen.getByRole("region", { name: "Ask LiteAdmin" });
    expect(within(home).getByPlaceholderText("What would you like to do?")).toHaveValue("First draft");
    expect(screen.getAllByPlaceholderText("What would you like to do?")).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "Recent chats" }));
    fireEvent.click(screen.getByRole("tab", { name: "Chat 2" }));
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("Second draft");
    expect(screen.getByRole("combobox", { name: "LiteAdmin model" })).toHaveValue("other-chat-model");
    expect(screen.getAllByLabelText("LiteAdmin conversation")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    expect(within(home).getByPlaceholderText("What would you like to do?")).toHaveValue("Second draft");
    fireEvent.click(within(home).getByRole("button", { name: "Send message" }));
    await screen.findByText("Second follow-up.");
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByRole("tab", { name: "Chat 1" }));
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("First draft");
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("First follow-up.");

    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions.map((request) => request.body.model)).toEqual([
      "chat-model",
      "other-chat-model",
      "other-chat-model",
      "chat-model",
    ]);
    expect(completions[1].body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Second question" },
    ]);
    expect(completions[2].body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Second question" },
      { role: "assistant", content: "Second answer." },
      { role: "user", content: "Second draft" },
    ]);
    expect(completions[3].body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "First question" },
      { role: "assistant", content: "First answer." },
      { role: "user", content: "First draft" },
    ]);
  });

  it("supports keyboard tab switching and keeps focus on a surviving tab after closing", async () => {
    gateway([]);
    renderWidget();
    await openWidget();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "New chat" }));
    await user.click(screen.getByRole("tab", { name: "Chat 1" }));
    await user.keyboard("{ArrowRight}{Enter}");
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Close Chat 2" }));
    expect(screen.getByRole("tab", { name: "Chat 1" })).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Close Chat 1" }));
    expect(screen.getByRole("tab", { name: "Chat 3" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Chat 3" })).toHaveAttribute("aria-selected", "true");
  });

  it("keeps tab focus when returning to a saved approval and focuses it when reopening the panel", async () => {
    const requests = gateway([toolReply("key_create", keyArguments)]);
    renderWidget();
    await openWidget();
    send("Prepare a key for approval");
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    expect(review).toHaveFocus();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "New chat" }));
    await user.click(screen.getByRole("tab", { name: "Chat 2" }));
    await user.keyboard("{ArrowLeft}{Enter}");
    expect(screen.getByRole("tab", { name: "Chat 1" })).toHaveFocus();
    expect(screen.getByRole("button", { name: "Confirm change" })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Chat 2" }));
    await user.click(screen.getByRole("button", { name: "Close Chat 2" }));
    expect(screen.getByRole("tab", { name: "Chat 1" })).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    await user.click(screen.getByRole("button", { name: "LiteAdmin" }));
    expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveFocus();
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("lets simultaneous turns finish independently without a background review stealing focus", async () => {
    const first = deferred<ModelReply>();
    const second = deferred<ModelReply>();
    const requests = gateway([first.promise, second.promise]);
    renderWidget();
    await openWidget();
    send("First pending question");
    await waitFor(() =>
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1),
    );
    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    await selectModel();
    send("Second pending question");
    await waitFor(() =>
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(2),
    );
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions.every((request) => !request.signal.aborted)).toBe(true);
    await act(async () => second.resolve(answer("Second completed first.")));
    await screen.findByText("Second completed first.");
    const composer = screen.getByPlaceholderText("Ask LiteAdmin…");
    fireEvent.change(composer, { target: { value: "Keep typing in the second chat" } });
    act(() => composer.focus());

    await act(async () => first.resolve(toolReply("key_create", keyArguments)));
    expect(composer).toHaveFocus();
    expect(composer).toHaveValue("Keep typing in the second chat");
    expect(composer).toBeEnabled();
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.queryByText("First pending question")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Chat 1" }));
    expect(await screen.findByRole("region", { name: "Create a virtual key" })).toHaveTextContent("Widget key");
    expect(screen.getByText("First pending question")).toBeInTheDocument();
    expect(screen.queryByText("Second completed first.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Chat 2" }));
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("Keep typing in the second chat");
    expect(screen.getByText("Second completed first.")).toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("cancels only a closed tab's approval and prevents closing an applying write even from another tab", async () => {
    const response = deferred<Response>();
    const requests = gateway(
      [
        toolReply("key_create", keyArguments),
        answer("Second chat is usable."),
        toolReply("key_create", { ...keyArguments, key_alias: "Second chat key" }),
        answer("Third chat is usable."),
        answer("Second chat write completed."),
      ],
      { write: () => response.promise },
    );
    renderWidget();
    await openWidget();
    send("Propose a first key");
    await screen.findByRole("button", { name: "Confirm change" });
    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    await selectModel();
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    send("Read in the second chat");
    await screen.findByText("Second chat is usable.");
    fireEvent.click(screen.getByRole("tab", { name: "Chat 1" }));
    expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveTextContent("Widget key");
    fireEvent.click(screen.getByRole("tab", { name: "Chat 2" }));
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 1" }));
    expect(screen.queryByRole("tab", { name: "Chat 1" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);

    send("Create the second chat key");
    fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
    await waitFor(() => expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(1));
    expect(screen.getByRole("button", { name: "Close Chat 2" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 2" }));
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "New chat" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.getByRole("button", { name: "Close Chat 2" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 2" }));
    expect(screen.getByRole("tab", { name: "Chat 2" })).toBeInTheDocument();
    await selectModel();
    send("Read while the other tab writes");
    await screen.findByText("Third chat is usable.");
    await act(async () => response.resolve(json({ key: NEW_KEY })));
    await waitFor(() => expect(screen.getByRole("button", { name: "Close Chat 2" })).toBeEnabled());
    expect(screen.getByRole("tab", { name: "Chat 3" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByText(NEW_KEY)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Chat 2" }));
    expect(await screen.findByText("Second chat write completed.")).toBeInTheDocument();
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveTextContent("Completed");
    expect(requests.filter((request) => request.url.endsWith("/key/generate")).map((request) => request.body)).toEqual([
      { key_alias: "Second chat key", team_id: "team-1", max_budget: 40 },
    ]);
    expect(screen.queryByText(/A submitted change may have completed/)).not.toBeInTheDocument();
  });

  it("closes inactive, active and last tabs without cancelling survivors or reusing closed histories", async () => {
    const first = deferred<ModelReply>();
    const second = deferred<ModelReply>();
    const requests = gateway([first.promise, second.promise, answer("Third answer."), answer("Fresh answer.")]);
    renderWidget();
    await openWidget();
    send("First pending question");
    await waitFor(() =>
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1),
    );
    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    await selectModel();
    send("Second pending question");
    await waitFor(() =>
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(2),
    );
    const pending = requests.filter((request) => request.url.endsWith("/chat/completions"));
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 1" }));
    await waitFor(() => expect(pending[0].signal.aborted).toBe(true));
    expect(pending[1].signal.aborted).toBe(false);
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    await act(async () => first.resolve(toolReply("key_create", keyArguments)));
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "LiteAdmin is working" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    await selectModel();
    send("Third question");
    await screen.findByText("Third answer.");
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 3" }));
    expect(screen.getByRole("tab", { name: "Chat 2" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Second pending question")).toBeInTheDocument();
    expect(pending[1].signal.aborted).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Close Chat 2" }));
    await waitFor(() => expect(pending[1].signal.aborted).toBe(true));
    expect(within(screen.getByRole("tablist", { name: "LiteAdmin chats" })).getAllByRole("tab")).toHaveLength(1);
    expect(screen.getByRole("tab", { name: "Chat 4" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("");
    expect(screen.queryByRole("button", { name: "Stop request" })).not.toBeInTheDocument();
    await act(async () => second.resolve(answer("Late closed answer.")));
    expect(screen.queryByText("Late closed answer.")).not.toBeInTheDocument();
    expect(screen.queryByText("Third answer.")).not.toBeInTheDocument();
    await selectModel();
    send("Fresh question");
    await screen.findByText("Fresh answer.");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Fresh question" },
    ]);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("keeps Home drafts unsent without a model or with empty or oversized input", async () => {
    const options: GatewayOptions = { models: [] };
    const requests = gateway([], options);
    const { client } = renderWidget(undefined, true);
    const home = await screen.findByRole("region", { name: "Ask LiteAdmin" });
    expect(await within(home).findByText("Add a chat model to your gateway to use LiteAdmin.")).toBeInTheDocument();
    expect(within(home).getByPlaceholderText("What would you like to do?")).toBeDisabled();
    expect(within(home).getByRole("button", { name: "Send message" })).toBeDisabled();
    options.models = [{ model_group: "chat-model", mode: "chat" }];
    await act(async () => client.invalidateQueries({ queryKey: ["liteadmin-models"] }));
    await selectModel();
    for (const text of ["   ", "x".repeat(MAX_INPUT_LENGTH + 1)]) {
      const input = within(home).getByPlaceholderText("What would you like to do?");
      fireEvent.change(input, { target: { value: text } });
      fireEvent.keyDown(input, { key: "Enter" });
      const submit = within(home).getByRole("button", { name: "Send message" });
      expect(submit).toBeDisabled();
      fireEvent.click(submit);
      expect(input).toHaveValue(text);
    }
    expect(home).toBeVisible();
    expect(within(home).getByRole("alert")).toHaveTextContent("8,000 characters");
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(0);
  });

  it("shows destination consent in Home and discards its draft when the destination changes", async () => {
    const settings = { target: EXTERNAL_INFERENCE, status: 200 };
    const settingsReady = deferred<void>();
    const requests = gateway([], {
      settings: async () => {
        await settingsReady.promise;
        return json({ PROXY_BASE_URL: MANAGEMENT, LITELLM_UI_API_DOC_BASE_URL: settings.target });
      },
    });
    const { client } = renderWidget(undefined, true);
    const home = await screen.findByRole("region", { name: "Ask LiteAdmin" });
    expect(await within(home).findByLabelText("Loading gateway settings")).toBeInTheDocument();
    await act(async () => settingsReady.resolve());
    const consent = await within(home).findByRole("button", { name: "Use configured gateway" });
    expect(within(home).getByText(EXTERNAL_INFERENCE)).toBeInTheDocument();
    expect(within(home).queryByPlaceholderText("What would you like to do?")).not.toBeInTheDocument();
    fireEvent.click(consent);
    await selectModel();
    fireEvent.change(within(home).getByPlaceholderText("What would you like to do?"), {
      target: { value: "Old gateway draft" },
    });
    settings.target = `${EXTERNAL_INFERENCE}/changed`;
    await act(async () => client.invalidateQueries({ queryKey: ["proxySettings"] }));
    fireEvent.click(await within(home).findByRole("button", { name: "Use configured gateway" }));
    await selectModel();
    expect(within(home).getByPlaceholderText("What would you like to do?")).toHaveValue("");
    expect(home).toBeVisible();
    expect(requests.some((request) => request.url.endsWith("/chat/completions"))).toBe(false);
  });

  it.each([
    ["sidebar", SidebarAccountMenu],
    ["navbar", UserDropdown],
  ] as const)("persists Hide LiteAdmin only for enterprise users in the %s account menu", async (_name, Menu) => {
    gateway([]);
    const user = userEvent.setup();
    const view = renderWidget(Menu, true);
    await screen.findByRole("button", { name: "LiteAdmin" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const toggle = await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    expect(toggle).toBeChecked();
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Ask LiteAdmin", hidden: true })).not.toBeInTheDocument();

    view.unmount();
    const restored = renderWidget(Menu, true);
    await screen.findByText("Session ready");
    await waitFor(() => expect(restored.client.isFetching()).toBe(0));
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const savedToggle = await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" });
    expect(savedToggle).toBeChecked();
    await user.click(savedToggle);
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(await screen.findByRole("region", { name: "Ask LiteAdmin" })).toBeVisible();

    for (const premiumUser of [false, null, undefined, "true"]) {
      session("proxy_admin", "first-admin", { premium_user: premiumUser });
      restored.refresh();
      expect(screen.getByRole("switch", { name: "Toggle hide all prompts" })).toBeInTheDocument();
      expect(screen.queryByRole("switch", { name: "Toggle hide LiteAdmin" })).not.toBeInTheDocument();
    }
  });

  it.each([false, null, undefined, "true"])(
    "does not expose LiteAdmin when premium_user is %s",
    async (premiumUser) => {
      session("proxy_admin", "first-admin", { premium_user: premiumUser });
      const requests = gateway([]);
      const { client } = renderWidget(undefined, true);
      await screen.findByText("Session ready");
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
      expect(screen.queryByRole("region", { name: "Ask LiteAdmin", hidden: true })).not.toBeInTheDocument();
      for (const modifiers of [{ metaKey: true }, { ctrlKey: true }]) {
        expect(fireEvent.keyDown(document, { key: "j", ...modifiers })).toBe(true);
        expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
      }
      expect(requests.every((request) => request.url.endsWith("/litellm-ui-config"))).toBe(true);
    },
  );

  it("removes an open panel and shortcuts when enterprise access is lost and restores it closed", async () => {
    const requests = gateway([]);
    const view = renderWidget();
    await openWidget();
    fireEvent.change(screen.getByPlaceholderText("Ask LiteAdmin…"), { target: { value: "Previous enterprise draft" } });
    await waitFor(() => expect(view.client.isFetching()).toBe(0));
    const requestCount = requests.length;

    session("proxy_admin", "first-admin", { premium_user: false });
    view.refresh();
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    for (const modifiers of [{ metaKey: true }, { ctrlKey: true }]) {
      expect(fireEvent.keyDown(document, { key: "j", ...modifiers })).toBe(true);
      expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    }
    expect(requests).toHaveLength(requestCount);

    session();
    view.refresh();
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: "j", ctrlKey: true });
    expect(await screen.findByRole("complementary", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("");
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

  it("keeps the page mounted and comes back closed after Hide LiteAdmin is turned off", async () => {
    gateway([]);
    pageMounts = 0;
    const user = userEvent.setup();
    renderWidget(SidebarAccountMenu);
    await screen.findByRole("button", { name: "LiteAdmin" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const toggle = await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" });
    fireEvent.keyDown(document, { key: "j", metaKey: true });
    expect(await screen.findByRole("complementary", { name: "LiteAdmin" })).toBeInTheDocument();
    await user.click(toggle);
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    await user.click(toggle);
    expect(await screen.findByRole("button", { name: "LiteAdmin" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    expect(pageMounts).toBe(1);
  });

  it("moves focus into the panel on open: the panel while loading, the message box, then a pending review", async () => {
    gateway([toolReply("key_create", keyArguments)]);
    renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    expect(screen.getByRole("complementary", { name: "LiteAdmin" })).toHaveFocus();
    await selectModel();
    const reopen = () => {
      fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
      fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    };
    reopen();
    expect(await screen.findByPlaceholderText("Ask LiteAdmin…")).toHaveFocus();
    send("Create a key for the team");
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    reopen();
    await waitFor(() => expect(review).toHaveFocus());
  });

  it("toggles with Cmd+J or Ctrl+J and ignores the shortcut while hidden", async () => {
    gateway([]);
    const user = userEvent.setup();
    renderWidget(SidebarAccountMenu);
    await screen.findByRole("button", { name: "LiteAdmin" });
    fireEvent.keyDown(document, { key: "j", metaKey: true, shiftKey: true });
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: "j", metaKey: true });
    expect(await screen.findByRole("complementary", { name: "LiteAdmin" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "LiteAdmin" })).toHaveAttribute("aria-expanded", "true");
    fireEvent.keyDown(document, { key: "j", ctrlKey: true });
    expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    await user.click(await screen.findByRole("switch", { name: "Toggle hide LiteAdmin" }));
    expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: "j", metaKey: true });
    expect(screen.queryByPlaceholderText("Ask LiteAdmin…")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "LiteAdmin" })).not.toBeInTheDocument();
  });

  it.each(["proxy_admin_viewer", "internal_user", "internal_user_viewer", "org_admin"])(
    "does not expose operations to %s",
    async (role) => {
      session(role);
      const requests = gateway([]);
      const { client } = renderWidget(undefined, true);
      await screen.findByText("Session ready");
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
      expect(screen.queryByRole("region", { name: "Ask LiteAdmin", hidden: true })).not.toBeInTheDocument();
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

  it("defaults to eligible Sonnet and preserves a manual choice across catalog refresh", async () => {
    const requests = gateway([answer("Default selected."), answer("Manual choice retained.")], {
      models: [
        { model_group: "claude-sonnet-5-5", mode: "embedding", providers: ["anthropic"] },
        { model_group: "openrouter/anthropic/claude-sonnet-5.5", mode: "chat", providers: ["openrouter"] },
        { model_group: "chat-model", mode: "chat" },
      ],
    });
    const { client } = renderWidget();
    fireEvent.click(await screen.findByRole("button", { name: "LiteAdmin" }));
    await waitFor(() => expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toBeEnabled());
    send("Use the default");
    await screen.findByText("Default selected.");
    expect(requests.find((request) => request.url.endsWith("/chat/completions"))?.body.model).toBe(
      "openrouter/anthropic/claude-sonnet-5.5",
    );
    await selectModel();
    await act(async () => client.invalidateQueries({ queryKey: ["liteadmin-models"] }));
    send("Keep my selection");
    await screen.findByText("Manual choice retained.");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.model).toBe(
      "chat-model",
    );
  });

  it("reviews the explicit team removal, sends nothing on cancel and detaches on confirmation", async () => {
    const key = "a".repeat(64);
    const proposal = toolReply("key_detach_from_team", { key });
    const requests = gateway([proposal, proposal, answer("Detached the key.")]);
    renderWidget();
    await openWidget();
    send("Remove the key from its team");
    const review = await screen.findByRole("region", {
      name: "Remove a virtual key from its team without deleting the key",
    });
    expect(review).toHaveTextContent("team id");
    expect(review).toHaveTextContent("null");
    fireEvent.click(within(review).getByRole("button", { name: "Cancel" }));
    await screen.findByText("Cancelled");
    expect(requests.filter((request) => request.url.endsWith("/key/update"))).toHaveLength(0);
    send("Detach it now");
    fireEvent.click(await screen.findByRole("button", { name: "Confirm change" }));
    await screen.findByText("Detached the key.");
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/key/update")).map(({ body }) => body)).toEqual([
      { key, team_id: null },
    ]);
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
    expect(screen.getAllByRole("complementary", { name: "LiteAdmin" })).toHaveLength(1);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    expect(within(review).getByText("Widget key")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close LiteAdmin" }));
    fireEvent.click(screen.getByRole("button", { name: "LiteAdmin" }));
    const confirm = await screen.findByRole("button", { name: "Confirm change" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    expect(screen.getByRole("button", { name: "Close Chat 1" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "New chat" })).toBeEnabled();
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

  it.each(["account", "gateway", "destination", "permission"])(
    "resets every chat when the %s changes",
    async (transition) => {
      const pending = deferred<ModelReply>();
      const settings = { target: INFERENCE, status: 200 };
      const requests = gateway([pending.promise, toolReply("key_create", keyArguments), answer("Fresh session.")], {
        settings,
      });
      const view = renderWidget(undefined, true);
      await openWidget();
      send("Old request");
      await waitFor(() =>
        expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1),
      );
      const original = requests.find((request) => request.url.endsWith("/chat/completions"));
      fireEvent.click(screen.getByRole("button", { name: "New chat" }));
      await selectModel();
      send("Old approval");
      await screen.findByRole("button", { name: "Confirm change" });
      fireEvent.click(screen.getByRole("button", { name: "New chat" }));
      await selectModel();
      fireEvent.change(screen.getByPlaceholderText("Ask LiteAdmin…"), { target: { value: "Old draft" } });
      if (transition === "account") {
        session("proxy_admin", "second-admin");
        view.refresh();
      } else if (transition === "gateway") {
        settings.target = "https://worker.test/inference";
        switchToWorkerUrl("https://worker.test/proxy");
        view.refresh();
      } else if (transition === "destination") {
        settings.target = `${INFERENCE}/updated`;
        await act(async () => view.client.invalidateQueries({ queryKey: ["proxySettings"] }));
      } else {
        session("proxy_admin_viewer", "first-admin", { premium_user: true }, "sk-session-first-admin-viewer");
        view.refresh();
        expect(screen.queryByRole("button", { name: "LiteAdmin" })).not.toBeInTheDocument();
        expect(screen.queryByRole("region", { name: "Ask LiteAdmin", hidden: true })).not.toBeInTheDocument();
        expect(screen.queryByRole("tablist", { name: "LiteAdmin chats", hidden: true })).not.toBeInTheDocument();
        fireEvent.keyDown(document, { key: "j", ctrlKey: true });
        expect(screen.queryByRole("complementary", { name: "LiteAdmin" })).not.toBeInTheDocument();
        session();
        view.refresh();
      }
      await waitFor(() => expect(original?.signal.aborted).toBe(true));
      const trigger = await screen.findByRole("button", { name: "LiteAdmin" });
      if (trigger.getAttribute("aria-expanded") === "false") fireEvent.click(trigger);
      const tabs = await screen.findByRole("tablist", { name: "LiteAdmin chats" });
      expect(within(tabs).getAllByRole("tab")).toHaveLength(1);
      expect(screen.getByRole("tab", { name: "Chat 1" })).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument());
      expect(screen.queryByText("Old request")).not.toBeInTheDocument();
      expect(screen.queryByText("Old approval")).not.toBeInTheDocument();
      expect(screen.getByPlaceholderText("Ask LiteAdmin…")).toHaveValue("");
      expect(screen.getByRole("combobox", { name: "LiteAdmin model" })).toHaveValue("");
      await act(async () => pending.resolve(toolReply("key_create", keyArguments)));
      expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
      expect(screen.queryByText(/A submitted change may have completed/)).not.toBeInTheDocument();
      expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
      await selectModel();
      send("Fresh question");
      await screen.findByText("Fresh session.");
      expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.messages).toEqual([
        { role: "system", content: expect.any(String) },
        { role: "user", content: "Fresh question" },
      ]);
    },
  );

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
    renderWidget(undefined, true);
    const home = await screen.findByRole("region", { name: "Ask LiteAdmin" });
    expect(await within(home).findByText("Could not load gateway settings.")).toBeInTheDocument();
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
