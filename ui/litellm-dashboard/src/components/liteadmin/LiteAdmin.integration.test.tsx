import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { TerminologyProvider } from "@/contexts/TerminologyContext";
import { setGlobalLitellmHeaderName, switchToWorkerUrl } from "@/components/networking";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";
import userEvent from "@testing-library/user-event";
import { Fragment, useEffect, type ComponentType } from "react";
import SidebarAccountMenu from "@/components/SidebarAccountMenu/SidebarAccountMenu";
import UserDropdown from "@/components/Navbar/UserDropdown/UserDropdown";
import { AriaUsageTrigger, LiteAdminFrame, LiteAdminPage, AriaSidebarLink } from "./LiteAdmin";
import { MAX_INPUT_LENGTH } from "./agent";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";

const { transport, push } = vi.hoisted(() => {
  const transport = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", transport);
  return { transport, push: vi.fn() };
});

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.unmock("@/lib/toast");
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

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
const toolReply = (name: string, args: Record<string, unknown>, id = "call-test"): ModelReply => ({
  role: "assistant",
  content: null,
  tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }],
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

function renderWidget(
  Menu?: ComponentType<{ onLogout: () => void }>,
  showHome = false,
  showUsage = false,
  { withTerminology = false, showPage = true }: { withTerminology?: boolean; showPage?: boolean } = {},
) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const Terminology = withTerminology ? TerminologyProvider : Fragment;
  const tree = (_home = showHome, usage = showUsage, page = showPage) => (
    <QueryClientProvider client={client}>
      <Toaster />
      <Terminology>
        <AuthProvider>
          <SessionReady />
          {Menu && <Menu onLogout={() => undefined} />}
          <LiteAdminFrame>
            <Page />
            <AriaSidebarLink active={page} />
            {page && <LiteAdminPage />}
            {usage && (
              <>
                <AriaUsageTrigger />
              </>
            )}
          </LiteAdminFrame>
        </AuthProvider>
      </Terminology>
    </QueryClientProvider>
  );
  const view = render(tree());
  return {
    ...view,
    refresh: (home = showHome, usage = showUsage, page = showPage) => view.rerender(tree(home, usage, page)),
    client,
  };
}

interface GatewayOptions {
  models?: ModelGroup[];
  callableModels?: string[] | (() => Promise<Response>);
  write?: () => Promise<Response>;
  read?: () => Promise<Response>;
  usage?: () => Promise<Response>;
  settings?: { target: string; status: number } | ((request: Request) => Promise<Response>);
  useSiTerminology?: boolean;
}

function completionResponse(message: ModelReply | undefined) {
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

function gateway(
  replies: (ModelReply | Promise<ModelReply> | ((request: RecordedRequest) => ModelReply))[],
  options: GatewayOptions = {},
) {
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
    if (path.endsWith("/litellm-ui-config")) {
      const uiConfig = {
        proxy_base_url: MANAGEMENT,
        server_root_path: "",
        admin_ui_disabled: false,
        use_si_terminology: options.useSiTerminology,
      };
      return json(uiConfig);
    }
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
    if (path.endsWith("/models")) {
      if (typeof options.callableModels === "function") return options.callableModels();
      const names = options.callableModels ??
        options.models?.map((model) => model.model_group) ?? ["a-embedding", "chat-model"];
      return json({ data: names.map((id) => ({ id })) });
    }
    if (path.endsWith("/chat/completions")) {
      const reply = replies.shift();
      return completionResponse(typeof reply === "function" ? reply(recorded) : await reply);
    }
    if (path.endsWith("/team/info"))
      return options.read ? options.read() : json({ team_info: { team_id: "team-1", max_budget: 40 } });
    if (path.endsWith("/user/daily/activity/aggregated"))
      return options.usage ? options.usage() : json({ results: [], metadata: { total_spend: 12.34 } });
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
  await user.click(await screen.findByRole("combobox", { name: "Aria Agent model" }));
  await user.click(await screen.findByRole("option", { name: model }));
}

async function openWidget() {
  await selectModel();
}

function send(text: string) {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
}

beforeEach(() => {
  pageMounts = 0;
  transport.mockReset();
  push.mockReset();
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

describe("Aria Agent in the gateway", () => {
  it.each([true, false])(
    "keeps the answer, draft and model when returning to Aria with premium=%s",
    async (premium) => {
      session("proxy_admin", "first-admin", { premium_user: premium });
      gateway([answer("Your usage is within budget.")]);
      const view = renderWidget();
      await selectModel();
      fireEvent.keyDown(document, { key: "j", metaKey: true });
      expect(screen.getByPlaceholderText("Ask Aria…")).toHaveFocus();
      send("Check my usage");
      await screen.findByText("Your usage is within budget.");
      fireEvent.change(screen.getByPlaceholderText("Ask Aria…"), { target: { value: "What changed?" } });
      view.refresh(false, false, false);
      expect(screen.queryByPlaceholderText("Ask Aria…")).not.toBeInTheDocument();
      fireEvent.keyDown(document, { key: "j", ctrlKey: true });
      expect(push).toHaveBeenCalledWith("/ui/aria");
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      view.refresh(false, false, true);
      expect(screen.getByText("Your usage is within budget.")).toBeVisible();
      expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue("What changed?");
      expect(screen.getByRole("combobox", { name: "Aria Agent model" })).toHaveValue("chat-model");
      fireEvent.keyDown(document, { key: "Escape" });
      expect(screen.getByRole("region", { name: "Aria Agent", exact: true })).toBeVisible();
    },
  );

  it("links Usage and the sidebar to one page without mounting a composer elsewhere", async () => {
    const requests = gateway([]);
    const { client } = renderWidget(undefined, false, true, { showPage: false });
    expect(await screen.findByRole("link", { name: "Ask Aria" })).toHaveAttribute("href", "/ui/aria");
    expect(screen.getByRole("link", { name: "Aria Agent" })).toHaveAttribute("href", "/ui/aria");
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(requests.some((r) => r.url.endsWith("/models") || r.url.endsWith("/sso/get/ui_settings"))).toBe(false);
  });

  it.each(["proxy_admin", "internal_user"])(
    "prefills suggestions without sending and submits the edited draft with %s permissions",
    async (role) => {
      session(role);
      const requests = gateway([answer("Here is your weekly spend.")]);
      renderWidget();
      const question = "How much have I spent this week?";
      const suggestion = await screen.findByRole("button", { name: question });
      await waitFor(() => expect(suggestion).toBeEnabled());
      act(() => fireEvent.click(suggestion));
      expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue(question);
      expect(screen.getByPlaceholderText("Ask Aria…")).toHaveFocus();
      expect(requests.filter((r) => r.url.endsWith("/chat/completions"))).toHaveLength(0);
      act(() => send(question + " Show a daily breakdown."));
      expect(await screen.findByText("Here is your weekly spend.")).toBeInTheDocument();
      for (const label of [question, "Which models are costing the most?", "How can I reduce my spend?"]) {
        expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
      }
      const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
      expect(completions).toHaveLength(1);
      expect(completions[0].body.messages).toEqual([
        { role: "system", content: expect.any(String) },
        { role: "user", content: question + " Show a daily breakdown." },
      ]);
      if (role === "internal_user") {
        expect(completions[0].body.tools).toEqual([
          expect.objectContaining({ function: expect.objectContaining({ name: "render_usage_chart" }) }),
          expect.objectContaining({ function: expect.objectContaining({ name: "get_usage_data" }) }),
        ]);
      }
      expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    },
  );

  it("should disable suggested questions when no chat model is available", async () => {
    const requests = gateway([], { models: [] });
    renderWidget();
    const picker = await screen.findByRole("combobox", { name: "Aria Agent model" });
    expect(picker).toBeEnabled();
    expect(screen.queryByText("No chat models are available for your account.")).not.toBeInTheDocument();
    await userEvent.setup().click(picker);
    expect(await screen.findByText("No available models")).toBeVisible();
    await userEvent.setup().keyboard("{Escape}");
    for (const question of [
      "How much have I spent this week?",
      "Which models are costing the most?",
      "How can I reduce my spend?",
    ]) {
      expect(screen.getByRole("button", { name: question })).toBeDisabled();
    }
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(0);
  });

  it("keeps Aria unavailable without a signed-in session", async () => {
    document.cookie = "token=; Max-Age=0; Path=/";
    const requests = gateway([]);
    const { client } = renderWidget();
    await screen.findByText("Session ready");
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(screen.queryByRole("link", { name: "Aria Agent" })).not.toBeInTheDocument();
    act(() => fireEvent.keyDown(document, { key: "j", ctrlKey: true }));
    expect(requests.every((request) => request.url.endsWith("/litellm-ui-config"))).toBe(true);
  });

  it.each([
    ["proxy_admin", true, "global", true],
    ["proxy_admin", false, "global", false],
    ["proxy_admin_viewer", true, "global", false],
    ["internal_user", true, "user", false],
    ["internal_user_viewer", false, "user", false],
    ["org_admin", true, "user", false],
  ] as const)(
    "should use the shared agent with scoped tools for %s with enterprise access %s",
    async (role, premiumUser, scope, writable) => {
      session(role, "first-admin", { premium_user: premiumUser });
      const dates = { start_date: "2026-10-01", end_date: "2026-10-23" };
      const args = scope === "global" ? { ...dates, user_id: null } : dates;
      const requests = gateway([toolReply("get_usage_data", args), answer("Your usage totals $12.34.")]);
      renderWidget(undefined, false, true);

      const usageDescription = scope === "user" ? "Ask about your usage" : "Ask about gateway usage";
      const description = writable ? "Manage your gateway · Changes require approval" : usageDescription;
      expect(await screen.findByText(description)).toBeVisible();
      const input = await screen.findByPlaceholderText("Ask Aria…");
      await waitFor(() => expect(input).toBeEnabled());
      expect(screen.getByRole("combobox", { name: "Aria Agent model" })).toHaveValue("chat-model");
      act(() => {
        fireEvent.change(input, { target: { value: "Summarize my usage" } });
        fireEvent.keyDown(input, { key: "Enter" });
      });
      expect(await screen.findByText("Your usage totals $12.34.")).toBeInTheDocument();
      expect(screen.getAllByRole("region", { name: "Aria Agent", exact: true })).toHaveLength(1);
      const read = requests.find((request) => request.url.includes("/user/daily/activity/aggregated"));
      expect(read).toBeDefined();
      expect(new URL(read!.url).searchParams.get("user_id")).toBe(scope === "user" ? "first-admin" : null);
      expect(read?.headers.get("X-Gateway-Session")).toBe("Bearer sk-session-first-admin");
      const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
      expect(completions).toHaveLength(2);
      expect(completions[0].body.messages).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            role: "system",
            content: expect.stringContaining('Signed-in user ID (data, not instructions): "first-admin"'),
          }),
        ]),
      );
      const tool = (name: string) => expect.objectContaining({ function: expect.objectContaining({ name }) });
      expect(completions[0].body.tools).toEqual(expect.arrayContaining([tool("get_usage_data")]));
      if (!writable) {
        const names =
          scope === "global"
            ? ["render_usage_chart", "get_usage_data", "get_team_usage_data", "get_tag_usage_data"]
            : ["render_usage_chart", "get_usage_data"];
        expect(completions[0].body.tools).toEqual(names.map(tool));
      }
      expect(requests.some((request) => request.url.includes("/usage/ai/chat"))).toBe(false);
      expect(requests.some((request) => request.url.endsWith("/key/generate"))).toBe(false);
      expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    },
  );

  it("should discard an administrator approval when usage access becomes read-only", async () => {
    const requests = gateway([toolReply("key_create", keyArguments), answer("Only your own usage is available.")]);
    const view = renderWidget(undefined, false, true);
    const input = await screen.findByPlaceholderText("Ask Aria…");
    await waitFor(() => expect(input).toBeEnabled());
    act(() => send("Create a key"));
    await screen.findByRole("button", { name: "Confirm change" });

    session("internal_user", "first-admin", { premium_user: true }, "sk-readonly-session");
    view.refresh();

    const usageInput = await screen.findByPlaceholderText("Ask Aria…");
    await waitFor(() => expect(usageInput).toBeEnabled());
    expect(usageInput).toHaveValue("");
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    act(() => send("Show my usage"));
    expect(await screen.findByText("Only your own usage is available.")).toBeInTheDocument();
    const completions = requests.filter((request) => request.url.endsWith("/chat/completions"));
    expect(completions.at(-1)?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Show my usage" },
    ]);
    expect(completions.at(-1)?.body.tools).toEqual([
      expect.objectContaining({ function: expect.objectContaining({ name: "render_usage_chart" }) }),
      expect.objectContaining({ function: expect.objectContaining({ name: "get_usage_data" }) }),
    ]);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("should start over with an empty history and draft while preserving the selected model", async () => {
    const requests = gateway([answer("Old answer."), answer("Fresh answer.")], {
      models: [
        { model_group: "chat-model", mode: "chat" },
        { model_group: "other-chat-model", mode: "chat" },
      ],
    });
    renderWidget(undefined, true);
    await selectModel("other-chat-model");
    act(() => send("Old question"));
    await screen.findByText("Old answer.");
    act(() => fireEvent.change(screen.getByPlaceholderText("Ask Aria…"), { target: { value: "Old draft" } }));
    await userEvent.setup().click(screen.getByRole("button", { name: "Start over" }));

    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByText("Old question")).not.toBeInTheDocument();
    expect(screen.queryByText("Old answer.")).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue("");
    expect(screen.getByRole("combobox", { name: "Aria Agent model" })).toHaveValue("other-chat-model");
    expect(screen.queryByRole("button", { name: "Start over" })).not.toBeInTheDocument();
    act(() => send("Fresh question"));
    await screen.findByText("Fresh answer.");
    const completion = requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1);
    expect(completion?.body.model).toBe("other-chat-model");
    expect(completion?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Fresh question" },
    ]);
  });

  it("should focus a pending approval when reopening the same conversation", async () => {
    const requests = gateway([toolReply("key_create", keyArguments)]);
    const view = renderWidget();
    await openWidget();
    act(() => send("Prepare a key for approval"));
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    expect(review).toHaveFocus();
    view.refresh(false, false, false);
    view.refresh(false, false, true);
    await waitFor(() => expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveFocus());
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it("should cancel a pending approval when starting over", async () => {
    const requests = gateway([toolReply("key_create", keyArguments), answer("Fresh answer.")]);
    renderWidget();
    await openWidget();
    act(() => send("Propose a key"));
    await screen.findByRole("button", { name: "Confirm change" });
    await userEvent.setup().click(screen.getByRole("button", { name: "Start over" }));
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.queryByText("Propose a key")).not.toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    act(() => send("Fresh question"));
    await screen.findByText("Fresh answer.");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Fresh question" },
    ]);
  });

  it("should abort an unfinished answer when starting over and ignore a late approval", async () => {
    const pending = deferred<ModelReply>();
    const requests = gateway([pending.promise, answer("Fresh answer.")]);
    renderWidget();
    await openWidget();
    act(() => send("Old pending question"));
    await waitFor(() =>
      expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1),
    );
    const original = requests.find((request) => request.url.endsWith("/chat/completions"));
    await userEvent.setup().click(screen.getByRole("button", { name: "Start over" }));
    expect(original?.signal.aborted).toBe(true);
    await act(async () => pending.resolve(toolReply("key_create", keyArguments)));
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.queryByText("Old pending question")).not.toBeInTheDocument();
    act(() => send("Fresh question"));
    await screen.findByText("Fresh answer.");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Fresh question" },
    ]);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
  });

  it.each([
    ["sidebar", SidebarAccountMenu],
    ["navbar", UserDropdown],
  ] as const)("persists Hide Aria Agent only for enterprise users in the %s account menu", async (_name, Menu) => {
    gateway([]);
    const user = userEvent.setup();
    const view = renderWidget(Menu, true);
    await screen.findByRole("link", { name: "Aria Agent" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const toggle = await screen.findByRole("switch", { name: "Toggle hide Aria Agent" });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    expect(toggle).toBeChecked();
    expect(screen.queryByRole("link", { name: "Aria Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Ask Aria Agent", hidden: true })).not.toBeInTheDocument();

    view.unmount();
    const restored = renderWidget(Menu, true);
    await screen.findByText("Session ready");
    await waitFor(() => expect(restored.client.isFetching()).toBe(0));
    expect(screen.queryByRole("link", { name: "Aria Agent" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    const savedToggle = await screen.findByRole("switch", { name: "Toggle hide Aria Agent" });
    expect(savedToggle).toBeChecked();
    await user.click(savedToggle);
    expect(await screen.findByRole("link", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Aria Agent" })).toBeVisible();
    expect(await screen.findByRole("region", { name: "Aria Agent", exact: true })).toBeVisible();

    for (const premiumUser of [false, null, undefined, "true"]) {
      session("proxy_admin", "first-admin", { premium_user: premiumUser });
      restored.refresh();
      expect(screen.getByRole("switch", { name: "Toggle hide all prompts" })).toBeInTheDocument();
      expect(screen.queryByRole("switch", { name: "Toggle hide Aria Agent" })).not.toBeInTheDocument();
    }
  });

  it("discards admin state and preserves scoped page access when enterprise access is lost", async () => {
    const requests = gateway([answer("Your usage remains available.")]);
    const view = renderWidget();
    await openWidget();
    fireEvent.change(screen.getByPlaceholderText("Ask Aria…"), { target: { value: "Previous enterprise draft" } });
    await waitFor(() => expect(view.client.isFetching()).toBe(0));
    session("proxy_admin", "first-admin", { premium_user: false });
    view.refresh();
    await waitFor(() => expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled());
    expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue("");
    act(() => send("Show my usage"));
    await screen.findByText("Your usage remains available.");
    const completion = requests.find((request) => request.url.endsWith("/chat/completions"));
    expect(completion?.body.tools).toEqual(
      ["render_usage_chart", "get_usage_data", "get_team_usage_data", "get_tag_usage_data"].map((name) =>
        expect.objectContaining({ function: expect.objectContaining({ name }) }),
      ),
    );

    session();
    view.refresh();
    fireEvent.keyDown(document, { key: "j", ctrlKey: true });
    expect(await screen.findByRole("region", { name: "Aria Agent", exact: true })).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue("");
  });

  it("isolates Hide Aria Agent by admin and gateway and reacts to another tab clearing it", async () => {
    gateway([]);
    const user = userEvent.setup();
    const view = renderWidget(SidebarAccountMenu);
    await screen.findByRole("link", { name: "Aria Agent" });
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    await user.click(await screen.findByRole("switch", { name: "Toggle hide Aria Agent" }));

    session("proxy_admin", "second-admin");
    view.refresh();
    expect(await screen.findByRole("link", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide Aria Agent" })).not.toBeChecked();
    session();
    view.refresh();
    expect(screen.queryByRole("link", { name: "Aria Agent" })).not.toBeInTheDocument();

    switchToWorkerUrl("https://other-gateway.test");
    view.refresh();
    expect(await screen.findByRole("link", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide Aria Agent" })).not.toBeChecked();
    switchToWorkerUrl(MANAGEMENT);
    view.refresh();
    expect(screen.queryByRole("link", { name: "Aria Agent" })).not.toBeInTheDocument();

    act(() => {
      localStorage.clear();
      window.dispatchEvent(new StorageEvent("storage", { key: null }));
    });
    expect(await screen.findByRole("link", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Toggle hide Aria Agent" })).not.toBeChecked();
  });

  it.each([
    ["sidebar", SidebarAccountMenu],
    ["navbar", UserDropdown],
  ] as const)("does not offer Hide Aria Agent to a view-only admin in the %s menu", async (_name, Menu) => {
    session("proxy_admin_viewer");
    gateway([]);
    renderWidget(Menu);
    await screen.findByText("Session ready");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /account menu/i }));
    expect(await screen.findByRole("switch", { name: "Toggle hide all prompts" })).toBeInTheDocument();
    expect(screen.queryByRole("switch", { name: "Toggle hide Aria Agent" })).not.toBeInTheDocument();
  });

  it.each(["proxy_admin_viewer", "internal_user", "internal_user_viewer", "org_admin"])(
    "exposes only scoped usage tools on the Aria page to %s",
    async (role) => {
      session(role);
      const requests = gateway([answer("Usage is available.")]);
      const { client } = renderWidget(undefined, true);
      await screen.findByText("Session ready");
      await waitFor(() => expect(client.isFetching()).toBe(0));
      expect(screen.getByRole("link", { name: "Aria Agent" })).toBeVisible();
      expect(screen.queryByRole("region", { name: "Ask Aria Agent", hidden: true })).not.toBeInTheDocument();
      await waitFor(() => expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled());
      act(() => send("Show my usage"));
      await screen.findByText("Usage is available.");
      const names =
        role === "proxy_admin_viewer"
          ? ["render_usage_chart", "get_usage_data", "get_team_usage_data", "get_tag_usage_data"]
          : ["render_usage_chart", "get_usage_data"];
      const completion = requests.find((request) => request.url.endsWith("/chat/completions"));
      expect(completion?.body.tools).toEqual(
        names.map((name) => expect.objectContaining({ function: expect.objectContaining({ name }) })),
      );
    },
  );

  it("uses the chosen model and existing session header for reads", async () => {
    const requests = gateway([toolReply("team_info", { team_id: "team-1" }), answer("The team budget is $40.")]);
    renderWidget();
    await waitFor(() => expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled());
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
    expect(within(screen.getByLabelText("Aria Agent conversation")).queryByRole("img")).not.toBeInTheDocument();
  });

  it("should render a chart from the scoped usage lookup and clear it when starting over without changing models", async () => {
    session("internal_user");
    const dates = { start_date: "2026-10-23", end_date: "2026-10-23" };
    const requests = gateway(
      [
        toolReply("get_usage_data", dates),
        (request) => {
          const body = z
            .object({ messages: z.array(z.object({ role: z.string(), content: z.string().nullish() })) })
            .parse(request.body);
          const lookup = body.messages.findLast((message) => message.role === "tool");
          if (!lookup?.content) throw new Error("The usage lookup must finish before rendering its chart.");
          const result = z
            .object({ operation: z.literal("get_usage_data"), success: z.literal(true), source_id: z.string().min(1) })
            .parse(JSON.parse(lookup.content));
          return toolReply("render_usage_chart", { source_id: result.source_id, view: "model_spend" }, "call-chart");
        },
        answer("The primary model accounts for most of your spend."),
        answer("This is a fresh conversation."),
      ],
      {
        models: [
          { model_group: "chat-model", mode: "chat" },
          { model_group: "other-chat-model", mode: "chat" },
        ],
        usage: async () =>
          json({
            results: [
              {
                date: dates.start_date,
                metrics: { spend: 4, api_requests: 3, total_tokens: 22 },
                breakdown: {
                  models: {
                    "primary-model": { metrics: { spend: 3.25, api_requests: 2, total_tokens: 18 } },
                    "fallback-model": { metrics: { spend: 0.75, api_requests: 1, total_tokens: 4 } },
                  },
                },
              },
            ],
            metadata: { total_spend: 4, total_api_requests: 3, total_tokens: 22 },
          }),
      },
    );
    renderWidget(undefined, false, true);
    const input = await screen.findByPlaceholderText("Ask Aria…");
    await waitFor(() => expect(input).toBeEnabled());
    await selectModel("other-chat-model");
    act(() => send("Which models cost the most?"));

    const reply = await screen.findByRole("article", { name: "Aria Agent answer" });
    expect(reply).toHaveTextContent("The primary model accounts for most of your spend.");
    const figure = within(reply).getByRole("figure", { name: "Spend by model" });
    const values = within(figure).getByRole("table", { name: "Spend by model values" });
    expect(within(values).getByRole("row", { name: "primary-model $3.25" })).toBeInTheDocument();
    expect(within(values).getByRole("row", { name: "fallback-model $0.75" })).toBeInTheDocument();
    expect(figure).toHaveTextContent("Your usage");
    expect(within(reply).getByRole("button", { name: "Export answer" })).toBeEnabled();
    expect(within(reply).queryByRole("button", { name: /copy/i })).not.toBeInTheDocument();
    const reads = requests.filter((request) => request.url.includes("/user/daily/activity/aggregated"));
    expect(reads).toHaveLength(1);
    expect(new URL(reads[0].url).searchParams.get("user_id")).toBe("first-admin");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(3);

    await userEvent.setup().click(screen.getByRole("button", { name: "Start over" }));
    expect(screen.queryByRole("figure", { name: "Spend by model" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export answer" })).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Aria Agent model" })).toHaveValue("other-chat-model");
    act(() => send("Start fresh"));
    expect(await screen.findByText("This is a fresh conversation.")).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
    const freshRequest = requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1);
    expect(freshRequest?.body.model).toBe("other-chat-model");
    expect(freshRequest?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Start fresh" },
    ]);
  });

  it("should render a usage-only assistant's Markdown table as accessible headers and cells", async () => {
    session("internal_user");
    const markdown = [
      "## Usage by model",
      "",
      "> Spend is shown in USD.",
      "",
      "| Model | Requests | Spend |",
      "| :--- | ---: | ---: |",
      "| primary-model | 12 | $3.40 |",
      "| fallback-model | 5 | $0.60 |",
    ].join("\n");
    gateway([answer(markdown), answer("The primary model accounts for $3.40 of your $4.00 spend.")]);
    renderWidget(undefined, false, true);
    const input = await screen.findByPlaceholderText("Ask Aria…");
    await waitFor(() => expect(input).toBeEnabled());
    act(() => send("Show my usage by model as a table"));
    const conversation = screen.getByLabelText("Aria Agent conversation");
    const table = await within(conversation).findByRole("table");
    expect(within(conversation).getByRole("heading", { name: "Usage by model", level: 2 })).toBeInTheDocument();
    expect(within(conversation).getByRole("article", { name: "Your question" })).toHaveTextContent(
      "Show my usage by model as a table",
    );
    expect(within(conversation).getByRole("article", { name: "Aria Agent answer" })).toHaveTextContent(
      "Spend is shown in USD.",
    );
    for (const header of ["Model", "Requests", "Spend"]) {
      expect(within(table).getByRole("columnheader", { name: header })).toBeInTheDocument();
    }
    expect(within(table).getAllByRole("row")).toHaveLength(3);
    for (const values of [
      ["primary-model", "12", "$3.40"],
      ["fallback-model", "5", "$0.60"],
    ]) {
      const row = within(table).getByRole("row", { name: values.join(" ") });
      for (const value of values) {
        expect(within(row).getByRole("cell", { name: value })).toBeInTheDocument();
      }
    }
    expect(within(conversation).queryByText(/\|.*Model.*\|/)).not.toBeInTheDocument();
    act(() => send("Which model is driving the cost?"));
    expect(
      await within(conversation).findByText("The primary model accounts for $3.40 of your $4.00 spend."),
    ).toBeInTheDocument();
    expect(within(conversation).getByRole("table")).toBeInTheDocument();
    expect(within(conversation).getAllByRole("article", { name: "Your question" })).toHaveLength(2);
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
    await waitFor(() => expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled());
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

  it("should let an admin viewer select only models their inference session can call", async () => {
    session("proxy_admin_viewer");
    const options: GatewayOptions = {
      models: [
        { model_group: "claude-sonnet-5-5", mode: "chat", providers: ["anthropic"] },
        { model_group: "allowed-chat", mode: "chat" },
        { model_group: "allowed-embedding", mode: "embedding" },
      ],
      callableModels: ["allowed-chat", "allowed-embedding"],
    };
    const requests = gateway([answer("Your usage is available.")], options);
    const { client } = renderWidget(undefined, false, true);
    const user = userEvent.setup();
    const input = await screen.findByPlaceholderText("Ask Aria…");
    await waitFor(() => expect(input).toBeEnabled());
    const picker = screen.getByRole("combobox", { name: "Aria Agent model" });
    expect(picker).toHaveValue("allowed-chat");
    await user.click(picker);
    expect(screen.getByRole("option", { name: "allowed-chat" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "claude-sonnet-5-5" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "allowed-embedding" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    act(() => send("Show our usage"));
    await screen.findByText("Your usage is available.");
    expect(requests.find((request) => request.url.endsWith("/chat/completions"))?.body.model).toBe("allowed-chat");
    const discovery = requests.find((request) => request.url === `${INFERENCE}/models`);
    expect(discovery?.headers.get("X-Gateway-Session")).toBe("Bearer sk-session-first-admin");
    options.callableModels = [];
    await act(async () => client.invalidateQueries({ queryKey: ["liteadmin-models"] }));
    await user.click(screen.getByRole("combobox", { name: "Aria Agent model" }));
    expect(await screen.findByText("No available models")).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.getByPlaceholderText("Ask Aria…")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(1);
  });

  it("should keep viewer prompts disabled when inference model discovery fails", async () => {
    session("proxy_admin_viewer");
    const requests = gateway([], { callableModels: async () => json({ error: { message: "Forbidden" } }, 403) });
    renderWidget(undefined, false, true);
    expect(await screen.findByText(/Could not load models/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Ask Aria…")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(0);
  });

  it("should send a callable team model's public name and respect its inference mode", async () => {
    const requests = gateway([answer("The team model answered.")], {
      models: [
        { model_group: "model_name_team1_def", mode: "chat" },
        { model_group: "renamed-model", mode: "chat" },
      ],
      callableModels: async () =>
        json({
          data: [
            { id: "team-chat", mode: "chat" },
            { id: "renamed-model", mode: "embedding" },
          ],
        }),
    });
    renderWidget(undefined, false, true);
    await waitFor(() => expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled());
    const picker = screen.getByRole("combobox", { name: "Aria Agent model" });
    expect(picker).toHaveValue("team-chat");
    const user = userEvent.setup();
    await user.click(picker);
    expect(screen.getByRole("option", { name: "team-chat" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "model_name_team1_def" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "renamed-model" })).not.toBeInTheDocument();
    await user.keyboard("{Escape}");
    act(() => send("Show our usage"));
    await screen.findByText("The team model answered.");
    expect(requests.find((request) => request.url.endsWith("/chat/completions"))?.body.model).toBe("team-chat");
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

  it("keeps one action across route navigation and one confirmed write", async () => {
    const response = deferred<Response>();
    const proposed = { ...toolReply("key_create", keyArguments), content: "Here is the requested change." };
    const requests = gateway([proposed, answer("Created the key."), answer("You are welcome.")], {
      write: () => response.promise,
    });
    const view = renderWidget();
    await openWidget();
    send("Create a key for the team");
    const review = await screen.findByRole("region", { name: "Create a virtual key" });
    expect(screen.getAllByRole("region", { name: "Aria Agent", exact: true })).toHaveLength(1);
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    expect(within(review).getByText("Widget key")).toBeInTheDocument();
    view.refresh(false, false, false);
    view.refresh(false, false, true);
    const confirm = await screen.findByRole("button", { name: "Confirm change" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    expect(screen.getByRole("button", { name: "Start over" })).toBeDisabled();
    act(() => fireEvent.click(screen.getByRole("button", { name: "Start over" })));
    expect(screen.getByRole("region", { name: "Create a virtual key" })).toHaveTextContent("Applying");
    expect(screen.queryByRole("button", { name: "Stop request" })).not.toBeInTheDocument();
    view.refresh(false, false, false);
    view.refresh(false, false, true);
    await act(async () => response.resolve(json({ key: NEW_KEY })));
    expect(await screen.findByText("Created the key.")).toBeInTheDocument();
    expect(screen.getByText(NEW_KEY)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy generated key" })).toBeInTheDocument();
    expect(screen.getAllByRole("region", { name: "Create a virtual key" })).toHaveLength(1);
    expect(screen.getByLabelText("Aria Agent conversation")).toHaveTextContent(
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

  it.each(
    ["account", "gateway", "destination", "permission"].flatMap((transition) =>
      ["pending", "approval", "draft"].map((stage) => [transition, stage] as const),
    ),
  )("should clear the conversation when %s changes during %s", async (transition, stage) => {
    const pending = deferred<ModelReply>();
    const settings = { target: INFERENCE, status: 200 };
    const replies = [
      answer("Old answer."),
      ...(stage === "pending" ? [pending.promise] : []),
      ...(stage === "approval" ? [toolReply("key_create", keyArguments)] : []),
      answer("Fresh session."),
    ];
    const requests = gateway(replies, { settings });
    const view = renderWidget(undefined, true);
    await openWidget();
    act(() => send("Old question"));
    await screen.findByText("Old answer.");
    if (stage === "pending") {
      act(() => send("Old pending request"));
      await waitFor(() =>
        expect(requests.filter((request) => request.url.endsWith("/chat/completions"))).toHaveLength(2),
      );
    } else if (stage === "approval") {
      act(() => send("Old approval"));
      await screen.findByRole("button", { name: "Confirm change" });
    } else {
      act(() => fireEvent.change(screen.getByPlaceholderText("Ask Aria…"), { target: { value: "Old draft" } }));
    }
    const original = requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1);
    if (transition === "account") {
      session("proxy_admin", "second-admin");
      view.refresh();
    } else if (transition === "gateway") {
      settings.target = "https://worker.test/inference";
      switchToWorkerUrl("https://worker.test/proxy");
      view.refresh();
    } else if (transition === "destination") {
      settings.target = INFERENCE + "/updated";
      await act(async () => view.client.invalidateQueries({ queryKey: ["proxySettings"] }));
    } else {
      session("proxy_admin_viewer", "first-admin", { premium_user: true }, "sk-session-first-admin-viewer");
      view.refresh();
      expect(await screen.findByRole("link", { name: "Aria Agent" })).toBeVisible();
      session();
      view.refresh();
    }
    if (stage === "pending") await waitFor(() => expect(original?.signal.aborted).toBe(true));
    await screen.findByRole("region", { name: "Aria Agent", exact: true });
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.queryByText("Old question")).not.toBeInTheDocument();
    expect(screen.queryByText("Old answer.")).not.toBeInTheDocument();
    expect(screen.queryByText("Old approval")).not.toBeInTheDocument();
    expect(await screen.findByPlaceholderText("Ask Aria…")).toHaveValue("");
    await selectModel();
    if (stage === "pending") await act(async () => pending.resolve(toolReply("key_create", keyArguments)));
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(screen.queryByText(/A submitted change may have completed/)).not.toBeInTheDocument();
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toHaveLength(0);
    act(() => send("Fresh question"));
    await screen.findByText("Fresh session.");
    expect(requests.filter((request) => request.url.endsWith("/chat/completions")).at(-1)?.body.messages).toEqual([
      { role: "system", content: expect.any(String) },
      { role: "user", content: "Fresh question" },
    ]);
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
    const home = await screen.findByRole("region", { name: "Aria Agent", exact: true });
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
    fireEvent.click(await screen.findByRole("button", { name: "Use configured gateway" }));
    await selectModel();
    send("Check the original gateway");
    await screen.findByText("Original gateway.");

    act(() => {
      switchToWorkerUrl(worker);
      view.refresh();
    });
    await settingsRequested.promise;
    expect(screen.queryByText(EXTERNAL_INFERENCE)).not.toBeInTheDocument();
    expect(await screen.findByLabelText("Loading gateway settings")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Ask Aria…")).not.toBeInTheDocument();
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
    expect(screen.getByPlaceholderText("Ask Aria…")).toBeEnabled();
    expect(screen.getByPlaceholderText("Ask Aria…")).toHaveValue(draft);
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("8,000 characters");
    expect(screen.queryByLabelText("Aria Agent conversation")).not.toBeInTheDocument();
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
