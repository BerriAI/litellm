import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useEffect } from "react";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import LiteAskWidget from "./LiteAskWidget";
import type { LiteAskConfig, LiteAskResponse } from "./api";

const session = (role = "proxy_admin", identity = "one") => {
  const claims = {
    user_role: role,
    user_id: `admin-${identity}`,
    key: `session-${identity}`,
    auth_header_name: "x-test-auth",
    exp: Math.floor(Date.now() / 1000) + 3600,
  };
  return `e30.${btoa(JSON.stringify(claims)).replace(/=/g, "")}.sig`;
};

const response = (changes: Partial<LiteAskResponse> = {}): LiteAskResponse => ({
  message: "This month’s spend is $42.",
  proposal: null,
  result: null,
  generated_key: null,
  ...changes,
});
const proposalResponse = () =>
  response({
    message: "Review the team budget update.",
    proposal: {
      token: "sealed-approval",
      tool: "update_team",
      title: "Set the Product team budget to $50",
      arguments: { team_id: "team-one", max_budget: 50 },
      expires_at: Math.floor(Date.now() / 1000) + 600,
    },
  });
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const fetchMock = vi.fn<typeof fetch>();
let config: LiteAskConfig;
let chat: () => Promise<Response>;
let approve: () => Promise<Response>;

function ChangeSession({ value }: { value?: string | null }) {
  const { setToken } = useAuth();
  useEffect(() => {
    if (value !== undefined) setToken(value);
  }, [value, setToken]);
  return null;
}

function Harness({ nextSession }: { nextSession?: string | null }) {
  return (
    <AuthProvider>
      <ChangeSession value={nextSession} />
      <LiteAskWidget />
    </AuthProvider>
  );
}

const requests = (endpoint: string) =>
  fetchMock.mock.calls.filter(([input]) => String(input).endsWith(`/liteask/${endpoint}`));
const bodyOf = (endpoint: string, index = 0) => JSON.parse(String(requests(endpoint)[index]?.[1]?.body));

async function openChat() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("button", { name: "LiteAsk" }));
  expect(await screen.findByRole("dialog", { name: "LiteAsk" })).toBeVisible();
  return user;
}

async function sendMessage(text = "Show this month’s spend") {
  fireEvent.change(screen.getByRole("textbox", { name: "Message LiteAsk" }), { target: { value: text } });
  await userEvent.click(screen.getByRole("button", { name: "Send message" }));
}

describe("LiteAsk dashboard conversation", () => {
  beforeEach(() => {
    config = { enabled: true, model: "gateway-chat", can_execute_mutations: true };
    chat = async () => json(response());
    approve = async () =>
      json(response({ message: "Budget updated.", result: { team_id: "team-one", max_budget: 50 } }));
    document.cookie = `token=${session()}; Path=/`;
    fetchMock.mockReset();
    fetchMock.mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("litellm-ui-config"))
        return json({ server_root_path: "/edge", proxy_base_url: "https://gateway.example/edge" });
      if (url.endsWith("/liteask/config")) return json(config);
      if (url.endsWith("/liteask/chat")) return chat();
      if (url.endsWith("/liteask/approve")) return approve();
      return json({ error: "unexpected request" }, 404);
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    document.cookie = "token=; Max-Age=0; Path=/";
    vi.unstubAllGlobals();
  });

  it("uses the current dashboard session and gateway base path, and returns keyboard focus on close", async () => {
    render(<Harness />);
    const user = await openChat();
    await sendMessage();
    expect(await screen.findByText("This month’s spend is $42.")).toBeVisible();
    expect(requests("chat")[0]?.[0]).toBe("https://gateway.example/edge/management/v1/liteask/chat");
    expect(requests("chat")[0]?.[1]?.headers).toMatchObject({ "x-test-auth": "Bearer session-one" });
    expect(bodyOf("chat")).toMatchObject({ messages: [{ role: "user", content: "Show this month’s spend" }] });
    expect(bodyOf("chat").conversation_id).toMatch(/^[0-9a-f-]{36}$/);
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "LiteAsk" })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "LiteAsk" })).toHaveFocus();
  });

  it.each(["proxy_admin_viewer", "org_admin", "internal_user"])(
    "never requests LiteAsk config for %s",
    async (role) => {
      document.cookie = `token=${session(role)}; Path=/`;
      render(<Harness />);
      await act(async () => {});
      expect(requests("config")).toHaveLength(0);
      expect(screen.queryByRole("button", { name: "LiteAsk" })).not.toBeInTheDocument();
    },
  );

  it("waits for enabled backend configuration before showing the widget", async () => {
    config = { enabled: false, model: null, can_execute_mutations: false };
    render(<Harness />);
    await waitFor(() => expect(requests("config")).toHaveLength(1));
    expect(screen.queryByRole("button", { name: "LiteAsk" })).not.toBeInTheDocument();
  });

  it("requires an explicit confirmation and sends only the sealed proposal with its conversation", async () => {
    chat = async () => json(proposalResponse());
    render(<Harness />);
    const user = await openChat();
    await sendMessage("Set Product’s budget to $50");
    const review = await screen.findByRole("region", { name: "Review change" });
    expect(review).toHaveTextContent('"max_budget": 50');
    expect(requests("approve")).toHaveLength(0);
    expect(screen.getByRole("textbox", { name: "Message LiteAsk" })).toBeDisabled();
    await user.click(within(review).getByRole("button", { name: "Confirm change" }));
    expect(await screen.findByText("Budget updated.")).toBeVisible();
    expect(bodyOf("approve")).toEqual({ conversation_id: bodyOf("chat").conversation_id, token: "sealed-approval" });
    expect(screen.queryByRole("region", { name: "Review change" })).not.toBeInTheDocument();
  });

  it("cancels a proposed change without executing it", async () => {
    chat = async () => json(proposalResponse());
    render(<Harness />);
    const user = await openChat();
    await sendMessage("Change a budget");
    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    expect(screen.getByText("Change canceled. Nothing was applied.")).toBeVisible();
    expect(requests("approve")).toHaveLength(0);
    expect(screen.getByRole("textbox", { name: "Message LiteAsk" })).toBeEnabled();
  });

  it("keeps an in-flight approval attached to its conversation until the result arrives", async () => {
    let finish: (value: Response) => void = () => {};
    chat = async () => json(proposalResponse());
    approve = () =>
      new Promise((resolve) => {
        finish = resolve;
      });
    render(<Harness />);
    const user = await openChat();
    await sendMessage("Change a budget");
    await user.click(await screen.findByRole("button", { name: "Confirm change" }));
    expect(screen.getByRole("button", { name: "New chat" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "New chat" }));
    expect(requests("approve")[0]?.[1]?.signal?.aborted).toBe(false);
    await user.keyboard("{Escape}");
    await openChat();
    expect(screen.getByRole("button", { name: "New chat" })).toBeDisabled();
    await act(async () => finish(json(response({ message: "Budget updated." }))));
    expect(await screen.findByText("Budget updated.")).toBeVisible();
    expect(screen.getByRole("button", { name: "New chat" })).toBeEnabled();
  });

  it("prevents approving an expired proposal", async () => {
    const proposed = proposalResponse();
    chat = async () =>
      json({ ...proposed, proposal: { ...proposed.proposal, expires_at: Math.floor(Date.now() / 1000) - 1 } });
    render(<Harness />);
    await openChat();
    await sendMessage("Change a budget");
    expect(await screen.findByRole("button", { name: "Confirm change" })).toBeDisabled();
    expect(screen.getByText("This approval expired. Ask for the change again.")).toBeVisible();
    expect(requests("approve")).toHaveLength(0);
  });

  it("removes a failed approval so it cannot be retried automatically", async () => {
    chat = async () => json(proposalResponse());
    approve = async () => {
      throw new TypeError("Network disconnected");
    };
    render(<Harness />);
    const user = await openChat();
    await sendMessage("Change a budget");
    await user.click(await screen.findByRole("button", { name: "Confirm change" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Check audit logs before trying this change again");
    expect(screen.queryByRole("button", { name: "Confirm change" })).not.toBeInTheDocument();
    expect(requests("approve")).toHaveLength(1);
  });

  it("keeps generated keys out of subsequent history and removes them after the next message", async () => {
    chat = async () => json(proposalResponse());
    approve = async () => json(response({ message: "Key created.", generated_key: "new-private-key" }));
    render(<Harness />);
    const user = await openChat();
    await sendMessage("Create a key");
    await user.click(await screen.findByRole("button", { name: "Confirm change" }));
    expect(await screen.findByRole("region", { name: "New API key" })).toHaveTextContent("new-private-key");
    chat = async () => json(response());
    await sendMessage("Check the team budget");
    expect(await screen.findByText("This month’s spend is $42.")).toBeVisible();
    expect(JSON.stringify(bodyOf("chat", 1))).not.toContain("new-private-key");
    expect(screen.queryByRole("region", { name: "New API key" })).not.toBeInTheDocument();
  });

  it("aborts a new-chat reset and discards a late response from the old conversation", async () => {
    let finish: (value: Response) => void = () => {};
    chat = () =>
      new Promise((resolve) => {
        finish = resolve;
      });
    render(<Harness />);
    const user = await openChat();
    await sendMessage();
    const signal = requests("chat")[0]?.[1]?.signal;
    await user.click(screen.getByRole("button", { name: "New chat" }));
    expect(signal?.aborted).toBe(true);
    await act(async () => finish(json(response({ message: "Old private data" }))));
    expect(screen.queryByText("Old private data")).not.toBeInTheDocument();
    chat = async () => json(response());
    await sendMessage("New question");
    expect(await screen.findByText("This month’s spend is $42.")).toBeVisible();
    expect(bodyOf("chat", 1).conversation_id).not.toBe(bodyOf("chat").conversation_id);
    expect(bodyOf("chat", 1).messages).toEqual([{ role: "user", content: "New question" }]);
  });

  it("discards an old admin’s response when the signed-in account changes", async () => {
    let finish: (value: Response) => void = () => {};
    chat = () =>
      new Promise((resolve) => {
        finish = resolve;
      });
    const view = render(<Harness />);
    await openChat();
    await sendMessage();
    view.rerender(<Harness nextSession={session("proxy_admin", "two")} />);
    await waitFor(() => expect(requests("chat")[0]?.[1]?.signal?.aborted).toBe(true));
    await act(async () => finish(json(response({ message: "Old admin data", generated_key: "old-admin-secret" }))));
    await openChat();
    expect(screen.queryByText("Old admin data")).not.toBeInTheDocument();
    expect(screen.queryByText("old-admin-secret")).not.toBeInTheDocument();
    chat = async () => json(response());
    await sendMessage("New admin request");
    expect(await screen.findByText("This month’s spend is $42.")).toBeVisible();
    expect(requests("chat")[1]?.[1]?.headers).toMatchObject({ "x-test-auth": "Bearer session-two" });
  });

  it("clears the conversation when the backend revokes admin access", async () => {
    chat = async () => json({ error: "forbidden" }, 403);
    render(<Harness />);
    await openChat();
    await sendMessage();
    await waitFor(() => expect(screen.queryByRole("button", { name: "LiteAsk" })).not.toBeInTheDocument());
    expect(screen.queryByRole("dialog", { name: "LiteAsk" })).not.toBeInTheDocument();
  });

  it("hides all conversation data on logout even while AuthContext retains its old access token", async () => {
    const view = render(<Harness />);
    await openChat();
    await sendMessage();
    expect(await screen.findByText("This month’s spend is $42.")).toBeVisible();
    view.rerender(<Harness nextSession={null} />);
    expect(screen.queryByRole("dialog", { name: "LiteAsk" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "LiteAsk" })).not.toBeInTheDocument();
    expect(screen.queryByText("This month’s spend is $42.")).not.toBeInTheDocument();
  });

  it("renders model content without active HTML, links, or remote images", async () => {
    chat = async () =>
      json(
        response({
          message:
            'Safe **answer** ![tracking](https://untrusted.example/pixel) [click](https://untrusted.example/) <img src="https://untrusted.example/html">',
        }),
      );
    render(<Harness />);
    await openChat();
    await sendMessage();
    expect(await screen.findByText("answer")).toBeVisible();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("explains when the gateway supports only reads", async () => {
    config = { ...config, can_execute_mutations: false };
    render(<Harness />);
    await openChat();
    expect(screen.getByText("Read-only mode. Changes are currently unavailable.")).toBeVisible();
  });
});
