import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import type { Conversation } from "@/components/chat/types";
import ChatConversationPage from "./page";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));

const storageKey = "litellm_chat_history_v1:background-user";
const requests: Record<string, unknown>[] = [];
let supported: boolean;
let status: "in_progress" | "completed" | "cancelled";
let retrievalStatus: number;
let cancellations: number;
const saved = (): Conversation[] => JSON.parse(localStorage.getItem(storageKey) ?? "[]");
const result = () => ({
  id: "litellm_poll_research",
  status,
  output:
    status === "completed"
      ? [
          {
            type: "mcp_call",
            id: "call_inventory",
            name: "lookup_inventory",
            arguments: "{}",
            output: "12 widgets",
            server_label: "inventory",
          },
          { type: "message", content: [{ type: "output_text", text: "There are 12 widgets in stock." }] },
        ].filter((item) => item.type !== "mcp_call" || Boolean(requests.at(-1)?.tools))
      : [],
});

function renderChat(id?: string) {
  return renderWithProviders(
    <ChatShellProvider
      accessToken="test-token"
      userId="background-user"
      userEmail="researcher@example.com"
      userRole="Internal Viewer"
      premiumUser={false}
    >
      <ChatConversationPage />
    </ChatShellProvider>,
    { searchParams: id ? { id } : {} },
  );
}

beforeEach(() => {
  localStorage.clear();
  testQueryClient.clear();
  requests.length = 0;
  cancellations = 0;
  supported = true;
  retrievalStatus = 200;
  status = "in_progress";
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request =
        input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
      const path = new URL(request.url).pathname;
      if (path.endsWith("/model_group/info"))
        return Response.json({
          data: [{ model_group: "research-chat", mode: "chat", supports_background_chat: supported }],
        });
      if (path.endsWith("/v1/mcp/server")) return Response.json([{ server_id: "inventory", server_name: "inventory" }]);
      if (path.endsWith("/mcp-rest/tools/list")) return Response.json({ tools: [{ name: "lookup_inventory" }] });
      if (path.endsWith("/responses")) {
        const body = JSON.parse(await request.text());
        requests.push(body);
        if (body.background) return Response.json({ ...result(), status: "queued" });
        return new Response(
          'data: {"type":"response.output_text.delta","delta":"Foreground answer"}\n\ndata: {"type":"response.completed","response":{"id":"resp_foreground","output":[]}}\n\n',
          { headers: { "Content-Type": "text/event-stream" } },
        );
      }
      if (path.endsWith("/litellm_poll_research/cancel")) {
        cancellations++;
        status = "cancelled";
        return Response.json(result());
      }
      if (path.endsWith("/litellm_poll_research"))
        return Response.json(retrievalStatus === 200 ? result() : { error: { message: "Unavailable" } }, {
          status: retrievalStatus,
        });
      return Response.json({}, { status: 404 });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

async function startResearch(withTools = true) {
  const user = userEvent.setup();
  await screen.findByRole("button", { name: "research-chat" });
  if (withTools) {
    await user.click(screen.getByRole("button", { name: "Connect tools" }));
    await user.click(await screen.findByRole("switch"));
    await user.keyboard("{Escape}");
  }
  fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
    target: { value: "Check inventory" },
  });
  await user.click(screen.getByRole("button", { name: "Send", exact: true }));
}

it("should retrieve the completed tool run after closing and reopening without submitting it again", async () => {
  const first = renderChat();
  await startResearch();
  await screen.findByText("Running in the background. You can close this tab and return to this chat.");
  await waitFor(() => expect(saved()[0]?.messages.at(-1)?.backgroundRun?.id).toBe("litellm_poll_research"));
  const id = saved()[0].id;
  first.unmount();
  expect(cancellations).toBe(0);
  status = "completed";
  renderChat(id);
  expect(await screen.findByText("There are 12 widgets in stock.")).toBeVisible();
  expect(screen.getByRole("button", { name: "lookup_inventory" })).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    background: true,
    stream: false,
    tools: [{ type: "mcp", server_label: "inventory" }],
  });
  expect(screen.queryByRole("button", { name: "Stop request" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
    target: { value: "What was the result?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send", exact: true }));
  await waitFor(() => expect(requests).toHaveLength(2));
  expect(requests[1]).toMatchObject({ background: true, stream: false });
  expect(requests[1]).not.toHaveProperty("previous_response_id");
  expect(requests[1].input).toEqual(
    expect.arrayContaining([{ type: "function_call_output", call_id: "call_inventory", output: "12 widgets" }]),
  );
});

it("should continue an ordinary chat without MCP tools after closing and reopening", async () => {
  const first = renderChat();
  await startResearch(false);
  await screen.findByText("Running in the background. You can close this tab and return to this chat.");
  await waitFor(() => expect(saved()[0]?.messages.at(-1)?.backgroundRun?.id).toBe("litellm_poll_research"));
  expect(requests[0]).toMatchObject({ background: true, stream: false });
  expect(requests[0]).not.toHaveProperty("tools");
  const id = saved()[0].id;
  first.unmount();
  status = "completed";
  renderChat(id);
  expect(await screen.findByText("There are 12 widgets in stock.")).toBeVisible();
  expect(requests).toHaveLength(1);
  expect(cancellations).toBe(0);
});

it("should cancel a background run only when Stop is clicked", async () => {
  renderChat();
  await startResearch();
  await screen.findByText("Running in the background. You can close this tab and return to this chat.");
  fireEvent.click(screen.getByRole("button", { name: "Stop request" }));
  expect(await screen.findByText("Cancellation requested")).toBeVisible();
  expect(cancellations).toBe(1);
});

it("should retain ordinary streaming when background support is unavailable", async () => {
  supported = false;
  renderChat();
  await startResearch();
  expect(await screen.findByText("Foreground answer")).toBeVisible();
  expect(requests[0]).toMatchObject({ stream: true });
  expect(requests[0]).not.toHaveProperty("background");
  expect(screen.getByText("Keep this tab open. Background runs are not available for this chat.")).toBeVisible();
});

it("should let the user reconnect after an authentication failure without resubmitting work", async () => {
  const first = renderChat();
  await startResearch();
  await waitFor(() => expect(saved()[0]?.messages.at(-1)?.backgroundRun?.id).toBe("litellm_poll_research"));
  const id = saved()[0].id;
  first.unmount();
  retrievalStatus = 401;
  renderChat(id);
  await screen.findByRole("button", { name: "Reconnect" });
  retrievalStatus = 200;
  status = "completed";
  fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
  expect(await screen.findByText("There are 12 widgets in stock.")).toBeVisible();
  expect(requests).toHaveLength(1);
});
