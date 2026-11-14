import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
import type { Conversation, MCPEvent } from "@/components/chat/types";
import ChatConversationPage from "./page";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));

const userId = "mcp-replay-user";
const storageKey = `litellm_chat_history_v1:${userId}`;
const requests: Record<string, unknown>[] = [];
let providers: string[];
const toolEvent: MCPEvent = {
  type: "response.output_item.done",
  item: {
    type: "mcp_call",
    id: "inventory-call",
    name: "lookup_inventory",
    arguments: '{"item":"widgets"}',
    output: '{"quantity":12}',
  },
};
const toolPair = [
  { type: "function_call", call_id: "inventory-call", name: "lookup_inventory", arguments: '{"item":"widgets"}' },
  { type: "function_call_output", call_id: "inventory-call", output: '{"quantity":12}' },
];

function renderChat(conversationId?: string) {
  return renderWithProviders(
    <ChatShellProvider
      accessToken="sk-test"
      userId={userId}
      userEmail="researcher@example.com"
      userRole="Internal Viewer"
      premiumUser={false}
    >
      <ChatConversationPage />
    </ChatShellProvider>,
    { searchParams: conversationId ? { id: conversationId } : {} },
  );
}

async function send(text: string) {
  const count = requests.length;
  act(() =>
    fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), { target: { value: text } }),
  );
  act(() => fireEvent.click(screen.getByRole("button", { name: "Send", exact: true })));
  await waitFor(() => expect(requests).toHaveLength(count + 1));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Stop request" })).not.toBeInTheDocument());
}

beforeEach(() => {
  localStorage.clear();
  testQueryClient.clear();
  requests.length = 0;
  providers = ["openai"];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request =
        input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
      const pathname = new URL(request.url).pathname;
      if (pathname.endsWith("/model_group/info"))
        return Response.json({
          data: [
            { model_group: "research-chat", providers, mode: "chat" },
            { model_group: "second-chat", providers: ["openai"], mode: "chat" },
          ],
        });
      if (pathname.endsWith("/responses")) {
        requests.push(JSON.parse(await request.text()));
        return new Response(
          [
            { type: "response.output_text.delta", delta: `Answer ${requests.length}` },
            { type: "response.completed", response: { id: `response-${requests.length}`, output: [] } },
          ]
            .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
            .join(""),
          {
            headers: { "Content-Type": "text/event-stream" },
          },
        );
      }
      return Response.json({}, { status: 404 });
    }),
  );
});

afterEach(() => vi.unstubAllGlobals());

describe("chat conversation replay", () => {
  it("should keep native response chaining for ordinary follow-up messages", async () => {
    renderChat();
    expect(await screen.findByRole("button", { name: "research-chat" })).toBeVisible();
    await send("Remember papaya");
    await send("What did I ask you to remember?");
    expect(requests[1].previous_response_id).toBe("response-1");
    expect(requests[1].input).toEqual([{ type: "message", role: "user", content: "What did I ask you to remember?" }]);
  });

  it("should preserve saved tool-only turns and image attachments when continuing and editing a conversation", async () => {
    const conversation: Conversation = {
      id: "saved-conversation",
      title: "Inventory",
      model: "research-chat",
      createdAt: 1,
      updatedAt: 1,
      mcpServerNames: [],
      messages: [
        {
          id: "image-message",
          role: "user",
          content: "Find this item",
          timestamp: 1,
          attachments: [
            { id: "saved-image", name: "item.png", url: "https://example.com/item.png", mimeType: "image/png" },
          ],
        },
        { id: "tool-message", role: "assistant", content: "", timestamp: 2, mcpEvents: [toolEvent] },
      ],
    };
    localStorage.setItem(storageKey, JSON.stringify([conversation]));
    renderChat(conversation.id);
    await screen.findByRole("button", { name: "research-chat" });
    expect(screen.getByRole("button", { name: "lookup_inventory" })).toHaveAttribute("aria-expanded", "true");
    await send("What was the quantity?");
    const imageMessage = {
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: "Find this item" },
        { type: "input_image", image_url: "https://example.com/item.png", detail: "auto" },
      ],
    };
    expect(requests[0].input).toEqual([
      imageMessage,
      ...toolPair,
      { type: "message", role: "user", content: "What was the quantity?" },
    ]);
    expect(requests[0]).not.toHaveProperty("tools");
    const user = userEvent.setup();
    await user.hover(screen.getByText("What was the quantity?"));
    await user.click(screen.getAllByRole("button", { name: "Edit message" }).at(-1)!);
    act(() =>
      fireEvent.change(screen.getByRole("textbox", { name: "Edit message" }), {
        target: { value: "Repeat that quantity" },
      }),
    );
    await user.click(screen.getByRole("button", { name: "Save & Send" }));
    await waitFor(() => expect(requests).toHaveLength(2));
    expect(requests[1]).not.toHaveProperty("previous_response_id");
    expect(requests[1].input).toEqual([
      imageMessage,
      ...toolPair,
      { type: "message", role: "user", content: "Repeat that quantity" },
    ]);
  });

  it("should replay history rather than reuse another model's response session after changing models", async () => {
    renderChat();
    await screen.findByRole("button", { name: "research-chat" });
    await send("Remember papaya");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "research-chat" }));
    await user.click(await screen.findByRole("button", { name: /^second-chat/ }));
    await send("What did I ask you to remember?");
    expect(requests[1]).not.toHaveProperty("previous_response_id");
    expect(requests[1]).toMatchObject({
      model: "second-chat",
      input: [
        { role: "user", content: "Remember papaya" },
        { role: "assistant", content: "Answer 1" },
        { role: "user", content: "What did I ask you to remember?" },
      ],
    });
  });
});
