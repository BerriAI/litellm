import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
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
let webSearchSupported: boolean;
let status: "in_progress" | "completed" | "cancelled";
let retrievalStatus: number;
let cancellations: number;
let acceptance: Promise<void> | undefined;
const saved = (): Conversation[] => JSON.parse(localStorage.getItem(storageKey) ?? "[]");
const hasMCPTools = () => {
  const tools = requests.at(-1)?.tools;
  return Array.isArray(tools) && tools.some((tool) => tool.type === "mcp");
};

const result = () => ({
  id: "litellm_poll_research",
  status,
  usage: status === "completed" ? { input_tokens: 4, output_tokens: 9, total_tokens: 13, cost: "0.125" } : undefined,
  litellm_response_timing:
    status === "completed" ? { time_to_first_token_ms: 250, total_latency_ms: 12500 } : undefined,
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
          {
            type: "message",
            content: [
              {
                type: "output_text",
                text: "There are 12 widgets in stock.",
                annotations: [
                  {
                    type: "url_citation",
                    url: "https://example.com/inventory",
                    title: "Inventory report",
                    start_index: 0,
                    end_index: 6,
                  },
                ],
              },
            ],
          },
        ].filter((item) => item.type !== "mcp_call" || hasMCPTools())
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
  acceptance = undefined;
  supported = true;
  webSearchSupported = true;
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
          data: [
            {
              model_group: "research-chat",
              mode: "chat",
              supports_background_chat: supported,
              supports_web_search: webSearchSupported,
              supports_reasoning: true,
            },
          ],
        });
      if (path.endsWith("/v1/mcp/server")) return Response.json([{ server_id: "inventory", server_name: "inventory" }]);
      if (path.endsWith("/mcp-rest/tools/list")) return Response.json({ tools: [{ name: "lookup_inventory" }] });
      if (path.endsWith("/responses")) {
        const body = JSON.parse(await request.text());
        requests.push(body);
        if (body.background) {
          await acceptance;
          request.signal.throwIfAborted();
          return Response.json({ ...result(), status: "queued" });
        }
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
  expect(screen.getByLabelText(/^Cost:/)).toHaveTextContent("$0.125");
  expect(screen.getByLabelText(/^TTFT:/)).toHaveTextContent("0.25s");
  expect(screen.getByLabelText(/^Total Latency:/)).toHaveTextContent("12.50s");
  await waitFor(() =>
    expect(saved()[0]?.messages.at(-1)).toMatchObject({ timeToFirstToken: 250, totalLatency: 12500 }),
  );
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
  expect(requests[0]).not.toHaveProperty("instructions");
  const id = saved()[0].id;
  first.unmount();
  status = "completed";
  renderChat(id);
  expect(await screen.findByText("There are 12 widgets in stock.")).toBeVisible();
  expect(screen.getByLabelText(/^TTFT:/)).toHaveTextContent("0.25s");
  expect(screen.getByLabelText(/^Total Latency:/)).toHaveTextContent("12.50s");
  await waitFor(() =>
    expect(saved()[0]?.messages.at(-1)).toMatchObject({ timeToFirstToken: 250, totalLatency: 12500 }),
  );
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
  expect(requests[0]).not.toHaveProperty("instructions");
  expect(requests[0].tools).not.toContainEqual(expect.objectContaining({ type: "web_search" }));
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

it.each([false, true])(
  "should honor Stop before a background submission returns its ID (tools=%s)",
  async (withTools) => {
    let accept!: () => void;
    acceptance = new Promise<void>((resolve) => {
      accept = resolve;
    });
    renderChat();
    await startResearch(withTools);
    await waitFor(() => expect(requests).toHaveLength(1));
    fireEvent.click(screen.getByRole("button", { name: "Stop request" }));
    await act(async () => accept());
    expect(await screen.findByText("Cancellation requested")).toBeVisible();
    expect(cancellations).toBe(1);
    expect(requests).toHaveLength(1);
    await waitFor(() => expect(saved()[0]?.messages.at(-1)?.backgroundRun?.status).toBe("cancelled"));
  },
);

it("should restore a web research report and its source links after closing and reopening twice", async () => {
  const first = renderChat();
  await screen.findByRole("button", { name: "research-chat" });
  fireEvent.click(screen.getByRole("button", { name: "Connect tools" }));
  fireEvent.click(await screen.findByRole("button", { name: "Deep research", exact: true }));
  fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
    target: { value: "Research the inventory" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send", exact: true }));
  await waitFor(() => expect(saved()[0]?.messages.at(-1)?.backgroundRun?.id).toBe("litellm_poll_research"));
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({ background: true, stream: false, tools: [{ type: "web_search" }] });
  expect(requests[0].instructions).toEqual(expect.stringContaining("sources you actually consulted"));
  expect(saved()[0]?.messages.at(-1)?.research).toBe(true);
  const id = saved()[0].id;
  first.unmount();
  expect(cancellations).toBe(0);
  status = "completed";
  const second = renderChat(id);
  const sources = await screen.findByLabelText("Sources");
  expect(within(sources).getByRole("link", { name: "Inventory report" })).toHaveAttribute(
    "href",
    "https://example.com/inventory",
  );
  await waitFor(() =>
    expect(saved()[0]?.messages.at(-1)?.citations).toEqual([
      { url: "https://example.com/inventory", title: "Inventory report" },
    ]),
  );
  second.unmount();
  renderChat(id);
  expect(
    within(await screen.findByLabelText("Sources")).getByRole("link", { name: "Inventory report" }),
  ).toHaveAttribute("href", "https://example.com/inventory");
  expect(requests).toHaveLength(1);
});

it("should explain when the selected model cannot search instead of sending an unsupported research request", async () => {
  webSearchSupported = false;
  renderChat();
  await screen.findByRole("button", { name: "research-chat" });
  fireEvent.click(screen.getByRole("button", { name: "Connect tools" }));
  expect(await screen.findByRole("button", { name: "Deep research", exact: true })).toBeDisabled();
  expect(screen.getByText("Choose a reasoning model with web search.")).toBeVisible();
  expect(requests).toHaveLength(0);
});
