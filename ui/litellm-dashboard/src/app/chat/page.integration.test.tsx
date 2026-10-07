import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatHistory } from "@/components/chat/useChatHistory";
import ChatConversationPage from "./page";
import { renderWithProviders } from "@/../tests/test-utils";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";

const { mockMakeOpenAIResponsesRequest, shellState } = vi.hoisted(() => ({
  mockMakeOpenAIResponsesRequest: vi.fn(),
  shellState: { storageUnavailable: false },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(async () => [{ model_group: "gpt-5.4-mini" }]),
}));

vi.mock("@/components/llm_calls/responses_api", () => ({
  makeOpenAIResponsesRequest: mockMakeOpenAIResponsesRequest,
}));

vi.mock("@/components/chat/MCPConnectPicker", () => ({
  default: () => <div data-testid="mcp-connect-picker" />,
}));

vi.mock("react-markdown", () => ({
  default: ({ children }: { children: string }) => <div>{children}</div>,
}));

vi.mock("remark-gfm", () => ({ default: () => undefined }));

vi.mock("react-syntax-highlighter", () => ({
  Prism: ({ children }: { children: string }) => <pre>{children}</pre>,
}));

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({ coy: {}, oneDark: {}, oneLight: {}, prism: {} }));

vi.mock("@/contexts/ChatShellContext", () => ({
  useChatShell: () => {
    const history = useChatHistory(null, "metrics-test-user");
    return {
      accessToken: "sk-test",
      userId: "metrics-test-user",
      userEmail: "tester@example.com",
      userRole: "Admin",
      premiumUser: false,
      selectedMCPServers: [],
      setSelectedMCPServers: vi.fn(),
      conversations: history.conversations,
      activeConversation: history.activeConversation,
      activeConversationId: history.currentActiveId,
      storageUnavailable: shellState.storageUnavailable,
      staleId: false,
      createConversation: history.createConversation,
      appendMessage: history.appendMessage,
      updateLastAssistantMessage: history.updateLastAssistantMessage,
      truncateFromMessage: history.truncateFromMessage,
      deleteConversation: vi.fn(),
      renameConversation: vi.fn(),
    };
  },
}));

const ONE_TURN_ARG_COUNT = 25;
const ON_TIMING_DATA_INDEX = 7;
const ON_USAGE_DATA_INDEX = 8;
const ON_TOTAL_LATENCY_INDEX = 24;

async function sendOneMessage(onUrlUpdate?: OnUrlUpdateFunction): Promise<void> {
  renderWithProviders(<ChatConversationPage />, { onUrlUpdate });
  expect(await screen.findByRole("button", { name: /gpt-5\.4-mini/ })).toBeInTheDocument();
  fireEvent.change(screen.getByPlaceholderText("How can I help you today?"), {
    target: { value: "How much did this cost?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  await waitFor(() => expect(mockMakeOpenAIResponsesRequest).toHaveBeenCalledTimes(1));
}

describe("/ui/chat request metrics", () => {
  beforeEach(() => {
    localStorage.clear();
    mockMakeOpenAIResponsesRequest.mockReset();
    shellState.storageUnavailable = false;
    vi.mocked(fetchAvailableModels).mockResolvedValue([{ model_group: "gpt-5.4-mini" }]);
  });

  it.each([
    { providers: ["openrouter"], stateless: true },
    { providers: ["openai", "openrouter"], stateless: true },
    { providers: ["openai"], stateless: false },
    { providers: undefined, stateless: false },
  ])("continues a conversation correctly with providers=$providers", async ({ providers, stateless }) => {
    vi.mocked(fetchAvailableModels).mockResolvedValue([{ model_group: "gpt-5.4-mini", providers }]);
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      (args[1] as (role: string, delta: string) => void)("assistant", "First answer");
      (args[15] as (id: string) => void)("response-first");
    });

    await sendOneMessage();
    expect(await screen.findByText("First answer")).toBeVisible();
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Follow up" },
    });
    await waitFor(() => expect(screen.getByRole("button", { name: "Send" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(mockMakeOpenAIResponsesRequest).toHaveBeenCalledTimes(2));

    const [first, second] = mockMakeOpenAIResponsesRequest.mock.calls;
    expect(first[14]).toBeNull();
    expect(second[14]).toBe(stateless ? null : "response-first");
    expect(second[0]).toEqual(
      stateless
        ? [
            { role: "user", content: "How much did this cost?" },
            { role: "assistant", content: "First answer" },
            { role: "user", content: "Follow up" },
          ]
        : [{ role: "user", content: "Follow up" }],
    );
  });

  it("resends history instead of reusing another model's response ID after changing models", async () => {
    vi.mocked(fetchAvailableModels).mockResolvedValue([
      { model_group: "gpt-5.4-mini", providers: ["openai"] },
      { model_group: "second-model", providers: ["openai"] },
    ]);
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      (args[1] as (role: string, delta: string) => void)("assistant", "First answer");
      (args[15] as (id: string) => void)("response-first");
    });

    await sendOneMessage();
    expect(await screen.findByText("First answer")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: /gpt-5\.4-mini/ }));
    fireEvent.click(await screen.findByRole("button", { name: "second-model" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Follow up" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Send" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(mockMakeOpenAIResponsesRequest).toHaveBeenCalledTimes(2));

    const second = mockMakeOpenAIResponsesRequest.mock.calls[1];
    expect(second[2]).toBe("second-model");
    expect(second[14]).toBeNull();
    expect(second[0]).toEqual([
      { role: "user", content: "How much did this cost?" },
      { role: "assistant", content: "First answer" },
      { role: "user", content: "Follow up" },
    ]);
  });

  it("renders latency, TTFT, token counts and cost reported for the assistant turn", async () => {
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      const updateTextUI = args[1] as (role: string, delta: string) => void;
      const onTimingData = args[ON_TIMING_DATA_INDEX] as ((ttft: number) => void) | undefined;
      const onUsageData = args[ON_USAGE_DATA_INDEX] as ((usage: Record<string, number>) => void) | undefined;
      const onTotalLatency = args[ON_TOTAL_LATENCY_INDEX] as ((latency: number) => void) | undefined;

      updateTextUI("assistant", "Sixty three microdollars.");
      onTimingData?.(250);
      onUsageData?.({ promptTokens: 12, completionTokens: 8, totalTokens: 20, cost: 0.000063 });
      onTotalLatency?.(1200);
    });

    await sendOneMessage();

    expect(await screen.findByLabelText("Total: 20")).toBeInTheDocument();
    expect(screen.getByLabelText("TTFT: 0.25s")).toBeInTheDocument();
    expect(screen.getByLabelText("Total Latency: 1.20s")).toBeInTheDocument();
    expect(screen.getByLabelText("In: 12")).toBeInTheDocument();
    expect(screen.getByLabelText("Out: 8")).toBeInTheDocument();
    expect(screen.getByLabelText("Cost: $0.000063")).toBeInTheDocument();
  });

  it("supplies the timing, usage and latency callbacks at the positional slots the Responses helper reads", async () => {
    mockMakeOpenAIResponsesRequest.mockResolvedValue(undefined);

    await sendOneMessage();

    const call = mockMakeOpenAIResponsesRequest.mock.calls[0];
    expect(call).toHaveLength(ONE_TURN_ARG_COUNT);
    expect(typeof call[ON_TIMING_DATA_INDEX]).toBe("function");
    expect(typeof call[ON_USAGE_DATA_INDEX]).toBe("function");
    expect(typeof call[ON_TOTAL_LATENCY_INDEX]).toBe("function");
  });

  it("puts the new conversation ID in the URL after the first message is sent", async () => {
    mockMakeOpenAIResponsesRequest.mockResolvedValue(undefined);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();

    await sendOneMessage(onUrlUpdate);

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const newConversationId = onUrlUpdate.mock.lastCall?.[0].searchParams.get("id");
    expect(newConversationId).toBeTruthy();
    expect(onUrlUpdate.mock.lastCall?.[0].options.history).toBe("push");
    expect(localStorage.getItem("litellm_chat_history_v1:metrics-test-user")).toContain(newConversationId);
  });

  it("shows no metrics bar for a turn the provider reported no usage for", async () => {
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      const updateTextUI = args[1] as (role: string, delta: string) => void;
      updateTextUI("assistant", "No usage here.");
    });

    await sendOneMessage();

    expect(await screen.findByText("No usage here.")).toBeInTheDocument();
    expect(document.querySelector(".response-metrics")).toBeNull();
  });
});

describe("/ui/chat storage banner", () => {
  beforeEach(() => {
    localStorage.clear();
    mockMakeOpenAIResponsesRequest.mockReset();
    shellState.storageUnavailable = true;
  });

  it("keeps the dismiss control amber on hover instead of the ghost variant's foreground", async () => {
    renderWithProviders(<ChatConversationPage />);

    const banner = await screen.findByText("Chat history won't be saved in this browser session");
    const dismiss = within(banner.parentElement!).getByRole("button");
    expect(dismiss).toHaveClass("hover:text-warning/80");
    expect(dismiss).not.toHaveClass("hover:text-foreground");
  });
});
