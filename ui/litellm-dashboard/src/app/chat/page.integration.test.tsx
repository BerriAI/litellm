import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatHistory } from "@/components/chat/useChatHistory";
import ChatConversationPage from "./page";

const { mockMakeOpenAIResponsesRequest } = vi.hoisted(() => ({
  mockMakeOpenAIResponsesRequest: vi.fn(),
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

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({ coy: {} }));

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
      storageUnavailable: false,
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

async function sendOneMessage(): Promise<void> {
  render(<ChatConversationPage />);
  await waitFor(() => expect(screen.getByRole("button", { name: /gpt-5\.4-mini/ })).toBeInTheDocument());
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

    await waitFor(() => expect(screen.getByLabelText("Total: 20")).toBeInTheDocument());
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

  it("shows no metrics bar for a turn the provider reported no usage for", async () => {
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      const updateTextUI = args[1] as (role: string, delta: string) => void;
      updateTextUI("assistant", "No usage here.");
    });

    await sendOneMessage();

    await waitFor(() => expect(screen.getByText("No usage here.")).toBeInTheDocument());
    expect(document.querySelector(".response-metrics")).toBeNull();
  });
});
