import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatHistory } from "@/components/chat/useChatHistory";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import ChatConversationPage from "./page";
import { renderWithProviders } from "@/../tests/test-utils";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";

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

const ONE_TURN_ARG_COUNT = 26;
const ON_TIMING_DATA_INDEX = 7;
const ON_USAGE_DATA_INDEX = 8;
const ON_TOTAL_LATENCY_INDEX = 24;
const CUSTOM_HEADERS_INDEX = 25;
const ON_RESPONSE_ID_INDEX = 15;
const PREVIOUS_RESPONSE_ID_INDEX = 14;

async function sendOneMessage(onUrlUpdate?: OnUrlUpdateFunction): Promise<void> {
  renderWithProviders(<ChatConversationPage />, { onUrlUpdate });
  expect(await screen.findByRole("button", { name: /gpt-5\.4-mini/ })).toBeInTheDocument();
  fireEvent.change(screen.getByPlaceholderText("Describe a change, paste an error, or ask a question…"), {
    target: { value: "How much did this cost?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  await waitFor(() => expect(mockMakeOpenAIResponsesRequest).toHaveBeenCalledTimes(1));
}

describe("/ui/chat request metrics", () => {
  beforeEach(() => {
    localStorage.clear();
    mockMakeOpenAIResponsesRequest.mockReset();
    vi.mocked(fetchAvailableModels).mockResolvedValue([{ model_group: "gpt-5.4-mini" }]);
    shellState.storageUnavailable = false;
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

  it("should preserve metric callback positions and omit a team header for a direct model grant", async () => {
    mockMakeOpenAIResponsesRequest.mockResolvedValue(undefined);

    await sendOneMessage();

    const call = mockMakeOpenAIResponsesRequest.mock.calls[0];
    expect(call).toHaveLength(ONE_TURN_ARG_COUNT);
    expect(typeof call[ON_TIMING_DATA_INDEX]).toBe("function");
    expect(typeof call[ON_USAGE_DATA_INDEX]).toBe("function");
    expect(typeof call[ON_TOTAL_LATENCY_INDEX]).toBe("function");
    expect(call[CUSTOM_HEADERS_INDEX]).toBeUndefined();
  });

  it("should ignore a late old-team response ID after selecting a different billing team", async () => {
    const user = userEvent.setup();
    vi.mocked(fetchAvailableModels).mockResolvedValue([
      {
        model_group: "gpt-5.4-mini",
        available_teams: [
          { team_id: "research", team_alias: "Research" },
          { team_id: "engineering", team_alias: "Engineering" },
        ],
      },
    ]);
    mockMakeOpenAIResponsesRequest.mockImplementation(async (...args: unknown[]) => {
      const turn = mockMakeOpenAIResponsesRequest.mock.calls.length;
      const updateText = args[1] as (role: string, delta: string) => void;
      const onResponseId = args[ON_RESPONSE_ID_INDEX] as (id: string) => void;
      updateText("assistant", `Reply ${turn}`);
      onResponseId(`response-${turn}`);
    });
    renderWithProviders(<ChatConversationPage />);
    await screen.findByRole("button", { name: "gpt-5.4-mini" });
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Research" }));

    async function send(text: string) {
      act(() =>
        fireEvent.change(screen.getByRole("textbox", { name: "Message", exact: true }), {
          target: { value: text },
        }),
      );
      await user.click(screen.getByRole("button", { name: "Send", exact: true }));
    }

    await send("First question");
    await screen.findByText("Reply 1");
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Team paying for this request" })).toBeEnabled());
    const oldResponseId = mockMakeOpenAIResponsesRequest.mock.calls[0][ON_RESPONSE_ID_INDEX] as (id: string) => void;
    await user.click(screen.getByRole("combobox", { name: "Team paying for this request" }));
    await user.click(await screen.findByRole("option", { name: "Engineering" }));
    act(() => oldResponseId("late-research-response"));

    await send("Continue with Engineering");
    await screen.findByText("Reply 2");
    const engineeringRequest = mockMakeOpenAIResponsesRequest.mock.calls[1];
    expect(engineeringRequest[PREVIOUS_RESPONSE_ID_INDEX]).toBeNull();
    expect(engineeringRequest[CUSTOM_HEADERS_INDEX]).toEqual({ "x-litellm-team-id": "engineering" });
    await send("Continue again");
    await screen.findByText("Reply 3");
    expect(mockMakeOpenAIResponsesRequest.mock.calls[2][PREVIOUS_RESPONSE_ID_INDEX]).toBe("response-2");
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

    const banner = await screen.findByText("Some chat history or attachments could not be saved or restored.");
    const dismiss = within(banner.parentElement!).getByRole("button");
    expect(dismiss).toHaveClass("hover:text-warning/80");
    expect(dismiss).not.toHaveClass("hover:text-foreground");
  });
});
