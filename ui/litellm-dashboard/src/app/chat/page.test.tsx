import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ChatConversationPage from "./page";
import * as responsesApiModule from "@/components/llm_calls/responses_api";

const { mockUseChatShell } = vi.hoisted(() => ({
  mockUseChatShell: vi.fn(() => ({
    accessToken: "token-123",
    userId: "user-1",
    userEmail: "user@example.com",
    selectedMCPServers: [] as string[],
    setSelectedMCPServers: vi.fn(),
    activeConversationId: "conv-1",
    activeConversation: { id: "conv-1", messages: [] },
    storageUnavailable: false,
    staleId: null,
    createConversation: vi.fn(() => "conv-1"),
    appendMessage: vi.fn(),
    updateLastAssistantMessage: vi.fn(),
    truncateFromMessage: vi.fn(),
  })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));
vi.mock("@/contexts/ChatShellContext", () => ({ useChatShell: mockUseChatShell }));
vi.mock("@/components/chat/ChatShell", () => ({
  getChatRoutes: () => ({ chats: "/chat", integrations: "/chat/integrations" }),
}));
vi.mock("@/components/chat/ChatMessages", () => ({
  default: () => <div data-testid="chat-messages" />,
}));
vi.mock("@/components/chat/MCPConnectPicker", () => ({
  default: () => <div data-testid="mcp-connect-picker" />,
}));
vi.mock("@/components/molecules/message_manager", () => ({
  default: { error: vi.fn(), success: vi.fn(), info: vi.fn() },
}));
vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-5.2" }]),
}));
vi.mock("@/components/llm_calls/responses_api", () => ({
  makeOpenAIResponsesRequest: vi.fn(),
}));

describe("ChatConversationPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("aborts the in-flight streaming request when the component unmounts", async () => {
    vi.mocked(responsesApiModule.makeOpenAIResponsesRequest).mockImplementation(() => new Promise(() => {}));

    const { unmount } = render(<ChatConversationPage />);

    await waitFor(() => {
      expect(screen.getByText("gpt-5.2")).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText("How can I help you today?");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter" });

    await waitFor(() => {
      expect(responsesApiModule.makeOpenAIResponsesRequest).toHaveBeenCalled();
    });

    const signal = vi.mocked(responsesApiModule.makeOpenAIResponsesRequest).mock.calls[0][5];
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal!.aborted).toBe(false);

    unmount();

    expect(signal!.aborted).toBe(true);
  });
});
