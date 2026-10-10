import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NuqsAdapter } from "nuqs/adapters/react";
import { ChatShellProvider } from "@/contexts/ChatShellContext";
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

vi.mock("react-syntax-highlighter/dist/esm/styles/prism", () => ({ coy: {}, oneDark: {}, oneLight: {}, prism: {} }));

describe("chat page URL state with ChatShellProvider", () => {
  beforeEach(() => {
    localStorage.clear();
    window.history.replaceState(null, "", "/chat");
    mockMakeOpenAIResponsesRequest.mockReset();
  });

  it("keeps the active conversation in sync with the URL across the first send and back navigation", async () => {
    mockMakeOpenAIResponsesRequest.mockResolvedValue(undefined);

    render(
      <NuqsAdapter>
        <ChatShellProvider
          accessToken="sk-test"
          userId="url-test-user"
          userEmail="t@example.com"
          userRole="Admin"
          premiumUser={false}
        >
          <ChatConversationPage />
        </ChatShellProvider>
      </NuqsAdapter>,
    );

    expect(await screen.findByRole("button", { name: /gpt-5\.4-mini/ })).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText("How can I help you today?"), {
      target: { value: "How much did this cost?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(window.location.search).toMatch(/^\?id=/));
    const conversationId = new URLSearchParams(window.location.search).get("id");
    expect(conversationId).toBeTruthy();
    expect(await screen.findByText("How much did this cost?")).toBeInTheDocument();
    await waitFor(() =>
      expect(localStorage.getItem("litellm_chat_history_v1:url-test-user")).toContain(conversationId),
    );

    act(() => window.history.back());
    await waitFor(() => expect(window.location.search).toBe(""));

    expect(await screen.findByPlaceholderText("How can I help you today?")).toBeInTheDocument();
    expect(screen.queryByText("How much did this cost?")).not.toBeInTheDocument();
  });
});
