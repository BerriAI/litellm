import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders as render, testQueryClient } from "@/../tests/test-utils";
import { makeOpenAIChatCompletionRequest } from "@/components/llm_calls/chat_completion";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { PlaygroundChat } from "./PlaygroundChat";

vi.mock("@/app/(dashboard)/hooks/useCan", () => ({ default: () => true }));

vi.mock("@/components/llm_calls/fetch_models", () => ({ fetchAvailableModels: vi.fn() }));

vi.mock("@/components/llm_calls/chat_completion", () => ({ makeOpenAIChatCompletionRequest: vi.fn() }));

vi.mock("@/components/networking", () => ({
  tagListCall: vi.fn().mockResolvedValue({}),
  vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [{ guardrail_name: "pii-mask" }] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
  fetchMCPServers: vi.fn().mockResolvedValue([]),
  fetchMCPToolsets: vi.fn().mockResolvedValue([]),
  listMCPTools: vi.fn().mockResolvedValue({ tools: [] }),
  callMCPTool: vi.fn(),
  getGatewayBaseUrl: () => "http://gateway.test",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const chatRequest = vi.mocked(makeOpenAIChatCompletionRequest);

type ChatArgs = Parameters<typeof makeOpenAIChatCompletionRequest>;

const sentMessages = (call: number) => (chatRequest.mock.calls[call] as ChatArgs)[0];

const replyWith =
  (text: string) =>
  async (...args: ChatArgs) => {
    args[1](text, args[2]);
  };

const renderChat = () =>
  render(<PlaygroundChat accessToken="session-token" disabledPersonalKeyCreation={false} proxySettings={undefined} />);

const send = async (user: ReturnType<typeof userEvent.setup>, text: string) => {
  await user.type(screen.getByRole("textbox", { name: "Message" }), text);
  await user.click(screen.getByRole("button", { name: "Send message" }));
};

describe("PlaygroundChat", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    testQueryClient.clear();
    Element.prototype.scrollIntoView = vi.fn();
    vi.mocked(fetchAvailableModels).mockResolvedValue([
      { model_group: "gpt-fast", mode: "chat", providers: ["openai"] },
    ]);
  });

  it("sends the chosen model and guardrails with the session key", async () => {
    const user = userEvent.setup();
    chatRequest.mockImplementation(replyWith("Hi there"));
    renderChat();

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.click(await screen.findByRole("option", { name: "gpt-fast" }));
    await user.click(screen.getByRole("combobox", { name: "Guardrails" }));
    await user.click(await screen.findByRole("option", { name: "pii-mask" }));
    await user.keyboard("{Escape}");
    await send(user, "hello");

    expect(await screen.findByText("Hi there")).toBeInTheDocument();
    const [messages, , model, apiKey, , , , , , , , guardrails] = chatRequest.mock.calls[0] as ChatArgs;
    expect(messages).toEqual([{ role: "user", content: "hello" }]);
    expect(model).toBe("gpt-fast");
    expect(apiKey).toBe("session-token");
    expect(guardrails).toEqual(["pii-mask"]);
  });

  it("keeps answered turns in the history but never resends a failed one", async () => {
    const user = userEvent.setup();
    chatRequest
      .mockImplementationOnce(replyWith("Hi there"))
      .mockRejectedValueOnce(new Error("rate limited"))
      .mockImplementationOnce(replyWith("Back again"));
    renderChat();

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.click(await screen.findByRole("option", { name: "gpt-fast" }));
    await send(user, "hello");
    expect(await screen.findByText("Hi there")).toBeInTheDocument();
    await send(user, "this one fails");
    expect(await screen.findByText("The request failed")).toBeInTheDocument();
    await send(user, "try again");

    await waitFor(() => expect(chatRequest).toHaveBeenCalledTimes(3));
    expect(sentMessages(2)).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "Hi there" },
      { role: "user", content: "try again" },
    ]);
  });
});
