import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatMessageContent } from "./ChatMessages";
import type { ChatMessage } from "./types";

vi.mock("@/components/networking", () => ({ getGatewayBaseUrl: () => "https://gateway.example/v1" }));
vi.mock("@/components/chat_ui/ReasoningContent", () => ({ default: () => null }));
vi.mock("@/components/chat_ui/MCPEventsDisplay", () => ({ default: () => null }));
vi.mock("@/components/chat_ui/ResponseMetrics", () => ({ default: () => null }));
vi.mock("@/hooks/useSyntaxTheme", () => ({ useSyntaxTheme: (theme: unknown) => theme }));

const image = { id: "image-1", name: "design.png", url: "data:image/png;base64,aW1hZ2U=" };
const message: ChatMessage = {
  id: "message-1",
  role: "user",
  content: "Describe this design",
  timestamp: 1,
  attachments: [image],
};

describe("ChatMessageContent images", () => {
  it("should render a user message and its image attachment", () => {
    render(<ChatMessageContent message={message} />);
    expect(screen.getByText("Describe this design")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "design.png" })).toHaveAttribute("src", image.url);
    expect(screen.getByRole("link", { name: "Open image" })).toHaveAttribute("href", image.url);
  });

  it("should display generated images even when assistant text is empty", () => {
    render(
      <ChatMessageContent
        message={{ ...message, role: "assistant", content: "", attachments: undefined, generatedImages: [image] }}
        isStreaming
      />,
    );
    expect(screen.getByRole("img", { name: "design.png" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy message" })).not.toBeInTheDocument();
  });

  it("should show image generation progress", () => {
    render(
      <ChatMessageContent
        message={{
          ...message,
          role: "assistant",
          content: "",
          imageGeneration: { prompt: "A design", model: "image-model" },
        }}
        isStreaming
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Creating your image");
    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
  });

  it("should respect image suppression for uploaded and generated images", () => {
    render(
      <>
        <ChatMessageContent message={message} allowImages={false} />
        <ChatMessageContent
          message={{ ...message, id: "assistant-1", role: "assistant", content: "", generatedImages: [image] }}
          allowImages={false}
        />
      </>,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open image" })).not.toBeInTheDocument();
    expect(screen.getAllByText("design.png")).toHaveLength(2);
  });

  it("should retain image previews while editing the message text", () => {
    const onEditMessage = vi.fn();
    render(<ChatMessageContent message={message} onEditMessage={onEditMessage} />);
    act(() => {
      fireEvent.mouseEnter(screen.getByText("Describe this design"));
    });
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Edit message" }));
    });
    expect(screen.getByRole("img", { name: "design.png" })).toBeInTheDocument();
    act(() => {
      fireEvent.change(screen.getByRole("textbox", { name: "Edit message" }), { target: { value: "" } });
    });
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Save & Send" }));
    });
    expect(onEditMessage).toHaveBeenCalledWith(message.id, "");
  });

  it("should show a helpful message when saved image bytes are missing", () => {
    render(<ChatMessageContent message={{ ...message, attachments: [{ ...image, url: "" }] }} />);
    expect(screen.getByText(/design.png — image unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("should show a helpful message when a provider image URL expires", () => {
    render(<ChatMessageContent message={message} />);
    act(() => {
      fireEvent.error(screen.getByRole("img", { name: "design.png" }));
    });
    expect(screen.getByText(/image unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
