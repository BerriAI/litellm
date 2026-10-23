import { act, fireEvent, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import UsageAIChatPanel from "./UsageAIChatPanel";
import { usageAiChatStream } from "@/components/networking";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("@/components/networking", () => ({
  modelHubCall: vi.fn().mockResolvedValue({
    data: [{ model_group: "gpt-4" }, { model_group: "claude-3-opus" }],
  }),
  usageAiChatStream: vi.fn(),
}));

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  accessToken: "test-token",
};

describe("UsageAIChatPanel", () => {
  it("should render the panel when open", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByRole("complementary", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Aria Agent" })).toBeInTheDocument();
    expect(screen.getByText("Ask about your spend, models, keys, and trends")).toBeInTheDocument();
  });

  it("should render model selector", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    // One library paints the prompt as its own text node and the other leaves it on the input's
    // placeholder attribute, so either one means the user is being told what to pick.
    const prompt = "Select a model (optional, defaults to gpt-4o-mini)";
    expect(screen.queryAllByText(prompt).length + screen.queryAllByPlaceholderText(prompt).length).toBeGreaterThan(0);
  });

  it("should render empty state message when no conversation", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByRole("heading", { name: "Hi, I’m Aria." })).toBeInTheDocument();
  });

  it("should render the send button", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Send")).toBeInTheDocument();
  });

  it("should render input placeholder", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByRole("textbox", { name: "Message Aria Agent" })).toHaveAttribute(
      "placeholder",
      "Ask Aria about your usage…",
    );
  });

  it("should render clear chat button", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Clear chat")).toBeInTheDocument();
  });

  it("should have the panel element even when closed (just off-screen)", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} open={false} />);

    expect(screen.getByTestId("usage-ai-chat-panel")).toBeInTheDocument();
    expect(screen.getByTestId("usage-ai-chat-panel")).toHaveClass("translate-x-full");
    expect(screen.queryByRole("complementary", { name: "Aria Agent" })).not.toBeInTheDocument();
  });

  it("should not have translate-x-full class when open", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} open={true} />);

    expect(screen.getByTestId("usage-ai-chat-panel")).not.toHaveClass("translate-x-full");
    expect(screen.getByTestId("usage-ai-chat-panel")).toHaveClass("translate-x-0");
  });

  it("should let users send a suggested question through the usage endpoint", async () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Which model costs me the most?" }));
    });
    expect(screen.getByRole("textbox", { name: "Message Aria Agent" })).toHaveValue("Which model costs me the most?");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });
    expect(usageAiChatStream).toHaveBeenCalledWith(
      "test-token",
      [{ role: "user", content: "Which model costs me the most?" }],
      "",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });
});
