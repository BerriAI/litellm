import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { modelHubCall, usageAiChatStream } from "../../networking";
import UsageAIChatPanel from "./UsageAIChatPanel";

beforeAll(() => {
  if (typeof window !== "undefined" && !window.ResizeObserver) {
    window.ResizeObserver = class ResizeObserver {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any;
  }
});

vi.mock("../../networking", () => ({
  modelHubCall: vi.fn().mockResolvedValue({
    data: [
      { model_group: "gpt-4" },
      { model_group: "claude-3-opus" },
    ],
  }),
  usageAiChatStream: vi.fn().mockResolvedValue(undefined),
}));

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  accessToken: "test-token",
};

describe("UsageAIChatPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the panel when open", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Ask AI")).toBeInTheDocument();
    expect(
      screen.getByText("Ask about your spend, models, keys, and trends")
    ).toBeInTheDocument();
  });

  it("should render model selector", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(
      screen.getByText("Select a model (optional, defaults to gpt-4o-mini)")
    ).toBeInTheDocument();
  });

  it("should render empty state message when no conversation", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Ask a question about your usage")).toBeInTheDocument();
  });

  it("should render the send button", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Send")).toBeInTheDocument();
  });

  it("should render input placeholder", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByPlaceholderText("Ask about your usage...")).toBeInTheDocument();
  });

  it("should render clear chat button", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    expect(screen.getByText("Clear chat")).toBeInTheDocument();
  });

  it("should have the panel element even when closed (just off-screen)", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} open={false} />);

    expect(screen.getByTestId("usage-ai-chat-panel")).toBeInTheDocument();
    expect(screen.getByTestId("usage-ai-chat-panel")).toHaveClass("translate-x-full");
  });

  it("should not have translate-x-full class when open", () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} open={true} />);

    expect(screen.getByTestId("usage-ai-chat-panel")).not.toHaveClass("translate-x-full");
    expect(screen.getByTestId("usage-ai-chat-panel")).toHaveClass("translate-x-0");
  });

  it("should submit the selected model value unchanged", async () => {
    renderWithProviders(<UsageAIChatPanel {...defaultProps} />);

    await waitFor(() => {
      expect(modelHubCall).toHaveBeenCalledWith("test-token");
    });

    const modelSelect = screen.getByRole("combobox");
    await act(async () => {
      fireEvent.mouseDown(modelSelect);
    });

    await waitFor(() => {
      expect(
        screen.getByRole("option", { name: "claude-3-opus" })
      ).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByTitle("claude-3-opus"));
    });

    await act(async () => {
      fireEvent.change(screen.getByPlaceholderText("Ask about your usage..."), {
        target: { value: "hello" },
      });
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    await waitFor(() => {
      expect(usageAiChatStream).toHaveBeenCalled();
    });

    expect(usageAiChatStream).toHaveBeenCalledWith(
      "test-token",
      [{ role: "user", content: "hello" }],
      "claude-3-opus",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(AbortSignal),
    );
  });
});
