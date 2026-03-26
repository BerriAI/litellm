import React from "react";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { InputCard } from "./InputCard";
import { ParsedMessage } from "./prettyMessagesTypes";

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd");
  return {
    ...actual,
    message: {
      success: vi.fn(),
    },
  };
});

describe("InputCard", () => {
  const mockWriteText = vi.fn().mockResolvedValue(undefined);
  const mockMessages: ParsedMessage[] = [
    {
      role: "user",
      content: "Hello, how are you?",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: mockWriteText,
      },
      writable: true,
      configurable: true,
    });
  });

  it("should render the InputCard component", () => {
    render(<InputCard messages={mockMessages} />);
    expect(screen.getByText("Input")).toBeInTheDocument();
  });

  it("should return null when messages array is empty", () => {
    const { container } = render(<InputCard messages={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("should display system message when present", () => {
    const messagesWithSystem: ParsedMessage[] = [
      {
        role: "system",
        content: "You are a helpful assistant",
      },
      {
        role: "user",
        content: "Hello",
      },
    ];
    render(<InputCard messages={messagesWithSystem} />);
    expect(screen.getByText("SYSTEM")).toBeInTheDocument();
    expect(screen.getByText("You are a helpful assistant")).toBeInTheDocument();
  });

  it("should display history messages when present", () => {
    const messagesWithHistory: ParsedMessage[] = [
      {
        role: "user",
        content: "First message",
      },
      {
        role: "assistant",
        content: "Response",
      },
      {
        role: "user",
        content: "Last message",
      },
    ];
    render(<InputCard messages={messagesWithHistory} />);
    expect(screen.getByText("Last message")).toBeInTheDocument();
  });

  it("should display last message content", () => {
    const messages: ParsedMessage[] = [
      {
        role: "user",
        content: "What is the weather?",
      },
    ];
    render(<InputCard messages={messages} />);
    expect(screen.getByText("What is the weather?")).toBeInTheDocument();
  });

  it("should display token count when provided", () => {
    render(<InputCard messages={mockMessages} promptTokens={150} />);
    expect(screen.getByText(/Tokens: 150/)).toBeInTheDocument();
  });

  it("should display cost when provided", () => {
    render(<InputCard messages={mockMessages} inputCost={0.0015} />);
    expect(screen.getByText(/Cost: \$0\.001500/)).toBeInTheDocument();
  });

  it("should copy last message content when copy button is clicked", async () => {
    const messages: ParsedMessage[] = [
      {
        role: "user",
        content: "Copy this text",
      },
    ];

    render(<InputCard messages={messages} />);

    const copyButtons = screen.getAllByRole("button");
    const copyButton = copyButtons.find((button) => {
      const icon = button.querySelector('[aria-label="copy"]');
      return icon !== null;
    });

    expect(copyButton).toBeInTheDocument();
    
    await act(async () => {
      fireEvent.click(copyButton!);
    });

    await waitFor(() => {
      expect(mockWriteText).toHaveBeenCalledWith("Copy this text");
    });
  });

  it("should toggle collapse state when header is clicked", async () => {
    const user = userEvent.setup();
    render(<InputCard messages={mockMessages} />);

    const header = screen.getByText("Input").closest("div");
    expect(header).toBeInTheDocument();

    const content = screen.getByText("Hello, how are you?");
    expect(content).toBeInTheDocument();
    expect(content).toBeVisible();

    if (header) {
      await user.click(header);
      await waitFor(() => {
        expect(content).not.toBeVisible();
      });
    }
  });

  it("should handle messages without system message", () => {
    const messages: ParsedMessage[] = [
      {
        role: "user",
        content: "User message only",
      },
    ];
    render(<InputCard messages={messages} />);
    expect(screen.queryByText("SYSTEM")).not.toBeInTheDocument();
    expect(screen.getByText("User message only")).toBeInTheDocument();
  });

  it("should handle messages with only system message", () => {
    const messages: ParsedMessage[] = [
      {
        role: "system",
        content: "System only",
      },
    ];
    render(<InputCard messages={messages} />);
    expect(screen.getByText("SYSTEM")).toBeInTheDocument();
    expect(screen.getByText("System only")).toBeInTheDocument();
  });

  it("should display last message role label correctly", () => {
    const messages: ParsedMessage[] = [
      {
        role: "assistant",
        content: "Assistant response",
      },
    ];
    render(<InputCard messages={messages} />);
    expect(screen.getByText("ASSISTANT")).toBeInTheDocument();
    expect(screen.getByText("Assistant response")).toBeInTheDocument();
  });

  it("should handle tool calls in last message", () => {
    const messages: ParsedMessage[] = [
      {
        role: "user",
        content: "Call a function",
        toolCalls: [
          {
            id: "call-1",
            name: "get_weather",
            arguments: { location: "NYC" },
          },
        ],
      },
    ];
    render(<InputCard messages={messages} />);
    expect(screen.getByText("Call a function")).toBeInTheDocument();
  });

  it("should handle empty content in last message", () => {
    const messages: ParsedMessage[] = [
      {
        role: "user",
        content: "",
      },
    ];
    render(<InputCard messages={messages} />);
    const copyButtons = screen.getAllByRole("button");
    const copyButton = copyButtons.find((button) => {
      const icon = button.querySelector('[aria-label="copy"]');
      return icon !== null;
    });
    expect(copyButton).toBeInTheDocument();
  });
});
