import { cloneElement, type MouseEventHandler, type ReactElement, type ReactNode } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CreatedKeyDisplay, { buildAgentInstructions } from "./CreatedKeyDisplay";

import { toast } from "@/lib/toast";

const mockCopy = vi.hoisted(() => vi.fn<(text: string) => void>());

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://gw.test:4000/"),
}));

vi.mock("react-copy-to-clipboard", () => ({
  CopyToClipboard: ({
    text,
    onCopy,
    children,
  }: {
    text: string;
    onCopy?: (text: string, result: boolean) => void;
    children: ReactNode;
  }) => {
    const child = children as ReactElement<{ onClick?: MouseEventHandler<HTMLButtonElement> }>;
    return cloneElement(child, {
      onClick: (event) => {
        child.props.onClick?.(event);
        mockCopy(text);
        onCopy?.(text, true);
      },
    });
  },
}));

describe("CreatedKeyDisplay", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockCopy.mockReset();
    vi.mocked(toast.success).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should render", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByText("sk-test-123")).toBeInTheDocument();
  });

  it("should theme the key box with tokens instead of a hardcoded light background", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    const keyBox = screen.getByText("sk-test-123").parentElement as HTMLElement;

    expect(keyBox).not.toHaveAttribute("style");
    expect(keyBox).toHaveClass("bg-muted");
  });

  it("should display the security warning", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByText(/you will not be able to view it again/i)).toBeInTheDocument();
  });

  it("should show the copy button with initial label", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByRole("button", { name: /copy virtual key/i })).toBeInTheDocument();
  });

  it("should change button text to Copied after clicking copy", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));

    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();
  });

  it("should show a success message when the key is copied", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));

    expect(toast.success).toHaveBeenCalledWith("Key copied to clipboard");
  });

  it("should revert button text back after 2 seconds", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));
    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(screen.getByRole("button", { name: /copy virtual key/i })).toBeInTheDocument();
  });

  it("shows the gateway URL without its trailing slash and copies it", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    expect(screen.getByText("http://gw.test:4000")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy URL" }));

    expect(mockCopy).toHaveBeenCalledWith("http://gw.test:4000");
    expect(toast.success).toHaveBeenCalledWith("Gateway URL copied to clipboard");
  });

  it("starts with collapsed Claude Code instructions and expands them", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    expect(screen.getByRole("button", { name: "Show" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/POST http:\/\/gw\.test:4000\/v1\/responses/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show" }));

    expect(screen.getByRole("button", { name: "Hide" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/POST http:\/\/gw\.test:4000\/v1\/responses/)).toBeInTheDocument();
  });

  it("shows Codex instructions with the OpenAI base URL setting", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: "Codex" }));
    await user.click(screen.getByRole("button", { name: "Show" }));

    expect(screen.getByText("Instructions for Codex")).toBeInTheDocument();
    expect(screen.getByText(/OPENAI_BASE_URL=http:\/\/gw\.test:4000\/v1/)).toBeInTheDocument();
  });

  it("copies complete instructions while the instructions card is collapsed", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: "Copy instructions" }));

    const copiedInstructions = mockCopy.mock.calls.at(-1)?.[0] ?? "";
    expect(copiedInstructions).toContain("Gateway URL: http://gw.test:4000");
    expect(copiedInstructions).toContain("Virtual key: sk-test-123");
    expect(copiedInstructions).toContain("POST http://gw.test:4000/v1/chat/completions");
    expect(copiedInstructions).toContain("POST http://gw.test:4000/v1/responses");
    expect(copiedInstructions).toContain("POST http://gw.test:4000/v1/messages");
    expect(copiedInstructions).toContain("GET http://gw.test:4000/v1/models");
    expect(screen.getByRole("button", { name: "Copied!" })).toBeInTheDocument();
    expect(toast.success).toHaveBeenCalledWith("Instructions copied");
  });
});

describe("buildAgentInstructions", () => {
  it("builds Claude Code setup instructions with a normalized URL", () => {
    expect(buildAgentInstructions("claude-code", "http://gw.test:4000/", "sk-test-123")).toBe(
      [
        "Use this LiteLLM AI gateway for model inference.",
        "",
        "Gateway URL: http://gw.test:4000",
        "Virtual key: sk-test-123",
        "",
        "LiteLLM is a gateway that exposes OpenAI and Anthropic compatible APIs. Send the virtual key as a Bearer token (Authorization: Bearer sk-test-123) to:",
        "- POST http://gw.test:4000/v1/chat/completions (OpenAI Chat Completions)",
        "- POST http://gw.test:4000/v1/responses (OpenAI Responses)",
        "- POST http://gw.test:4000/v1/messages (Anthropic Messages)",
        "- GET http://gw.test:4000/v1/models (models this key can use)",
        "",
        "To run Claude Code through this gateway, set ANTHROPIC_BASE_URL=http://gw.test:4000 and ANTHROPIC_AUTH_TOKEN=sk-test-123.",
        "",
        "Docs for everything else LiteLLM supports: https://docs.litellm.ai/docs/",
      ].join("\n"),
    );
  });

  it("builds Codex setup instructions with a normalized URL", () => {
    const instructions = buildAgentInstructions("codex", "http://gw.test:4000/", "sk-test-123");

    expect(instructions).toContain(
      "To run Codex through this gateway, set OPENAI_BASE_URL=http://gw.test:4000/v1 and OPENAI_API_KEY=sk-test-123.",
    );
    expect(instructions).toContain("POST http://gw.test:4000/v1/responses (OpenAI Responses)");
  });
});
