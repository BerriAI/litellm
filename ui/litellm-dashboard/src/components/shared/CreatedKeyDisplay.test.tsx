import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CreatedKeyDisplay from "./CreatedKeyDisplay";

vi.mock("@/components/molecules/message_manager", () => ({
  default: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn(), loading: vi.fn(), destroy: vi.fn() },
}));

vi.mock("@/components/networking", () => ({
  proxyBaseUrl: "https://litellm.example.com",
}));

import MessageManager from "@/components/molecules/message_manager";

describe("CreatedKeyDisplay", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.mocked(MessageManager.success).mockClear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the API key value", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByTestId("created-key-value")).toHaveTextContent("sk-test-123");
  });

  it("displays the security warning and modal heading", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    expect(screen.getByRole("heading", { name: /api key created/i })).toBeInTheDocument();
    expect(screen.getByText(/make sure to copy your api key now/i)).toBeInTheDocument();
    expect(screen.getByText(/you won't be able to see it again/i)).toBeInTheDocument();
  });

  it("builds a coding agent prompt that advertises the gateway endpoints and docs without leaking the key", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    const prompt = screen.getByTestId("coding-agent-prompt").textContent ?? "";

    expect(prompt).toMatch(/Base URL: https:\/\/litellm\.example\.com/);
    expect(prompt).not.toContain("sk-test-123");
    expect(prompt).toContain("$LITELLM_API_KEY");
    expect(prompt).toContain("Authorization: Bearer <key>");

    expect(prompt).toContain("/chat/completions");
    expect(prompt).toContain("/v1/messages");
    expect(prompt).toContain("/v1/responses");
    expect(prompt).toContain("/mcp");
    expect(prompt).toContain("/v1/models");

    expect(prompt).toContain("https://docs.litellm.ai/docs/learn/gateway_quickstart");
    expect(prompt).toContain("https://docs.litellm.ai/docs/tutorials/claude_responses_api");
    expect(prompt).toContain("https://docs.litellm.ai/docs/anthropic_unified");
    expect(prompt).toContain("https://docs.litellm.ai/docs/mcp");
    expect(prompt).toContain("https://docs.litellm.ai/llms.txt");
  });

  it("does not produce a trailing slash in the base URL when proxyBaseUrl is provided", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    const prompt = screen.getByTestId("coding-agent-prompt").textContent ?? "";
    expect(prompt).not.toMatch(/Base URL: https:\/\/litellm\.example\.com\//);
  });

  it("copies the API key to clipboard when the primary button is clicked", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));

    expect(MessageManager.success).toHaveBeenCalledWith("Key copied to clipboard");
    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();
  });

  it("renders coding agent logos next to the prompt heading", () => {
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);
    const logos = screen.getByTestId("coding-agent-logos");
    const images = logos.querySelectorAll("img");
    const altText = Array.from(images).map((img) => img.getAttribute("alt"));

    expect(altText).toEqual(["Cursor", "Claude Code", "OpenAI Codex", "GitHub Copilot"]);
  });

  it("copies the coding agent prompt to clipboard when its copy button is clicked", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy prompt for coding agents/i }));

    expect(MessageManager.success).toHaveBeenCalledWith("Prompt copied to clipboard");
  });

  it("reverts the primary button label after 2 seconds", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<CreatedKeyDisplay apiKey="sk-test-123" />);

    await user.click(screen.getByRole("button", { name: /copy virtual key/i }));
    expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(screen.getByRole("button", { name: /copy virtual key/i })).toBeInTheDocument();
  });
});
