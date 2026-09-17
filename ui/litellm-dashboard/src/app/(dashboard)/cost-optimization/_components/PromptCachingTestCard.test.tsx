import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => "http://proxy:4000"),
}));

vi.mock("@/components/common_components/ModelSelector", () => ({
  __esModule: true,
  default: ({ onChange }: { onChange?: (value: string | null) => void }) => (
    <button data-testid="model-selector" onClick={() => onChange?.("claude-haiku-4-5")} />
  ),
}));

import PromptCachingTestCard from "./PromptCachingTestCard";

const successOutcome = {
  first: {
    cacheCreationTokens: 5000,
    cacheReadTokens: 0,
    promptTokens: 5010,
    responseCost: 0.004,
    model: "claude-haiku-4-5",
    durationMs: 120,
  },
  second: {
    cacheCreationTokens: 0,
    cacheReadTokens: 5000,
    promptTokens: 5010,
    responseCost: 0.001,
    model: "claude-haiku-4-5",
    durationMs: 80,
  },
  verdict: "injected" as const,
};

describe("PromptCachingTestCard", () => {
  it("keeps the run button disabled until a model is picked", () => {
    render(<PromptCachingTestCard accessToken="token" enabled={true} runTest={vi.fn()} />);

    expect(screen.getByRole("button", { name: /run test/i })).toBeDisabled();
  });

  it("shows the injected verdict after a successful run", async () => {
    const runTest = vi.fn(async () => successOutcome);
    render(<PromptCachingTestCard accessToken="token" enabled={true} runTest={runTest} />);

    fireEvent.click(screen.getByTestId("model-selector"));
    fireEvent.click(screen.getByRole("button", { name: /run test/i }));

    expect(await screen.findByText(/LiteLLM injected cache_control: call 1 wrote 5000 tokens/)).toBeInTheDocument();
    expect(runTest).toHaveBeenCalledWith(
      expect.objectContaining({ accessToken: "token", model: "claude-haiku-4-5", baseUrl: "http://proxy:4000" }),
    );
  });

  it("shows an error alert when the run throws", async () => {
    const runTest = vi.fn(async () => {
      throw new Error("Request failed with status 400: boom");
    });
    render(<PromptCachingTestCard accessToken="token" enabled={true} runTest={runTest} />);

    fireEvent.click(screen.getByTestId("model-selector"));
    fireEvent.click(screen.getByRole("button", { name: /run test/i }));

    expect(await screen.findByText(/status 400: boom/)).toBeInTheDocument();
  });
});
