import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import ResponseMetrics, { type TokenUsage } from "./ResponseMetrics";

const baseUsage: TokenUsage = { promptTokens: 5000, completionTokens: 12, totalTokens: 5012 };

describe("ResponseMetrics prompt cache chips", () => {
  it("renders both cache chips when the provider reports reads and writes", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 4695, cacheCreationTokens: 1234 }} />);

    expect(screen.getByText("Cache Read: 4695")).toBeInTheDocument();
    expect(screen.getByText("Cache Write: 1234")).toBeInTheDocument();
  });

  it("renders only the read chip when the provider reports reads alone", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 4695 }} />);

    expect(screen.getByText("Cache Read: 4695")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("renders no cache chips for a provider that reports no cache fields", () => {
    render(<ResponseMetrics usage={baseUsage} />);

    expect(screen.getByText("In: 5000")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("renders no cache chips when the provider reports zero cache tokens", () => {
    render(<ResponseMetrics usage={{ ...baseUsage, cacheReadTokens: 0, cacheCreationTokens: 0 }} />);

    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("shows the response cache indicator instead of the provider cache chips on a response-cache hit", () => {
    render(
      <ResponseMetrics
        usage={{ ...baseUsage, cacheReadTokens: 4695, cacheCreationTokens: 1234, servedFromResponseCache: true }}
      />,
    );

    expect(screen.getByText("Response Cache: Hit")).toBeInTheDocument();
    expect(screen.queryByText(/Cache Read/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Cache Write/)).not.toBeInTheDocument();
  });

  it("does not show the response cache indicator when the flag is absent", () => {
    render(<ResponseMetrics usage={baseUsage} />);

    expect(screen.queryByText(/Response Cache/)).not.toBeInTheDocument();
  });
});
