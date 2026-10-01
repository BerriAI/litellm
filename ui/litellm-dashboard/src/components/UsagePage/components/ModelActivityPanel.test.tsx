import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ModelActivityData } from "../types";
import ModelActivityPanel from "./ModelActivityPanel";

vi.mock("@/components/activity_metrics", () => ({
  ActivityMetrics: ({ modelMetrics }: { modelMetrics: Record<string, ModelActivityData> }) => (
    <ul data-testid="rendered-models">
      {Object.keys(modelMetrics).map((model) => (
        <li key={model}>{model}</li>
      ))}
    </ul>
  ),
}));

function activity(label: string): ModelActivityData {
  return {
    label,
    total_requests: 1,
    total_successful_requests: 1,
    total_failed_requests: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    total_spend: 0.01,
    top_api_keys: [],
    top_models: [],
    daily_data: [],
  };
}

const modelMetrics: Record<string, ModelActivityData> = {
  "gpt-5.5": activity("GPT-5.5"),
  "claude-sonnet-4": activity("Claude Sonnet"),
};

describe("ModelActivityPanel", () => {
  it("renders every model and the full count before searching", () => {
    render(<ModelActivityPanel modelMetrics={modelMetrics} />);
    expect(screen.getByTestId("rendered-models")).toHaveTextContent("gpt-5.5claude-sonnet-4");
    expect(screen.getByText("Showing 2 of 2 models")).toBeInTheDocument();
  });

  it("narrows the rendered models to those matching the query", () => {
    render(<ModelActivityPanel modelMetrics={modelMetrics} />);
    fireEvent.change(screen.getByLabelText("Search models"), { target: { value: "sonnet" } });
    expect(screen.getByTestId("rendered-models")).toHaveTextContent("claude-sonnet-4");
    expect(screen.getByTestId("rendered-models")).not.toHaveTextContent("gpt-5.5");
    expect(screen.getByText("Showing 1 of 2 models")).toBeInTheDocument();
  });

  it("shows an empty state instead of zeroed metrics when nothing matches", () => {
    render(<ModelActivityPanel modelMetrics={modelMetrics} />);
    fireEvent.change(screen.getByLabelText("Search models"), { target: { value: "llama" } });
    expect(screen.queryByTestId("rendered-models")).not.toBeInTheDocument();
    expect(screen.getByText('No models match "llama" in this date range')).toBeInTheDocument();
  });

  it("clears the search and restores every model", () => {
    render(<ModelActivityPanel modelMetrics={modelMetrics} />);
    fireEvent.change(screen.getByLabelText("Search models"), { target: { value: "gpt" } });
    expect(screen.getByTestId("rendered-models")).toHaveTextContent("gpt-5.5");
    fireEvent.click(screen.getByLabelText("Clear model search"));
    expect(screen.getByLabelText("Search models")).toHaveValue("");
    expect(screen.getByTestId("rendered-models")).toHaveTextContent("gpt-5.5claude-sonnet-4");
  });
});
