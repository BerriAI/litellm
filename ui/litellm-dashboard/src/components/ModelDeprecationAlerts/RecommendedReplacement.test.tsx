import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RecommendedReplacement from "./RecommendedReplacement";

const recommendation = {
  successor_model: "new-model",
  successor_model_source: "model_cost_map" as const,
  deprecation_source: "https://provider.example/model-retirements",
};

describe("recommended replacement", () => {
  it("should render the model with its cost-map source and provider notice", () => {
    render(<RecommendedReplacement recommendation={recommendation} />);
    expect(screen.getByText("new-model")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "LiteLLM model cost map" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json",
    );
    expect(screen.getByRole("link", { name: /Provider deprecation notice/ })).toHaveAttribute(
      "href",
      recommendation.deprecation_source,
    );
  });
  it("should identify deployment overrides without attributing them to the cost map", () => {
    render(
      <RecommendedReplacement recommendation={{ ...recommendation, successor_model_source: "deployment_config" }} />,
    );
    expect(screen.getByText(/Deployment configuration/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "LiteLLM model cost map" })).not.toBeInTheDocument();
  });
  it("should omit provenance when no replacement is provided", () => {
    render(<RecommendedReplacement recommendation={{}} />);
    expect(screen.getByText(/Not provided/)).toBeInTheDocument();
    expect(screen.queryByText(/Source:/)).not.toBeInTheDocument();
  });
  it("should avoid inventing a source when an older proxy only returns a model name", () => {
    render(<RecommendedReplacement recommendation={{ successor_model: "new-model" }} />);
    expect(screen.getByText(/Source: Not provided/)).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
  it("should omit provider links with an unsafe URL scheme", () => {
    render(
      <RecommendedReplacement recommendation={{ ...recommendation, deprecation_source: "javascript:alert(1)" }} />,
    );
    expect(screen.queryByRole("link", { name: /Provider deprecation notice/ })).not.toBeInTheDocument();
  });
});
