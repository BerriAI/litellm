import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import DecisionModelsBanner from "./DecisionModelsBanner";

const STORAGE_KEY = "hideDecisionModelsBanner";

const modelCostMap = vi.fn();

vi.mock("@/components/networking", () => ({
  modelCostMap: (...args: unknown[]) => modelCostMap(...args),
}));

const COST_MAP = {
  "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
  "gpt-6-luna": { litellm_provider: "openai", mode: "chat", supported_endpoints: ["/v1/decisions"] },
  "gpt-5.5": { litellm_provider: "openai", mode: "chat", supported_endpoints: ["/v1/chat/completions"] },
  "azure_ai/Microsoft-Decision-1": { litellm_provider: "azure_ai", mode: "evaluation" },
  "anthropic/claude-opus-5.5": { litellm_provider: "anthropic", mode: "chat" },
};

describe("DecisionModelsBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem("hideCostOptimizationFeedbackBanner");
    testQueryClient.clear();
    modelCostMap.mockReset();
    modelCostMap.mockResolvedValue(COST_MAP);
  });

  it("lists the providers that have decision models in the catalog, linked to the docs table", async () => {
    renderWithProviders(<DecisionModelsBanner />);

    const link = await screen.findByRole("link", { name: "3 providers" });

    expect(link).toHaveAttribute("href", "https://docs.litellm.ai/docs/decisions#supported-providers");
    expect(screen.getByRole("paragraph")).toHaveTextContent(
      "Works with 3 providers: Azure AI Foundry (Studio), OpenAI, TypeSafe. Call them at /v1/decisions",
    );
    expect(modelCostMap).toHaveBeenCalledWith(true);
  });

  it("still renders the announcement when the catalog cannot be loaded", async () => {
    modelCostMap.mockRejectedValue(new Error("offline"));
    renderWithProviders(<DecisionModelsBanner />);

    await waitFor(() => expect(testQueryClient.getQueryCache().findAll({ status: "error" })).toHaveLength(1));

    expect(screen.getByRole("link", { name: "How to call them" })).toBeInTheDocument();
    expect(screen.getByRole("paragraph")).not.toHaveTextContent("Works with");
    expect(screen.getByRole("paragraph")).toHaveTextContent("Call them at /v1/decisions or /v1/systemone");
  });

  it("links directly to the System One playground", () => {
    renderWithProviders(<DecisionModelsBanner />);
    const link = screen.getByRole("link", { name: "Try decision models" });
    expect(link).toHaveAttribute("href", "/ui/playground?tab=system-one");
  });

  it("links to the docs on how to call decision models", () => {
    renderWithProviders(<DecisionModelsBanner />);
    expect(screen.getByRole("link", { name: "How to call them" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/decisions",
    );
  });

  it("offers Add a decision model only when the page can add models, and calls back on click", () => {
    const onAddModel = vi.fn();
    const { rerender } = renderWithProviders(<DecisionModelsBanner />);
    expect(screen.queryByRole("button", { name: "Add a decision model" })).not.toBeInTheDocument();

    rerender(<DecisionModelsBanner onAddModel={onAddModel} />);
    fireEvent.click(screen.getByRole("button", { name: "Add a decision model" }));

    expect(onAddModel).toHaveBeenCalledTimes(1);
  });

  it("points only users who can add models at Add Model", async () => {
    const { rerender } = renderWithProviders(<DecisionModelsBanner />);
    await screen.findByRole("link", { name: "3 providers" });
    expect(screen.getByRole("paragraph")).not.toHaveTextContent("Add Model");
    expect(screen.getByRole("paragraph")).toHaveTextContent("Call them at /v1/decisions or /v1/systemone");

    rerender(<DecisionModelsBanner onAddModel={vi.fn()} />);

    expect(screen.getByRole("paragraph")).toHaveTextContent(
      "OpenAI, TypeSafe. Search decision in Add Model to find them. Call them at /v1/decisions",
    );
  });

  it("shows the announcement even if the old feedback banner was dismissed", () => {
    localStorage.setItem("hideCostOptimizationFeedbackBanner", "true");
    renderWithProviders(<DecisionModelsBanner />);
    expect(screen.getByText("Decision models are now supported")).toBeInTheDocument();
  });

  it("hides itself and persists the dismissal when the dismiss button is clicked", () => {
    renderWithProviders(<DecisionModelsBanner />);
    expect(screen.getByText("Decision models are now supported")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Dismiss banner"));

    expect(screen.queryByText("Decision models are now supported")).not.toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("stays dismissed on remount once persisted, without fetching the catalog", () => {
    localStorage.setItem(STORAGE_KEY, "true");
    renderWithProviders(<DecisionModelsBanner />);
    expect(screen.queryByText("Decision models are now supported")).not.toBeInTheDocument();
    expect(modelCostMap).not.toHaveBeenCalled();
  });
});
