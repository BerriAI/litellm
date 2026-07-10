import { fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ComplexityRouterConfig, { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const mockModelInfo = [
  { model_group: "gpt-4" },
  { model_group: "gpt-3.5-turbo" },
  { model_group: "claude-3-opus" },
] as any[];

const defaultValue: ComplexityRouterConfigValue = {
  tiers: {
    SIMPLE: "gpt-3.5-turbo",
    MEDIUM: "gpt-3.5-turbo",
    COMPLEX: "gpt-4",
    REASONING: "claude-3-opus",
  },
  classifier_type: "heuristic",
};

describe("ComplexityRouterConfig", () => {
  it("should render", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText("Complexity Tier Configuration")).toBeInTheDocument();
  });

  it("should display all four tier labels", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText("Simple Tier")).toBeInTheDocument();
    expect(screen.getByText("Medium Tier")).toBeInTheDocument();
    expect(screen.getByText("Complex Tier")).toBeInTheDocument();
    expect(screen.getByText("Reasoning Tier")).toBeInTheDocument();
  });

  it("should show example queries for each tier", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText(/Hello!/)).toBeInTheDocument();
    expect(screen.getByText(/Explain how REST APIs work/)).toBeInTheDocument();
    expect(screen.getByText(/Design a microservices architecture/)).toBeInTheDocument();
    expect(screen.getByText(/Think step by step/)).toBeInTheDocument();
  });

  it("should display the how classification works section", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
  });

  it("should show score thresholds in the classification section", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText(/Score < 0.15/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.15 - 0.35/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.35 - 0.60/)).toBeInTheDocument();
    expect(screen.getByText(/Score > 0.60/)).toBeInTheDocument();
  });

  it("should default to heuristic and hide classifier model/timeout fields", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText("Advanced: Classification Method")).toBeInTheDocument();
    expect(screen.queryByText("Classifier Model")).not.toBeInTheDocument();
  });

  it("should reveal classifier model and timeout fields when llm is selected", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />);

    // Collapse panel content isn't rendered until first expanded.
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByText("LLM Classifier"));

    expect(onChange).toHaveBeenCalledWith({
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "", timeout_ms: 3000 },
    });
  });

  it("should show classifier fields and use the configured values when classifier_type is llm", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 750 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    expect(screen.getByText("Classifier Model")).toBeInTheDocument();
    expect(screen.getByText("Timeout (ms)")).toBeInTheDocument();
    expect(screen.getByDisplayValue("750")).toBeInTheDocument();
  });

  it("should render the custom technical keywords field", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.getByText("Custom Technical Keywords")).toBeInTheDocument();
  });

  it("should display existing custom technical keywords as tags", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={defaultValue}
        onChange={vi.fn()}
        customTechnicalKeywords={["udp", "kafka"]}
        onCustomTechnicalKeywordsChange={vi.fn()}
      />,
    );
    expect(screen.getByText("udp")).toBeInTheDocument();
    expect(screen.getByText("kafka")).toBeInTheDocument();
  });

  it("should call onCustomTechnicalKeywordsChange when a keyword is entered", async () => {
    const user = userEvent.setup();
    const onCustomTechnicalKeywordsChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={defaultValue}
        onChange={vi.fn()}
        customTechnicalKeywords={[]}
        onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
      />,
    );
    const keywordsCard = screen.getByText("Custom Technical Keywords").closest(".ant-card") as HTMLElement;
    const input = within(keywordsCard).getByRole("combobox");
    await user.type(input, "udp,");
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp"]);
  });
});
