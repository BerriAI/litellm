import { fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ComplexityRouterConfig, { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const mockModelInfo = [
  { model_group: "gpt-4", mode: "chat" },
  { model_group: "gpt-3.5-turbo", mode: "chat" },
  { model_group: "claude-3-opus", mode: "chat" },
  { model_group: "text-embedding-3-small", mode: "embedding" },
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

const baseProps = {
  modelInfo: mockModelInfo,
  value: defaultValue,
  onChange: vi.fn(),
  keywordTierRules: [],
  onKeywordTierRulesChange: vi.fn(),
  semanticMatchingEnabled: false,
  onSemanticMatchingEnabledChange: vi.fn(),
  embeddingModel: undefined,
  onEmbeddingModelChange: vi.fn(),
  matchThreshold: 0.5,
  onMatchThresholdChange: vi.fn(),
};

describe("ComplexityRouterConfig", () => {
  it("should render", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Complexity Tier Configuration")).toBeInTheDocument();
  });

  it("should display all four tier labels", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Simple Tier")).toBeInTheDocument();
    expect(screen.getByText("Medium Tier")).toBeInTheDocument();
    expect(screen.getByText("Complex Tier")).toBeInTheDocument();
    expect(screen.getByText("Reasoning Tier")).toBeInTheDocument();
  });

  it("should show example queries for each tier", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText(/Hello!/)).toBeInTheDocument();
    expect(screen.getByText(/Explain how REST APIs work/)).toBeInTheDocument();
    expect(screen.getByText(/Design a microservices architecture/)).toBeInTheDocument();
    expect(screen.getByText(/Think step by step/)).toBeInTheDocument();
  });

  it("should display the how classification works section", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
  });

  it("should show score thresholds in the classification section", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
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
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Custom Technical Keywords")).toBeInTheDocument();
  });

  it("should display existing custom technical keywords as tags", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
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
        {...baseProps}
        customTechnicalKeywords={[]}
        onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
      />,
    );
    const keywordsCard = screen.getByText("Custom Technical Keywords").closest(".ant-card") as HTMLElement;
    const input = within(keywordsCard).getByRole("combobox");
    await user.type(input, "udp,");
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp"]);
  });

  it("should render an empty state when no keyword tier rules exist", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Keyword Tier Overrides")).toBeInTheDocument();
    expect(screen.getByText("No keyword tier overrides configured")).toBeInTheDocument();
  });

  it("hides the keyword-tier and semantic sections when their change handlers are absent (edit modal)", () => {
    // The edit-auto-router modal renders ComplexityRouterConfig without these handlers;
    // the sections must stay hidden rather than render interactive-but-dead controls.
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.queryByText("Keyword Tier Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Semantic keyword matching")).not.toBeInTheDocument();
    // Core tier config still renders.
    expect(screen.getByText("Complexity Tier Configuration")).toBeInTheDocument();
  });

  it("should call onKeywordTierRulesChange with a new rule when 'Add keyword rule' is clicked", async () => {
    const user = userEvent.setup();
    const onKeywordTierRulesChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onKeywordTierRulesChange={onKeywordTierRulesChange} />);
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(onKeywordTierRulesChange).toHaveBeenCalledTimes(1);
    const newRules = onKeywordTierRulesChange.mock.calls[0][0];
    expect(newRules).toHaveLength(1);
    expect(newRules[0]).toMatchObject({ keywords: [], tier: "COMPLEX" });
  });

  it("should render an existing keyword tier rule and remove it when the delete button is clicked", async () => {
    const user = userEvent.setup();
    const onKeywordTierRulesChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        keywordTierRules={[{ id: "rule-1", keywords: ["invoice", "refund"], tier: "MEDIUM" }]}
        onKeywordTierRulesChange={onKeywordTierRulesChange}
      />,
    );
    expect(screen.getByText("invoice")).toBeInTheDocument();
    expect(screen.getByText("refund")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove keyword rule 1/i }));
    expect(onKeywordTierRulesChange).toHaveBeenCalledWith([]);
  });

  it("should not show embedding model or match score fields when semantic matching is disabled", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={false} />);
    expect(screen.getByText("Semantic keyword matching")).toBeInTheDocument();
    expect(screen.queryByText("Embedding model")).not.toBeInTheDocument();
    expect(screen.queryByText("Minimum match score")).not.toBeInTheDocument();
  });

  it("should show embedding model and match score fields when semantic matching is enabled", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={true} />);
    expect(screen.getByText("Embedding model")).toBeInTheDocument();
    expect(screen.getByText("Minimum match score")).toBeInTheDocument();
  });

  it("should call onSemanticMatchingEnabledChange when the semantic matching switch is toggled", async () => {
    const user = userEvent.setup();
    const onSemanticMatchingEnabledChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        semanticMatchingEnabled={false}
        onSemanticMatchingEnabledChange={onSemanticMatchingEnabledChange}
      />,
    );
    await user.click(screen.getByRole("switch"));
    expect(onSemanticMatchingEnabledChange).toHaveBeenCalledWith(true, expect.anything());
  });

  it("excludes embedding-mode models from the tier and classifier dropdowns", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);

    const simpleTierSection = screen.getByText("Simple Tier").closest(".mb-4") as HTMLElement;
    const combobox = within(simpleTierSection).getByRole("combobox");
    await user.click(combobox);

    expect((await screen.findAllByText("gpt-3.5-turbo")).length).toBeGreaterThan(0);
    expect(screen.queryAllByText("text-embedding-3-small")).toHaveLength(0);
  });

  it("does not show tier validation errors by default", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByText("This tier is required")).not.toBeInTheDocument();
  });

  it("shows a validation error only under unfilled tiers when showValidationErrors is true", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tiers: { ...defaultValue.tiers, REASONING: "" } }}
        showValidationErrors={true}
      />,
    );
    expect(screen.getAllByText("This tier is required")).toHaveLength(1);
  });
});
