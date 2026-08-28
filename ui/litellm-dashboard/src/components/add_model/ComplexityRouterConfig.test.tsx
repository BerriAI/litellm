import { fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ComplexityRouterConfig, { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

const mockModelInfo = [
  {
    model_group: "gpt-4",
    mode: "chat",
    supports_reasoning: true,
    supported_reasoning_efforts: ["medium", "high", "xhigh"],
  },
  { model_group: "gpt-3.5-turbo", mode: "chat" },
  { model_group: "claude-3-opus", mode: "chat", supports_reasoning: true },
  { model_group: "text-embedding-3-small", mode: "embedding" },
] as any[];

const defaultValue: ComplexityRouterConfigValue = {
  tiers: {
    SIMPLE: ["gpt-3.5-turbo"],
    MEDIUM: ["gpt-3.5-turbo"],
    COMPLEX: ["gpt-4"],
    REASONING: ["claude-3-opus"],
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
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
  });

  it("should show score thresholds in the classification section", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
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

  it("should toggle returning the raw model name", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);

    await user.click(screen.getByText("Advanced: Response Format"));
    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith({
      ...defaultValue,
      return_raw_model_name: true,
    });
  });

  it("should reveal classifier model and timeout fields when llm is selected", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />);

    // Collapse panel content isn't rendered until first expanded.
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByText("LLM Classifier"));

    const expectedValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "", timeout_ms: 3000, classification_rubric: "agentic" },
      classifier_context_window_size: 3,
      classifier_context_budget_chars: 8000,
    };
    expect(onChange).toHaveBeenCalledWith(expectedValue);
  });

  it("should show classifier fields and use the configured values when classifier_type is llm", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 750 },
      classifier_context_window_size: 5,
      classifier_context_per_turn_chars: 400,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    expect(screen.getByText("Classifier Model")).toBeInTheDocument();
    expect(screen.getByText("Timeout (ms)")).toBeInTheDocument();
    expect(screen.getByDisplayValue("750")).toBeInTheDocument();
    expect(screen.getByText("Context Window Size")).toBeInTheDocument();
    expect(screen.getByDisplayValue("5")).toBeInTheDocument();
    expect(screen.queryByText("Context Per-Turn Character Limit")).not.toBeInTheDocument();
  });

  it("should default the context window and budget when llm is selected", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    const windowSizeSection = screen.getByText("Context Window Size").closest("div") as HTMLElement;
    expect(within(windowSizeSection).getByDisplayValue("3")).toBeInTheDocument();

    const budgetSection = screen.getByText("Context Character Budget").closest("div") as HTMLElement;
    expect(within(budgetSection).getByDisplayValue("8000")).toBeInTheDocument();
  });

  it("should warn when the budget is too small to quote any turn that does not already fit", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      classifier_context_budget_chars: 50,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    expect(screen.getByText(/no room to quote a turn/i)).toBeInTheDocument();
  });

  it("should not warn on a budget large enough to quote a turn, nor on a deliberate zero", () => {
    for (const budget of [120, 8000, 0]) {
      const { unmount } = renderWithProviders(
        <ComplexityRouterConfig
          modelInfo={mockModelInfo}
          value={{
            ...defaultValue,
            classifier_type: "llm",
            classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
            classifier_context_budget_chars: budget,
          }}
          onChange={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByText("Advanced: Classification Method"));
      expect(screen.queryByText(/no room to quote a turn/i)).not.toBeInTheDocument();
      unmount();
    }
  });

  it("should show the assistant-turns switch with its configured value when classifier_type is llm", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 750 },
      classifier_context_include_assistant_turns: true,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    expect(screen.getByText("Include Assistant Turns")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Include Assistant Turns" })).toBeChecked();
  });

  it("should render the assistant-turns switch off when it is not set", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    expect(screen.getByRole("switch", { name: "Include Assistant Turns" })).not.toBeChecked();
  });

  it("should hide the assistant-turns switch when classifier_type is heuristic", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.queryByText("Include Assistant Turns")).not.toBeInTheDocument();
  });

  it("should call onChange when the assistant-turns switch is toggled", () => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByRole("switch", { name: "Include Assistant Turns" }));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_context_include_assistant_turns: true }),
    );
  });

  it("should hide classifier context fields when classifier_type is heuristic", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.queryByText("Context Window Size")).not.toBeInTheDocument();
    expect(screen.queryByText("Context Per-Turn Character Limit")).not.toBeInTheDocument();
  });

  it("should call onChange with the updated classifier_context_window_size when edited", () => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    const windowSizeSection = screen.getByText("Context Window Size").closest("div") as HTMLElement;
    const input = within(windowSizeSection).getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "7" } });

    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classifier_context_window_size: 7,
    });
  });

  it("should render the custom technical keywords field", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
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
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
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
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    const keywordsSection = screen.getByText("Custom Technical Keywords").closest("div")?.parentElement as HTMLElement;
    await user.type(within(keywordsSection).getByRole("combobox"), "udp");
    await user.click(await screen.findByText('Create "udp"'));
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp"]);
  });

  it("splits a comma-separated keyword entry into one keyword per token", async () => {
    const user = userEvent.setup();
    const onCustomTechnicalKeywordsChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        customTechnicalKeywords={[]}
        onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    const keywordsSection = screen.getByText("Custom Technical Keywords").closest("div")?.parentElement as HTMLElement;
    await user.type(within(keywordsSection).getByRole("combobox"), "udp, kafka ,terraform");
    await user.click(await screen.findByText('Create "udp, kafka ,terraform"'));
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp", "kafka", "terraform"]);
  });

  it("should render an empty state when no keyword tier rules exist", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
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
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("button", { name: /add keyword rule/i }));
    expect(onKeywordTierRulesChange).toHaveBeenCalledTimes(1);
    const newRules = onKeywordTierRulesChange.mock.calls[0][0];
    expect(newRules).toHaveLength(1);
    expect(newRules[0]).toMatchObject({ keywords: [], tier: "COMPLEX" });
  });

  it("commits a typed keyword to the rule it was typed into", async () => {
    const user = userEvent.setup();
    const onKeywordTierRulesChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        keywordTierRules={[{ id: "rule-1", keywords: [], tier: "COMPLEX" }]}
        onKeywordTierRulesChange={onKeywordTierRulesChange}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));

    const field = screen.getByText("Keywords 1").closest("div") as HTMLElement;
    await user.type(within(field).getByRole("combobox"), "invoice");
    await user.click(await screen.findByText('Create "invoice"'));

    expect(onKeywordTierRulesChange).toHaveBeenCalledWith([{ id: "rule-1", keywords: ["invoice"], tier: "COMPLEX" }]);
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
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    expect(screen.getByText("invoice")).toBeInTheDocument();
    expect(screen.getByText("refund")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove keyword rule 1/i }));
    expect(onKeywordTierRulesChange).toHaveBeenCalledWith([]);
  });

  it("should not show embedding model or match score fields when semantic matching is disabled", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={false} />);
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    expect(screen.getByText("Semantic keyword matching")).toBeInTheDocument();
    expect(screen.queryByText("Embedding model")).not.toBeInTheDocument();
    expect(screen.queryByText("Minimum match score")).not.toBeInTheDocument();
  });

  it("should show embedding model and match score fields when semantic matching is enabled", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={true} />);
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
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
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    await user.click(screen.getByRole("switch"));
    expect(onSemanticMatchingEnabledChange).toHaveBeenCalledWith(true, expect.anything());
  });

  it("excludes embedding-mode models from the tier and classifier dropdowns", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);

    const simpleTierSection = screen.getByText("Simple Tier").closest(".mb-4") as HTMLElement;
    const combobox = within(simpleTierSection).getByRole("combobox", { name: "Select model(s) for simple queries" });
    await user.click(combobox);

    expect((await screen.findAllByText("gpt-3.5-turbo")).length).toBeGreaterThan(0);
    expect(screen.queryAllByText("text-embedding-3-small")).toHaveLength(0);
  });

  it("does not show tier validation errors by default", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByText("This tier is required")).not.toBeInTheDocument();
  });

  it("shows an inline error on the classifier model select when llm is selected without a model", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={llmValue} showValidationErrors={true} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByText("A classifier model is required")).toBeInTheDocument();
  });

  it("does not show the classifier model error once a classifier model is set", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={llmValue} showValidationErrors={true} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.queryByText("A classifier model is required")).not.toBeInTheDocument();
  });

  it("shows a validation error only under unfilled tiers when showValidationErrors is true", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tiers: { ...defaultValue.tiers, REASONING: [] } }}
        showValidationErrors={true}
      />,
    );
    expect(screen.getByText("The Reasoning tier is required")).toBeInTheDocument();
    expect(screen.getAllByText(/tier is required/)).toHaveLength(1);
  });

  it("renders the escalation keywords section with current keywords when the handler is provided", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        escalationKeywords={["LITELLM ESCALATE"]}
        onEscalationKeywordsChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Escalation Keywords"));
    expect(screen.getByText("Escalation Keywords")).toBeInTheDocument();
    expect(screen.getByText("LITELLM ESCALATE")).toBeInTheDocument();
  });

  it("hides the escalation keywords section when no handler is provided", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByText("Advanced: Escalation Keywords")).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig classifier fallback", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
  };

  it("defaults the fallback to the heuristic, matching the backend field default", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByRole("radio", { name: /Score with the heuristic/ })).toBeChecked();
  });

  it("records a switch to the default model fallback", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByRole("radio", { name: /Route to the default model/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ classifier_fallback: "default_model" }));
  });

  it("disables the default model fallback when no tier would produce one", () => {
    // The deployment's default model is derived from the tiers on submit, so offering the option
    // with no tiers picked would save a config the backend rejects at startup.
    const noTiers: ComplexityRouterConfigValue = {
      ...llmValue,
      tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={noTiers} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("hides the fallback choice for the heuristic classifier, which has nothing to fall back from", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.queryByText("If the classifier fails")).not.toBeInTheDocument();
  });

  it("stops describing the heuristic as the fallback once a custom prompt routes failures to the default model", () => {
    // With both set, the heuristic scorer never runs, so the panel must not keep implying a
    // score decides anything on this router.
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{
          ...llmValue,
          classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, system_prompt: "Grade data sensitivity" },
          classifier_fallback: "default_model",
        }}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByText(/no longer runs at all/)).toBeInTheDocument();
  });

  it("still describes the heuristic as the fallback when a custom prompt keeps heuristic fallback", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{
          ...llmValue,
          classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, system_prompt: "Grade data sensitivity" },
        }}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByText(/only when the classifier call fails/)).toBeInTheDocument();
  });

  it("clears a stored fallback when switching back to the heuristic classifier", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{ ...llmValue, classifier_fallback: "default_model" }}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByRole("radio", { name: /rule-based scoring/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ classifier_fallback: undefined }));
  });
});

describe("ComplexityRouterConfig classifier rubric", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
  };

  const openClassificationPanel = (value: ComplexityRouterConfigValue, onChange = vi.fn()) => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={value} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    return onChange;
  };

  it("shows an existing router with no stored preset as legacy, not as the calibrated default", () => {
    // This router predates the setting. Displaying a calibrated preset it does not have would tell the
    // operator their traffic is graded by examples the classifier never receives, and saving the form
    // unchanged would then move its tier decisions.
    openClassificationPanel(llmValue);
    expect(screen.getByText("Legacy (uncalibrated)")).toBeInTheDocument();
    expect(screen.getByText(/tier decisions and spend are unchanged/)).toBeInTheDocument();
  });

  it("stamps the calibrated preset on a classifier being switched on for the first time", () => {
    // A heuristic router turning on the LLM classifier has no prior tier behaviour to preserve, so a
    // newly configured classifier starts on the calibrated rubric rather than the legacy one.
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    fireEvent.click(screen.getByText("LLM Classifier"));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_llm_config: expect.objectContaining({ classification_rubric: "agentic" }) }),
    );
  });

  it("shows the calibrated preset when a router stores one", () => {
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, classification_rubric: "agentic" },
    });
    expect(screen.getByText("Agentic")).toBeInTheDocument();
    expect(screen.getByText(/does not route to your most expensive tier/)).toBeInTheDocument();
  });

  it("records the chat preset the operator picks", async () => {
    const onChange = openClassificationPanel(llmValue);
    await userEvent.click(screen.getByRole("combobox", { name: "Classification Rubric" }));
    await userEvent.click(await screen.findByRole("option", { name: "Chat" }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_llm_config: expect.objectContaining({ classification_rubric: "chat" }) }),
    );
  });

  it("shows the stored preset when editing a router already on chat", () => {
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, classification_rubric: "chat" },
    });
    expect(screen.getByText(/only conversational traffic/)).toBeInTheDocument();
  });

  it("records the business preset the operator picks", async () => {
    const onChange = openClassificationPanel(llmValue);
    await userEvent.click(screen.getByRole("combobox", { name: "Classification Rubric" }));
    await userEvent.click(await screen.findByRole("option", { name: "Business" }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier_llm_config: expect.objectContaining({ classification_rubric: "business" }),
      }),
    );
  });

  it("shows the stored preset when editing a router already on business", () => {
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, classification_rubric: "business" },
    });
    expect(screen.getByText(/business-oriented tier definitions/)).toBeInTheDocument();
  });

  it("disables the preset once a custom prompt replaces the rubric it would select", () => {
    // The backend rejects both together, so the picker must not look like it still applies.
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, system_prompt: "Grade data sensitivity" },
    });
    expect(screen.getByText(/the custom prompt below is the classifier's entire rubric/)).toBeInTheDocument();
  });

  it("hides the preset for the heuristic classifier, which sends no prompt at all", () => {
    openClassificationPanel(defaultValue);
    expect(screen.queryByRole("combobox", { name: "Classification Rubric" })).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig tier labels", () => {
  const renamedValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    tier_labels: { SIMPLE: "Cheap", MEDIUM: "Standard", COMPLEX: "Premium", REASONING: "Deep" },
  };

  it("shows the operator's names in the tier headers instead of the defaults", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByText("Cheap Tier")).toBeInTheDocument();
    expect(screen.getByText("Deep Tier")).toBeInTheDocument();
    expect(screen.queryByText("Simple Tier")).not.toBeInTheDocument();
    expect(screen.queryByText("Reasoning Tier")).not.toBeInTheDocument();
  });

  it("keeps the rung ordinal and canonical name visible under a rename", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByText(/Tier 1 of 4/)).toHaveTextContent("Tier 1 of 4 · SIMPLE");
    expect(screen.getByText(/Tier 4 of 4/)).toHaveTextContent("Tier 4 of 4 · REASONING");
  });

  it("names the renamed tier in the required-field error", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...renamedValue, tiers: { ...defaultValue.tiers, REASONING: [] } }}
        showValidationErrors={true}
      />,
    );
    expect(screen.getByText("The Deep tier is required")).toBeInTheDocument();
  });

  it("reports a typed label back to the caller under its canonical tier key", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Display name for the Simple tier"), { target: { value: "Cheap" } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ tier_labels: { SIMPLE: "Cheap" } }));
  });

  it("shows a stored label in its input so an edit round-trips", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByLabelText("Display name for the Reasoning tier")).toHaveValue("Deep");
  });

  it("leaves the label inputs empty when nothing was renamed", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByLabelText("Display name for the Simple tier")).toHaveValue("");
  });

  it("uses the operator's names in the classification score table", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByText("Cheap")).toBeInTheDocument();
    expect(screen.getByText("Deep")).toBeInTheDocument();
  });

  it("uses the operator's names in the keyword rule tier picker", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={renamedValue}
        keywordTierRules={[{ id: "r1", keywords: ["invoice"], tier: "REASONING" }]}
      />,
    );
    fireEvent.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
    expect(screen.getByRole("combobox", { name: "Route keyword rule 1 to tier" })).toHaveTextContent("Deep");
  });
});

describe("ComplexityRouterConfig affinity panel", () => {
  it("holds both affinity switches with their backend defaults", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    fireEvent.click(screen.getByText("Advanced: Affinity"));

    expect(screen.getByRole("switch", { name: "Pin a session to one deployment per model group" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Pin a session to its first model" })).not.toBeChecked();
  });

  it("writes deployment_affinity through onChange without touching other keys", () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Affinity"));

    fireEvent.click(screen.getByRole("switch", { name: "Pin a session to one deployment per model group" }));

    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, deployment_affinity: false });
  });

  it("renders a stored deployment_affinity=false as off", () => {
    renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={{ ...defaultValue, deployment_affinity: false }} />,
    );
    fireEvent.click(screen.getByText("Advanced: Affinity"));

    expect(screen.getByRole("switch", { name: "Pin a session to one deployment per model group" })).not.toBeChecked();
  });
});

describe("ComplexityRouterConfig default model", () => {
  const getDefaultModelSelect = () => screen.getByRole("combobox", { name: "Default model" });

  it("shows what the tiers currently imply, so an untouched router still names its default", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(getDefaultModelSelect()).toHaveAttribute("placeholder", "Derived from tiers: gpt-3.5-turbo");
  });

  it("asks for a model rather than naming a derived one when no tier holds one", () => {
    const noTiers: ComplexityRouterConfigValue = {
      ...defaultValue,
      tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={noTiers} />);
    expect(getDefaultModelSelect()).toHaveAttribute("placeholder", "Add a model to the Simple or Medium tier");
  });

  it("records a pinned model", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);

    await user.click(getDefaultModelSelect());
    await user.click(await screen.findByRole("option", { name: "claude-3-opus" }));

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ default_model: "claude-3-opus" }));
  });

  it("drops the key when the pin is cleared, so an emptied select reads as tier-tracking", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const pinned: ComplexityRouterConfigValue = { ...defaultValue, default_model: "claude-3-opus" };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinned} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ default_model: undefined }));
  });

  it("shows a pinned model as the selection instead of the tier-derived one", () => {
    const pinned: ComplexityRouterConfigValue = { ...defaultValue, default_model: "claude-3-opus" };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinned} />);
    expect(getDefaultModelSelect()).toHaveValue("claude-3-opus");
  });

  it("unlocks the default model fallback on a pin alone, with no tier to derive from", () => {
    const pinnedNoTiers: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
      default_model: "claude-3-opus",
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinnedNoTiers} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).not.toHaveAttribute("aria-disabled");
  });

  it("names the resolved default on the fallback option, so the destination is not a guess", () => {
    const pinned: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      default_model: "claude-3-opus",
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinned} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
    expect(screen.getByRole("radio", { name: /Route to the default model \(claude-3-opus\)/ })).toBeInTheDocument();
  });
});

describe("plan-mode override", () => {
  const openPanel = () => fireEvent.click(screen.getByText("Advanced: Plan-Mode Override"));
  const switchName = "Route plan-mode requests to a minimum tier";

  it("toggling on floors at the highest tier that has models", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openPanel();
    fireEvent.click(await screen.findByRole("switch", { name: switchName }));
    expect(onChange.mock.calls.at(-1)?.[0].plan_mode_min_tier).toBe("REASONING");
  });

  it("toggling off drops the key entirely instead of storing an empty value", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, plan_mode_min_tier: "COMPLEX" }}
        onChange={onChange}
      />,
    );
    openPanel();
    const control = await screen.findByRole("switch", { name: switchName });
    expect(control).toBeChecked();
    fireEvent.click(control);
    const updated = onChange.mock.calls.at(-1)?.[0];
    expect(updated.plan_mode_min_tier).toBeUndefined();
  });

  it("only offers tiers that have models, since the backend rejects a floor at an empty tier", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{
          ...defaultValue,
          tiers: { ...defaultValue.tiers, REASONING: [] },
          plan_mode_min_tier: "COMPLEX",
        }}
      />,
    );
    openPanel();
    await userEvent.click(await screen.findByRole("combobox", { name: "Plan-mode minimum tier" }));
    expect(await screen.findByRole("option", { name: "Medium" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Reasoning" })).not.toBeInTheDocument();
  });

  it("disables the toggle until some tier has models", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] } }}
      />,
    );
    openPanel();
    expect(await screen.findByRole("switch", { name: switchName })).toHaveAttribute("aria-disabled", "true");
  });
});

describe("ComplexityRouterConfig per-model reasoning effort", () => {
  it("renders one effort select per selected model, defaulting to Default", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    const select = screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" });
    expect(select).toHaveTextContent("Default");
  });

  it("shows the hydrated effort for a model that has one stored", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tier_model_params: { COMPLEX: { "gpt-4": { reasoning_effort: "high" } } } }}
      />,
    );
    const select = screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" });
    expect(select).toHaveTextContent("high");
  });

  it("emits tier_model_params scoped to the tier and model when an effort is picked", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" }));
    await user.click(await screen.findByRole("option", { name: "high" }));
    expect(onChange).toHaveBeenCalledWith({
      ...defaultValue,
      tier_model_params: { COMPLEX: { "gpt-4": { reasoning_effort: "high" } } },
    });
  });

  it("picking Default removes the stored effort", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tier_model_params: { COMPLEX: { "gpt-4": { reasoning_effort: "high" } } } }}
        onChange={onChange}
      />,
    );
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" }));
    await user.click(await screen.findByRole("option", { name: "Default" }));
    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, tier_model_params: undefined });
  });
});

describe("ComplexityRouterConfig reasoning effort gating", () => {
  it("offers no effort select for a model group without reasoning support", () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(
      screen.queryByRole("combobox", { name: "Reasoning effort for gpt-3.5-turbo in the Simple tier" }),
    ).not.toBeInTheDocument();
  });

  // A stored effort on a model the group info calls non-reasoning must stay visible, or the
  // operator has no way to clear it.
  it("keeps the select for a non-reasoning model that already has a stored effort", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tier_model_params: { SIMPLE: { "gpt-3.5-turbo": { reasoning_effort: "low" } } } }}
      />,
    );
    expect(
      screen.getByRole("combobox", { name: "Reasoning effort for gpt-3.5-turbo in the Simple tier" }),
    ).toHaveTextContent("low");
  });
});

describe("ComplexityRouterConfig per-model effort filtering", () => {
  it("offers only the efforts the model group supports", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" }));
    const options = (await screen.findAllByRole("option")).map((option) => option.textContent);
    expect(options).toEqual(["Default", "medium", "high", "xhigh"]);
  });

  it("falls back to every effort when the group only reports supports_reasoning", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("combobox", { name: "Reasoning effort for claude-3-opus in the Reasoning tier" }),
    );
    const options = (await screen.findAllByRole("option")).map((option) => option.textContent);
    expect(options).toEqual(["Default", "none", "minimal", "low", "medium", "high", "xhigh"]);
  });

  // An empty list is the group's own answer that its deployments share no level, which is different
  // from the field being absent, so the control is dropped rather than falling back to every level.
  it("offers no effort at all when the group intersects to nothing", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        modelInfo={[
          ...mockModelInfo.filter((model) => model.model_group !== "claude-3-opus"),
          { model_group: "claude-3-opus", mode: "chat", supports_reasoning: true, supported_reasoning_efforts: [] },
        ]}
      />,
    );
    expect(
      screen.queryByRole("combobox", { name: "Reasoning effort for claude-3-opus in the Reasoning tier" }),
    ).not.toBeInTheDocument();
  });

  // Hand-authored configs can carry a level outside the supported set (e.g. max); it must render
  // and stay clearable rather than being masked as Default.
  it("keeps showing a stored effort outside the supported set", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, tier_model_params: { COMPLEX: { "gpt-4": { reasoning_effort: "max" } } } }}
      />,
    );
    expect(screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" })).toHaveTextContent(
      "max",
    );
  });
});

describe("ComplexityRouterConfig custom technical keywords", () => {
  const openClassificationPanel = (value: ComplexityRouterConfigValue) => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={value} onChange={vi.fn()} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));
  };

  const llmConfig = { model: "gpt-3.5-turbo", timeout_ms: 3000 };

  it.each([
    ["heuristic", { ...defaultValue, classifier_type: "heuristic" as const }],
    [
      "heuristic_first",
      {
        ...defaultValue,
        classifier_type: "heuristic_first" as const,
        heuristic_first_max_tier: "SIMPLE",
        classifier_llm_config: llmConfig,
      },
    ],
    [
      "llm falling back to the scorer",
      {
        ...defaultValue,
        classifier_type: "llm" as const,
        classifier_llm_config: llmConfig,
        classifier_fallback: "heuristic" as const,
      },
    ],
  ])("offers the keywords on a router whose scorer runs: %s", (_label, value) => {
    openClassificationPanel(value);
    expect(screen.getByText("Custom Technical Keywords")).toBeInTheDocument();
  });

  it("hides the keywords when the scorer never runs, so they cannot imply an effect they have none", () => {
    openClassificationPanel({
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: llmConfig,
      classifier_fallback: "default_model",
    });
    expect(screen.queryByText("Custom Technical Keywords")).not.toBeInTheDocument();
  });
});
