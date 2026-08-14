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
      classifier_context_per_turn_chars: 200,
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
    expect(screen.getByText("Context Per-Turn Character Limit")).toBeInTheDocument();
    expect(screen.getByDisplayValue("400")).toBeInTheDocument();
  });

  it("should default classifier context fields to 3 and 200 when llm is selected without explicit values", () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    const windowSizeSection = screen.getByText("Context Window Size").closest("div") as HTMLElement;
    expect(within(windowSizeSection).getByDisplayValue("3")).toBeInTheDocument();

    const perTurnCharsSection = screen.getByText("Context Per-Turn Character Limit").closest("div") as HTMLElement;
    expect(within(perTurnCharsSection).getByDisplayValue("200")).toBeInTheDocument();
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

  it("should call onChange with the updated classifier_context_per_turn_chars when edited", () => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    fireEvent.click(screen.getByText("Advanced: Classification Method"));

    const perTurnCharsSection = screen.getByText("Context Per-Turn Character Limit").closest("div") as HTMLElement;
    const input = within(perTurnCharsSection).getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "500" } });

    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classifier_context_per_turn_chars: 500,
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
    const input = within(keywordsSection).getByRole("combobox");
    await user.type(input, "udp,");
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp"]);
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

  // The dropdown is closed, so antd has nothing for Enter to select and the word would only land
  // on blur. Submitting used to provide that blur; it no longer can while the row reads as empty.
  it("commits a typed keyword on Enter, with the dropdown closed", async () => {
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
    await user.type(within(field).getByRole("combobox"), "invoice{enter}");

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
    const combobox = within(simpleTierSection).getByRole("combobox");
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
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toBeDisabled();
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
    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Classification Rubric" }));
    await userEvent.click(await screen.findByTitle("Chat"));
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
    expect(screen.getByTitle("Deep")).toBeInTheDocument();
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
