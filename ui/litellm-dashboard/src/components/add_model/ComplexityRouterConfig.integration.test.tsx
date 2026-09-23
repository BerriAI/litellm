import { openAutoRouterAdvanced, selectAutoRouterOption } from "../../../tests/autoRouterSetup";
import { fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, it, expect, vi, type Mock } from "vitest";
import ComplexityRouterConfigView, { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
const ComplexityRouterConfig = (props: React.ComponentProps<typeof ComplexityRouterConfigView>) => (
  <AutoRouterClassifierTabs value={props.value} onChange={props.onChange}>
    <ComplexityRouterConfigView {...props} />
  </AutoRouterClassifierTabs>
);
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
  it("should render", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Models by tier")).toBeInTheDocument();
  });

  it("should display all four tier labels", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText("Simple Tier")).toBeInTheDocument();
    expect(screen.getByText("Medium Tier")).toBeInTheDocument();
    expect(screen.getByText("Complex Tier")).toBeInTheDocument();
    expect(screen.getByText("Reasoning Tier")).toBeInTheDocument();
  });

  it("should show example queries for each tier", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByText(/Hello!/)).toBeInTheDocument();
    expect(screen.getByText(/Explain how REST APIs work/)).toBeInTheDocument();
    expect(screen.getByText(/Design a microservices architecture/)).toBeInTheDocument();
    expect(screen.getByText(/Think step by step/)).toBeInTheDocument();
  });

  it("should display the how classification works section", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
  });

  it("should show score thresholds in the classification section", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText(/Score < 0.15/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.15 - 0.35/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.35 - 0.60/)).toBeInTheDocument();
    expect(screen.getByText(/Score > 0.60/)).toBeInTheDocument();
  });

  it("leaves the score threshold list color to the theme instead of an inline style", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Classification Method");
    const list = screen.getByText(/Score < 0.15/).closest<HTMLUListElement>("ul");
    expect(list).toBeInTheDocument();
    expect(list).toHaveClass("text-muted-foreground");
    expect(list?.style.color).toBe("");
  });

  it("should default to heuristic and hide classifier model/timeout fields", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("Classifier tuning")).toBeInTheDocument();
    expect(screen.queryByText("Judge model")).not.toBeInTheDocument();
  });

  it("shows heuristic advanced sections and hides keyword overrides for capability classifiers", async () => {
    const { rerender } = renderWithProviders(<ComplexityRouterConfig {...baseProps} />);

    openAutoRouterAdvanced("Heuristic Keyword Overrides");

    expect(screen.getByText("Heuristic Keyword Overrides")).toBeInTheDocument();
    openAutoRouterAdvanced("Housekeeping Routing");
    expect(screen.getByText("Housekeeping Routing")).toBeInTheDocument();
    openAutoRouterAdvanced("Ignore Custom Tags");
    expect(screen.getByText("Ignore Custom Tags")).toBeInTheDocument();

    const capabilityValue = { ...defaultValue, classifier_type: "capability" as const };
    rerender(<ComplexityRouterConfig {...baseProps} value={capabilityValue} />);
    expect(screen.queryByText("Heuristic Keyword Overrides")).not.toBeInTheDocument();
  });

  it.each([
    ["custom", true],
    ["heuristic", false],
  ] as const)("shows plugin timeout only for %s classifiers", (classifierType, visible) => {
    renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={{ ...defaultValue, classifier_type: classifierType }} />,
    );
    openAutoRouterAdvanced("Classification Method");
    if (visible) {
      expect(screen.getByLabelText("Classifier plugin timeout (ms)")).toBeInTheDocument();
    } else {
      expect(screen.queryByLabelText("Classifier plugin timeout (ms)")).not.toBeInTheDocument();
    }
  });

  it.each([true, false])("shows reminder marker validation only when requested: %s", (showValidationErrors) => {
    const value = { ...defaultValue, reminder_markers: [{ open: "", close: "x" }] };
    renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={value} showValidationErrors={showValidationErrors} />,
    );
    openAutoRouterAdvanced("Ignore Custom Tags");
    const validation = screen.queryByText(/needs both/i);
    if (showValidationErrors) {
      expect(validation).toBeInTheDocument();
    } else {
      expect(validation).not.toBeInTheDocument();
    }
  });

  it("disables housekeeping sentinels when cheapest-tier routing is off", async () => {
    renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={{ ...defaultValue, route_housekeeping_to_cheapest_tier: false }} />,
    );
    openAutoRouterAdvanced("Housekeeping Routing");
    const sentinelInput = screen.getByRole("combobox", { name: "e.g., conversation title" });
    expect(sentinelInput).toBeDisabled();
  });

  it("should toggle returning the raw model name", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);

    openAutoRouterAdvanced("Response Format");
    await user.click(screen.getByRole("switch", { name: "Return raw model name" }));

    expect(onChange).toHaveBeenCalledWith({
      ...defaultValue,
      return_raw_model_name: true,
    });
  });

  it("should reveal classifier model and timeout fields when llm is selected", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />);

    // Collapse panel content isn't rendered until first expanded.
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("radio", { name: /^LLM$/ }));

    const expectedValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "", timeout_ms: 3000, classification_rubric: "agentic" },
      classifier_context_window_size: 3,
      classifier_context_budget_chars: 8000,
    };
    expect(onChange).toHaveBeenCalledWith(expectedValue);
  });

  it("selects heuristic v2 without requiring a classifier model or showing weighted scoring", async () => {
    const onChange = vi.fn();
    const { rerender } = renderWithProviders(
      <ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />,
    );

    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("Heuristic", "Heuristic v2");

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier_type: "heuristic_v2",
        classifier_llm_config: undefined,
      }),
    );

    const heuristicV2Value: ComplexityRouterConfigValue = { ...defaultValue, classifier_type: "heuristic_v2" };
    rerender(<ComplexityRouterConfig modelInfo={mockModelInfo} value={heuristicV2Value} onChange={onChange} />);

    expect(screen.queryByText("Judge model")).not.toBeInTheDocument();
    expect(screen.queryByText("Advanced scoring")).not.toBeInTheDocument();
    expect(screen.getByText(/estimates success probability for all four tiers/)).toBeInTheDocument();
    expect(screen.queryByText(/Score < 0.15/)).not.toBeInTheDocument();
  });

  it.each<[string, Partial<ComplexityRouterConfigValue>]>([
    ["heuristic", { classifier_type: "heuristic" }],
    ["LLM", { classifier_type: "llm" }],
    ["heuristic first", { classifier_type: "heuristic_first" }],
    ["hybrid", { classifier_type: "hybrid" }],
    ["Capability", { classifier_type: "capability" }],
    ["Fuse v2", { classifier_type: "llm_v2" }],
    [
      "custom tiers",
      {
        classifier_type: "heuristic_v2",
        custom_tier_set: {
          tiers: [{ id: "review", name: "REVIEW", definition: "Review code", models: ["gpt-4"] }],
          fallback_tier_id: "review",
        },
      },
    ],
  ])("shows and clears an invalid inactive threshold under %s", (_label, overrides) => {
    const value = { ...defaultValue, ...overrides, heuristic_v2_success_threshold: Number.NaN };
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={value} onChange={onChange} />);
    const retained = screen.getByRole("region", { name: "Inactive Heuristic v2 threshold" });
    expect(within(retained).getByRole("status", { name: "Retained Heuristic v2 threshold" })).toHaveTextContent(
      "Invalid value",
    );
    expect(within(retained).getByRole("alert")).toHaveTextContent("Success threshold must be a number between 0 and 1");
    fireEvent.click(within(retained).getByRole("button", { name: "Clear Heuristic v2 threshold" }));
    expect(onChange).toHaveBeenCalledWith({ ...value, heuristic_v2_success_threshold: undefined });
  });

  it("shows an inactive zero threshold until explicitly cleared and hides the summary for active or absent values", async () => {
    const onChange = vi.fn();
    const value = { ...defaultValue, heuristic_v2_success_threshold: 0 };
    const { rerender } = renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={value} onChange={onChange} />,
    );
    expect(screen.getByRole("status", { name: "Retained Heuristic v2 threshold" })).toHaveTextContent("0");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    rerender(<ComplexityRouterConfig {...baseProps} value={{ ...value, classifier_type: "heuristic_v2" }} />);
    expect(screen.queryByRole("region", { name: "Inactive Heuristic v2 threshold" })).not.toBeInTheDocument();
    rerender(<ComplexityRouterConfig {...baseProps} value={defaultValue} />);
    expect(screen.queryByRole("region", { name: "Inactive Heuristic v2 threshold" })).not.toBeInTheDocument();
  });

  it("should show classifier fields and use the configured values when classifier_type is llm", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 750 },
      classifier_context_window_size: 5,
      classifier_context_per_turn_chars: 400,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    openAutoRouterAdvanced("Classification Method");

    expect(screen.getByText("Judge model")).toBeInTheDocument();
    expect(screen.getByLabelText("Timeout (ms)")).toHaveValue("750");
    expect(screen.getByRole("switch", { name: "Classifier circuit breaker" })).toBeChecked();
    expect(screen.getByLabelText("Circuit breaker cooldown (seconds)")).toHaveValue("30");
    expect(screen.getByLabelText("Context Window Size")).toHaveValue("5");
    expect(screen.queryByText("Context Per-Turn Character Limit")).not.toBeInTheDocument();
  });

  it("should allow the default-on classifier circuit breaker to be disabled", async () => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");

    fireEvent.click(screen.getByRole("switch", { name: "Classifier circuit breaker" }));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier_llm_config: expect.objectContaining({ circuit_breaker_enabled: false }),
      }),
    );
  });

  it("should default the context window and budget when llm is selected", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    openAutoRouterAdvanced("Classification Method");

    expect(screen.getByLabelText("Context Window Size")).toHaveValue("3");
    expect(screen.getByLabelText("Context Character Budget")).toHaveValue("8000");
  });

  it("should warn when the budget is too small to quote any turn that does not already fit", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      classifier_context_budget_chars: 50,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    openAutoRouterAdvanced("Classification Method");

    expect(screen.getByText(/no room to quote a turn/i)).toBeInTheDocument();
  });

  it("should not warn on a budget large enough to quote a turn, nor on a deliberate zero", async () => {
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
      openAutoRouterAdvanced("Classification Method");
      expect(screen.queryByText(/no room to quote a turn/i)).not.toBeInTheDocument();
      unmount();
    }
  });

  it("should show the assistant-turns switch with its configured value when classifier_type is llm", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 750 },
      classifier_context_include_assistant_turns: true,
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    openAutoRouterAdvanced("Classification Method");

    expect(screen.getByText("Include Assistant Turns")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Include Assistant Turns" })).toBeChecked();
  });

  it("should render the assistant-turns switch off when it is not set", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);

    openAutoRouterAdvanced("Classification Method");

    expect(screen.getByRole("switch", { name: "Include Assistant Turns" })).not.toBeChecked();
  });

  it("should hide the assistant-turns switch when classifier_type is heuristic", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByText("Include Assistant Turns")).not.toBeInTheDocument();
  });

  it("should call onChange when the assistant-turns switch is toggled", async () => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);

    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("switch", { name: "Include Assistant Turns" }));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_context_include_assistant_turns: true }),
    );
  });

  it("should hide classifier context fields when classifier_type is heuristic", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByText("Context Window Size")).not.toBeInTheDocument();
    expect(screen.queryByText("Context Per-Turn Character Limit")).not.toBeInTheDocument();
  });

  it.each([
    ["Timeout (ms)", "7", { classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 7 } }],
    [
      "Circuit breaker cooldown (seconds)",
      "45",
      {
        classifier_llm_config: {
          model: "gpt-3.5-turbo",
          timeout_ms: 3000,
          circuit_breaker_cooldown_seconds: 45,
        },
      },
    ],
    ["Context Window Size", "0", { classifier_context_window_size: 0 }],
    ["Context Character Budget", "7", { classifier_context_budget_chars: 7 }],
  ])("keeps %s empty while it is being edited, then commits %s", (label, replacement, expected) => {
    const onChange = vi.fn();
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");

    const input = screen.getByLabelText(label);
    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue("");
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: replacement } });

    expect(onChange).toHaveBeenLastCalledWith({ ...llmValue, ...expected });
  });

  it("restores the committed context window size after an empty field loses focus", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");

    const input = screen.getByLabelText("Context Window Size");
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);

    expect(input).toHaveValue("3");
  });

  it("should render the custom technical keywords field", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("Custom Technical Keywords")).toBeInTheDocument();
  });

  it("should display existing custom technical keywords as tags", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        customTechnicalKeywords={["udp", "kafka"]}
        onCustomTechnicalKeywordsChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
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
    openAutoRouterAdvanced("Classification Method");
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
    openAutoRouterAdvanced("Classification Method");
    const keywordsSection = screen.getByText("Custom Technical Keywords").closest("div")?.parentElement as HTMLElement;
    await user.type(within(keywordsSection).getByRole("combobox"), "udp, kafka ,terraform");
    await user.click(await screen.findByText('Create "udp, kafka ,terraform"'));
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp", "kafka", "terraform"]);
  });

  it("should render an empty state when no keyword tier rules exist", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    expect(screen.getByText("Keyword Tier Overrides")).toBeInTheDocument();
    expect(screen.getByText("No keyword tier overrides configured")).toBeInTheDocument();
  });

  it("hides the keyword-tier and semantic sections when their change handlers are absent (edit modal)", async () => {
    // The edit-auto-router modal renders ComplexityRouterConfig without these handlers;
    // the sections must stay hidden rather than render interactive-but-dead controls.
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    expect(screen.queryByText("Keyword Tier Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Semantic keyword matching")).not.toBeInTheDocument();
    // Core tier config still renders.
    expect(screen.getByText("Models by tier")).toBeInTheDocument();
  });

  it("should call onKeywordTierRulesChange with a new rule when 'Add keyword rule' is clicked", async () => {
    const user = userEvent.setup();
    const onKeywordTierRulesChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onKeywordTierRulesChange={onKeywordTierRulesChange} />);
    openAutoRouterAdvanced("Keyword/Semantic Matching");
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
    openAutoRouterAdvanced("Keyword/Semantic Matching");

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
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    expect(screen.getByText("invoice")).toBeInTheDocument();
    expect(screen.getByText("refund")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove keyword rule 1/i }));
    expect(onKeywordTierRulesChange).toHaveBeenCalledWith([]);
  });

  it("should not show embedding model or match score fields when semantic matching is disabled", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={false} />);
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    expect(screen.getByText("Semantic keyword matching")).toBeInTheDocument();
    expect(screen.queryByText("Embedding model")).not.toBeInTheDocument();
    expect(screen.queryByText("Minimum match score")).not.toBeInTheDocument();
  });

  it("should show embedding model and match score fields when semantic matching is enabled", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} semanticMatchingEnabled={true} />);
    openAutoRouterAdvanced("Keyword/Semantic Matching");
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
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    await user.click(screen.getByRole("switch", { name: "Semantic keyword matching" }));
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

  it("does not show tier validation errors by default", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByText("This tier is required")).not.toBeInTheDocument();
  });

  it("shows an inline error on the classifier model select when llm is selected without a model", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={llmValue} showValidationErrors={true} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("A judge model is required")).toBeInTheDocument();
  });

  it("does not show the classifier model error once a classifier model is set", async () => {
    const llmValue: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={llmValue} showValidationErrors={true} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByText("A judge model is required")).not.toBeInTheDocument();
  });

  it("shows a validation error only under unfilled tiers when showValidationErrors is true", async () => {
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

  it("renders the escalation keywords section with current keywords when the handler is provided", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        escalationKeywords={["LITELLM ESCALATE"]}
        onEscalationKeywordsChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Escalation Keywords");
    expect(screen.getAllByText("Escalation Keywords")).not.toHaveLength(0);
    expect(screen.getByText("LITELLM ESCALATE")).toBeInTheDocument();
  });

  it("hides the escalation keywords section when no handler is provided", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByText("Escalation Keywords")).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig classifier fallback", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
  };

  it("defaults the fallback to the heuristic, matching the backend field default", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("radio", { name: /Score with the heuristic/ })).toBeChecked();
  });

  it("records a switch to the default model fallback", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("radio", { name: /Route to the default model/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ classifier_fallback: "default_model" }));
  });

  it("disables the default model fallback when no tier would produce one", async () => {
    // The deployment's default model is derived from the tiers on submit, so offering the option
    // with no tiers picked would save a config the backend rejects at startup.
    const noTiers: ComplexityRouterConfigValue = {
      ...llmValue,
      tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    };
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={noTiers} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("hides the fallback choice for the heuristic classifier, which has nothing to fall back from", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByText("If the classifier fails")).not.toBeInTheDocument();
  });

  it("stops describing the heuristic as the fallback once a custom prompt routes failures to the default model", async () => {
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
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText(/no longer runs at all/)).toBeInTheDocument();
  });

  it("still describes the heuristic as the fallback when a custom prompt keeps heuristic fallback", async () => {
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
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText(/only when the classifier call fails/)).toBeInTheDocument();
  });

  it("clears a stored fallback when switching back to the heuristic classifier", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{ ...llmValue, classifier_fallback: "default_model" }}
        onChange={onChange}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("radio", { name: /^Heuristics$/ }));
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ classifier_fallback: undefined }));
  });
});

describe("ComplexityRouterConfig classification frequency", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
  };

  it("defaults to every request, matching both backend field defaults", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).toHaveTextContent("Every request");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).not.toHaveTextContent(
      "Every new user message",
    );
    expect(screen.getByRole("combobox", { name: "How often to classify" })).not.toHaveTextContent("Once per session");
  });

  it("writes both wire fields when the frequency moves to every new user message", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("How often to classify", "Every new user message");
    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classification_mode: "user_turn",
      session_affinity: false,
    });
  });

  it("writes session affinity, not a classification mode, when the frequency moves to once per session", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={llmValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    await selectAutoRouterOption("How often to classify", "Once per session");
    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classification_mode: "every_request",
      session_affinity: true,
    });
  });

  it("shows a hand-authored config that sets both fields as once per session, matching the backend", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{ ...llmValue, classification_mode: "user_turn", session_affinity: true }}
        onChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).toHaveTextContent("Once per session");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).not.toHaveTextContent(
      "Every new user message",
    );
  });

  it("records a switch back to every request", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={{ ...llmValue, classification_mode: "user_turn" }}
        onChange={onChange}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).toHaveTextContent("Every new user message");
    await selectAutoRouterOption("How often to classify", "Every request");
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ classification_mode: "every_request" }));
  });

  it("offers the frequency on a heuristic router, where holding the tier still pins the model", async () => {
    // The backend pin is gated on the two fields alone, so a heuristic router that switches models
    // mid tool loop is fixed by this control too.
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("combobox", { name: "How often to classify" })).toBeVisible();
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
    openAutoRouterAdvanced("Classification Method");
    return onChange;
  };

  it("shows an existing router with no stored preset as legacy in the prompt control", async () => {
    // This router predates the setting. Displaying a calibrated preset it does not have would tell the
    // operator their traffic is graded by examples the classifier never receives, and saving the form
    // unchanged would then move its tier decisions.
    openClassificationPanel(llmValue);
    expect(screen.getByText("Legacy (uncalibrated) rubric")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customize prompt" })).toBeInTheDocument();
  });

  it("stamps the calibrated preset on a classifier being switched on for the first time", async () => {
    // A heuristic router turning on the LLM classifier has no prior tier behaviour to preserve, so a
    // newly configured classifier starts on the calibrated rubric rather than the legacy one.
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("radio", { name: /^LLM$/ }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_llm_config: expect.objectContaining({ classification_rubric: "agentic" }) }),
    );
  });

  it("shows the calibrated preset when a router stores one", async () => {
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, classification_rubric: "agentic" },
    });
    expect(screen.getByText("Agentic rubric")).toBeInTheDocument();
  });

  it("records the chat preset the operator picks inside the prompt editor", async () => {
    // The rubric now lives with the prompt it supplies, so picking one is an edit to the same control.
    const onChange = openClassificationPanel(llmValue);
    await userEvent.click(screen.getByRole("button", { name: "Customize prompt" }));
    await userEvent.click(await screen.findByRole("combobox", { name: "Base rubric" }));
    await userEvent.click(await screen.findByRole("option", { name: "Chat" }));
    // The pick is a draft until Save, so Cancel cannot leave a preset the operator only previewed.
    expect(onChange).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ classifier_llm_config: expect.objectContaining({ classification_rubric: "chat" }) }),
    );
  });

  it("describes the stored preset inside the editor when editing a router already on chat", async () => {
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, classification_rubric: "chat" },
    });
    await userEvent.click(screen.getByRole("button", { name: "Customize prompt" }));
    expect(await screen.findByText(/only conversational traffic/)).toBeInTheDocument();
  });

  it("records the business preset the operator picks inside the prompt editor", async () => {
    const onChange = openClassificationPanel(llmValue);
    await userEvent.click(screen.getByRole("button", { name: "Customize prompt" }));
    await userEvent.click(await screen.findByRole("combobox", { name: "Base rubric" }));
    await userEvent.click(await screen.findByRole("option", { name: "Business" }));
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier_llm_config: expect.objectContaining({ classification_rubric: "business" }),
      }),
    );
  });

  it("keeps the rubric out of the legacy whole-prompt editor, which replaces it entirely", async () => {
    // The backend rejects both together, so the legacy editor must not offer a rubric to pick.
    openClassificationPanel({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000, system_prompt: "Grade data sensitivity" },
    });
    expect(screen.getByRole("button", { name: "Edit custom prompt" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Customize prompt" })).not.toBeInTheDocument();
  });

  it("hides the prompt control for the heuristic classifier, which sends no prompt at all", async () => {
    openClassificationPanel(defaultValue);
    expect(screen.queryByRole("button", { name: "Customize prompt" })).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig tier labels", () => {
  const renamedValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    tier_labels: { SIMPLE: "Cheap", MEDIUM: "Standard", COMPLEX: "Premium", REASONING: "Deep" },
  };

  it("shows the operator's names in the tier headers instead of the defaults", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByText("Cheap Tier")).toBeInTheDocument();
    expect(screen.getByText("Deep Tier")).toBeInTheDocument();
    expect(screen.queryByText("Simple Tier")).not.toBeInTheDocument();
    expect(screen.queryByText("Reasoning Tier")).not.toBeInTheDocument();
  });

  it("keeps the rung ordinal and canonical name visible under a rename", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByText(/Tier 1 of 4/)).toHaveTextContent("Tier 1 of 4 · SIMPLE");
    expect(screen.getByText(/Tier 4 of 4/)).toHaveTextContent("Tier 4 of 4 · REASONING");
  });

  it("names the renamed tier in the required-field error", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...renamedValue, tiers: { ...defaultValue.tiers, REASONING: [] } }}
        showValidationErrors={true}
      />,
    );
    expect(screen.getByText("The Deep tier is required")).toBeInTheDocument();
  });

  it("reports a typed label back to the caller under its canonical tier key", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Display name for the Simple tier"), { target: { value: "Cheap" } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ tier_labels: { SIMPLE: "Cheap" } }));
  });

  it("shows a stored label in its input so an edit round-trips", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    expect(screen.getByLabelText("Display name for the Reasoning tier")).toHaveValue("Deep");
  });

  it("leaves the label inputs empty when nothing was renamed", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.getByLabelText("Display name for the Simple tier")).toHaveValue("");
  });

  it("uses the operator's names in the classification score table", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={renamedValue} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("Cheap")).toBeInTheDocument();
    expect(screen.getByText("Deep")).toBeInTheDocument();
  });

  it("uses the operator's names in the keyword rule tier picker", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={renamedValue}
        keywordTierRules={[{ id: "r1", keywords: ["invoice"], tier: "REASONING" }]}
      />,
    );
    openAutoRouterAdvanced("Keyword/Semantic Matching");
    expect(screen.getByRole("combobox", { name: "Route keyword rule 1 to tier" })).toHaveTextContent("Deep");
  });
});

describe("ComplexityRouterConfig modality panel", () => {
  it("defaults the image-routing switch off and writes modality_routing through onChange", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openAutoRouterAdvanced("Modality Routing");

    const toggle = screen.getByRole("switch", { name: "Route image requests to vision-capable models" });
    expect(toggle).not.toBeChecked();
    fireEvent.click(toggle);

    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, modality_routing: true });
  });

  it("renders a stored modality_routing=true as on", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={{ ...defaultValue, modality_routing: true }} />);
    openAutoRouterAdvanced("Modality Routing");

    expect(screen.getByRole("switch", { name: "Route image requests to vision-capable models" })).toBeChecked();
  });

  // The backend ignores modality_pin_override unless modality_routing is on, so offering it while
  // image routing is off would let an operator save a flag that does nothing.
  it("disables the pin-override switch while image routing is off", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openAutoRouterAdvanced("Modality Routing");

    const override = screen.getByRole("switch", { name: "Override session pin for image requests" });
    expect(override).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(override);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("writes modality_pin_override through onChange once image routing is on", async () => {
    const onChange = vi.fn();
    const value = { ...defaultValue, modality_routing: true };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={value} onChange={onChange} />);
    openAutoRouterAdvanced("Modality Routing");

    const override = screen.getByRole("switch", { name: "Override session pin for image requests" });
    expect(override).not.toBeChecked();
    fireEvent.click(override);

    expect(onChange).toHaveBeenCalledWith({ ...value, modality_pin_override: true });
  });

  it("renders a stored modality_pin_override=true as on", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, modality_routing: true, modality_pin_override: true }}
      />,
    );
    openAutoRouterAdvanced("Modality Routing");

    expect(screen.getByRole("switch", { name: "Override session pin for image requests" })).toBeChecked();
  });
});

describe("ComplexityRouterConfig affinity panel", () => {
  it("holds the deployment switch at its backend default, session pinning having moved to the frequency choice", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    openAutoRouterAdvanced("Affinity");

    expect(screen.getByRole("switch", { name: "Pin one model deployment per tier" })).toBeChecked();
    expect(screen.queryByRole("switch", { name: "Pin a session to its first model" })).not.toBeInTheDocument();
  });

  it("writes deployment_affinity through onChange without touching other keys", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openAutoRouterAdvanced("Affinity");

    fireEvent.click(screen.getByRole("switch", { name: "Pin one model deployment per tier" }));

    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, deployment_affinity: false });
  });

  it("renders a stored deployment_affinity=false as off", async () => {
    renderWithProviders(
      <ComplexityRouterConfig {...baseProps} value={{ ...defaultValue, deployment_affinity: false }} />,
    );
    openAutoRouterAdvanced("Affinity");

    expect(screen.getByRole("switch", { name: "Pin one model deployment per tier" })).not.toBeChecked();
  });

  it("writes an idle TTL on blur and keeps the partial input as a draft while typing", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openAutoRouterAdvanced("Affinity");

    const ttl = screen.getByLabelText("How long a pin survives idle (seconds)");
    expect(ttl).toHaveAttribute("placeholder", "3600");
    fireEvent.change(ttl, { target: { value: "300" } });
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.blur(ttl);

    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, session_affinity_ttl_seconds: 300 });
  });

  it("clearing the idle TTL returns the router to its backend default", async () => {
    const onChange = vi.fn();
    const value = { ...defaultValue, session_affinity_ttl_seconds: 300 };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={value} onChange={onChange} />);
    openAutoRouterAdvanced("Affinity");

    const ttl = screen.getByLabelText("How long a pin survives idle (seconds)");
    expect(ttl).toHaveValue("300");
    fireEvent.change(ttl, { target: { value: "" } });
    fireEvent.blur(ttl);

    expect(onChange).toHaveBeenCalledWith({ ...value, session_affinity_ttl_seconds: undefined });
  });

  it("clamps a non-positive idle TTL to the backend's minimum", async () => {
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onChange={onChange} />);
    openAutoRouterAdvanced("Affinity");

    const ttl = screen.getByLabelText("How long a pin survives idle (seconds)");
    fireEvent.change(ttl, { target: { value: "0" } });
    fireEvent.blur(ttl);

    expect(onChange).toHaveBeenCalledWith({ ...defaultValue, session_affinity_ttl_seconds: 1 });
  });
});

describe("ComplexityRouterConfig default model", () => {
  const getDefaultModelSelect = () => screen.getByRole("combobox", { name: "Default model" });

  it("shows what the tiers currently imply, so an untouched router still names its default", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(getDefaultModelSelect()).toHaveAttribute("placeholder", "Derived from tiers: gpt-3.5-turbo");
  });

  it("asks for a model rather than naming a derived one when no tier holds one", async () => {
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

  it("shows a pinned model as the selection instead of the tier-derived one", async () => {
    const pinned: ComplexityRouterConfigValue = { ...defaultValue, default_model: "claude-3-opus" };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinned} />);
    expect(getDefaultModelSelect()).toHaveValue("claude-3-opus");
  });

  it("unlocks the default model fallback on a pin alone, with no tier to derive from", async () => {
    const pinnedNoTiers: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
      default_model: "claude-3-opus",
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinnedNoTiers} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).not.toHaveAttribute("aria-disabled");
  });

  it("names the resolved default on the fallback option, so the destination is not a guess", async () => {
    const pinned: ComplexityRouterConfigValue = {
      ...defaultValue,
      classifier_type: "llm",
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
      default_model: "claude-3-opus",
    };
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={pinned} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("radio", { name: /Route to the default model \(claude-3-opus\)/ })).toBeInTheDocument();
  });
});

describe("plan-mode override", () => {
  const openPanel = () => openAutoRouterAdvanced("Plan-Mode Override");
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
  it("renders one effort select per selected model, defaulting to Default", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    const select = screen.getByRole("combobox", { name: "Reasoning effort for gpt-4 in the Complex tier" });
    expect(select).toHaveTextContent("Default");
  });

  it("shows the hydrated effort for a model that has one stored", async () => {
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

describe("ComplexityRouterConfig classifier reasoning effort", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-4", timeout_ms: 3000 },
  };

  const renderClassifier = (value: ComplexityRouterConfigValue = llmValue, onChange = vi.fn()) => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={value} onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    return onChange;
  };

  it("defaults to the classifier provider setting and offers only supported efforts", async () => {
    renderClassifier();
    const user = userEvent.setup();
    const select = screen.getByRole("combobox", { name: "Reasoning effort for classifier model gpt-4" });
    expect(select).toHaveTextContent("Default");
    await user.click(select);
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "Default",
      "medium",
      "high",
      "xhigh",
    ]);
  });

  it("stores an explicit effort on the classifier config", async () => {
    const onChange = renderClassifier();
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Reasoning effort for classifier model gpt-4" }));
    await user.click(await screen.findByRole("option", { name: "high" }));
    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classifier_llm_config: { model: "gpt-4", timeout_ms: 3000, reasoning_effort: "high" },
    });
  });

  it("removes the effort override when Default is selected", async () => {
    const onChange = renderClassifier({
      ...llmValue,
      classifier_llm_config: { model: "gpt-4", timeout_ms: 3000, reasoning_effort: "high" },
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Reasoning effort for classifier model gpt-4" }));
    await user.click(await screen.findByRole("option", { name: "Default" }));
    expect(onChange).toHaveBeenCalledWith(llmValue);
  });

  it("clears the old effort when the classifier model changes", async () => {
    const onChange = renderClassifier({
      ...llmValue,
      classifier_llm_config: { model: "gpt-4", timeout_ms: 3000, reasoning_effort: "high" },
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("combobox", { name: "Judge model" }));
    await user.click(await screen.findByRole("option", { name: "gpt-3.5-turbo" }));
    expect(onChange).toHaveBeenCalledWith({
      ...llmValue,
      classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
    });
  });

  it.each(["click", "enter"] as const)(
    "keeps the effort when the selected model is confirmed by %s",
    async (action) => {
      const onChange = renderClassifier({
        ...llmValue,
        classifier_llm_config: { model: "gpt-4", timeout_ms: 3000, reasoning_effort: "high" },
      });
      const user = userEvent.setup();
      await user.click(screen.getByRole("combobox", { name: "Judge model" }));
      if (action === "click") await user.click(await screen.findByRole("option", { name: "gpt-4" }));
      else await user.keyboard("{Enter}");
      expect(onChange).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["gpt-4", "max", "max (unsupported)", /not supported by every deployment/],
    ["claude-3-opus", "low", "low (unverified)", /cannot be verified/],
  ])("keeps a saved exceptional value visible for %s", (model, effort, label, warning) => {
    renderClassifier({
      ...llmValue,
      classifier_llm_config: { model, timeout_ms: 3000, reasoning_effort: effort },
    });
    expect(screen.getByRole("combobox", { name: `Reasoning effort for classifier model ${model}` })).toHaveTextContent(
      label,
    );
    expect(screen.getByText(warning)).toBeInTheDocument();
  });

  it.each(["claude-3-opus", "gpt-3.5-turbo"])("hides the effort control when %s has no advertised options", (model) => {
    renderClassifier({
      ...llmValue,
      classifier_llm_config: { model, timeout_ms: 3000 },
    });
    expect(
      screen.queryByRole("combobox", { name: `Reasoning effort for classifier model ${model}` }),
    ).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig reasoning effort gating", () => {
  it("offers no effort select for a model group without reasoning support", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(
      screen.queryByRole("combobox", { name: "Reasoning effort for gpt-3.5-turbo in the Simple tier" }),
    ).not.toBeInTheDocument();
  });

  // A stored effort on a model the group info calls non-reasoning must stay visible, or the
  // operator has no way to clear it.
  it("keeps the select for a non-reasoning model that already has a stored effort", async () => {
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
  it("offers no effort at all when the group intersects to nothing", async () => {
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
  it("keeps showing a stored effort outside the supported set", async () => {
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
    openAutoRouterAdvanced("Classification Method");
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

  it("hides the keywords when the scorer never runs, so they cannot imply an effect they have none", async () => {
    const llmWithDefaultFallback = {
      ...defaultValue,
      classifier_type: "llm" as const,
      classifier_llm_config: llmConfig,
      classifier_fallback: "default_model" as const,
    };
    openClassificationPanel(llmWithDefaultFallback);
    expect(screen.queryByText("Custom Technical Keywords")).not.toBeInTheDocument();
  });
});

describe("ComplexityRouterConfig tier editing", () => {
  const renderEditor = (
    value?: ComplexityRouterConfigValue,
    props: Partial<React.ComponentProps<typeof ComplexityRouterConfig>> = {},
  ) => {
    const onChange = vi.fn();
    const view = renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        {...(value ? { value } : {})}
        onChange={onChange}
        editingTiers
        onEditingTiersChange={vi.fn()}
        {...props}
      />,
    );
    return { ...view, committed: () => onChange.mock.calls[0][0] as ComplexityRouterConfigValue, onChange };
  };

  const customValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-4", timeout_ms: 3000 },
    custom_tier_set: {
      tiers: [
        { id: "CASUAL", name: "CASUAL", definition: "small talk", models: ["gpt-3.5-turbo"] },
        { id: "sec", name: "SECURITY_REVIEW", definition: "audits", models: ["gpt-4"] },
      ],
      fallback_tier_id: "CASUAL",
    },
  };

  it("offers Edit tiers only when the parent owns the editor flag", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} />);
    expect(screen.queryByRole("button", { name: "Edit tiers" })).not.toBeInTheDocument();
  });

  it("surfaces the caller's orphaned-rule verdict while editing, so Done is not a silent exit", async () => {
    renderEditor(customValue, { keywordRulesError: "Keyword rule(s) 1 route to a tier this router no longer has" });
    expect(
      screen.getByText("Keyword rule(s) 1 route to a tier this router no longer has", { exact: false }),
    ).toBeInTheDocument();
  });

  it("keeps the orphaned-rule verdict out of the collapsed view, where the submit tooltip owns it", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={customValue}
        onEditingTiersChange={vi.fn()}
        keywordRulesError="Keyword rule(s) 1 route to a tier this router no longer has"
      />,
    );
    expect(screen.queryByText("route to a tier this router no longer has", { exact: false })).not.toBeInTheDocument();
  });

  it("renders the four built-in tiers before any edit, unchanged", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onEditingTiersChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Edit tiers" })).toBeInTheDocument();
    expect(screen.getByText("Tier 1 of 4", { exact: false })).toHaveTextContent("SIMPLE");
  });

  it("adds a row and moves the form into an edited tier set, which the built-in record never leaves", async () => {
    const { committed } = renderEditor();
    fireEvent.click(screen.getByRole("button", { name: "Add tier" }));
    const next = committed();
    expect(next.custom_tier_set?.tiers).toHaveLength(5);
    expect(next.tiers).toEqual(defaultValue.tiers);
  });

  it("renames a built-in tier straight from the editor, which is what makes the set custom", async () => {
    const { committed } = renderEditor();
    fireEvent.change(screen.getByLabelText("Name for tier 3"), { target: { value: "SECURITY_REVIEW" } });
    const next = committed();
    expect(next.custom_tier_set?.tiers.map((row) => row.name)).toEqual([
      "SIMPLE",
      "MEDIUM",
      "SECURITY_REVIEW",
      "REASONING",
    ]);
    expect(next.tiers).toEqual(defaultValue.tiers);
  });

  it("opening the editor and changing nothing leaves the router on the built-in tiers", async () => {
    const { onChange } = renderEditor();
    expect(screen.getByRole("button", { name: "Done" })).toBeEnabled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("swaps the display-name field for the tier-name field while the editor is open", async () => {
    const { rerender } = renderWithProviders(<ComplexityRouterConfig {...baseProps} onEditingTiersChange={vi.fn()} />);
    expect(screen.getByLabelText("Display name for the Simple tier")).toBeInTheDocument();
    rerender(<ComplexityRouterConfig {...baseProps} editingTiers onEditingTiersChange={vi.fn()} />);
    expect(screen.queryByLabelText("Display name for the Simple tier")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Name for tier 1")).toBeInTheDocument();
  });

  it("drops the scorer card entirely once an edited tier set replaces the heuristic", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={customValue} onEditingTiersChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.queryByText("How Classification Works")).not.toBeInTheDocument();
    expect(
      screen.queryByText("scores each request across 7 built-in dimensions", { exact: false }),
    ).not.toBeInTheDocument();
  });

  it("keeps the scorer card on a built-in router, whose tiers the score still decides", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onEditingTiersChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
    expect(screen.getByText("scores each request across 7 built-in dimensions", { exact: false })).toBeInTheDocument();
  });

  it("says why a custom row is blocked instead of only reddening its border", async () => {
    const missingDefinition: ComplexityRouterConfigValue = {
      ...customValue,
      custom_tier_set: {
        tiers: [customValue.custom_tier_set!.tiers[0], { id: "b", name: "AUDIT", definition: "", models: ["gpt-4"] }],
        fallback_tier_id: "CASUAL",
      },
    };
    renderEditor(missingDefinition, { showValidationErrors: true });
    expect(screen.getByText("A definition is required", { exact: false })).toBeInTheDocument();
  });

  it("keeps Done disabled while a row is incomplete and says what is missing", async () => {
    const incomplete: ComplexityRouterConfigValue = {
      ...customValue,
      custom_tier_set: {
        tiers: [customValue.custom_tier_set!.tiers[0], { id: "new", name: "", definition: "", models: [] }],
        fallback_tier_id: "CASUAL",
      },
    };
    renderEditor(incomplete);
    expect(screen.getByRole("button", { name: "Done" })).toBeDisabled();
  });

  it("enables Done once every row carries a name, a definition and a model", async () => {
    renderEditor(customValue);
    expect(screen.getByRole("button", { name: "Done" })).toBeEnabled();
  });

  it("refuses to remove a row that would take the set below the backend's minimum", async () => {
    renderEditor(customValue);
    expect(screen.getByRole("button", { name: "Remove the CASUAL tier" })).toBeDisabled();
  });

  it("keeps a definition on one line, because the backend rejects a newline in it", async () => {
    const { committed } = renderEditor(customValue);
    fireEvent.change(screen.getByLabelText("Definition for tier 2"), { target: { value: "audits\nand reviews" } });
    const next = committed();
    expect(next.custom_tier_set?.tiers[1].definition).toBe("audits and reviews");
  });

  it("moves a keyword rule with the tier it points at when that tier is renamed", async () => {
    const onKeywordTierRulesChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={customValue}
        keywordTierRules={[{ id: "r1", keywords: ["audit"], tier: "SECURITY_REVIEW" }]}
        onKeywordTierRulesChange={onKeywordTierRulesChange}
        editingTiers
        onEditingTiersChange={vi.fn()}
      />,
    );
    fireEvent.change(screen.getByLabelText("Name for tier 2"), { target: { value: "AUDIT" } });
    expect(onKeywordTierRulesChange).toHaveBeenCalledWith([{ id: "r1", keywords: ["audit"], tier: "AUDIT" }]);
  });

  it("re-points the fallback tier when the row it named is removed, never leaving it dangling", async () => {
    const threeRows: ComplexityRouterConfigValue = {
      ...customValue,
      custom_tier_set: {
        tiers: [
          ...customValue.custom_tier_set!.tiers,
          { id: "third", name: "MEDIUM", definition: "", models: ["gpt-4"] },
        ],
        fallback_tier_id: "sec",
      },
    };
    const { committed } = renderEditor(threeRows);
    fireEvent.click(screen.getByRole("button", { name: "Remove the SECURITY_REVIEW tier" }));
    const next = committed();
    expect(next.custom_tier_set?.tiers.some((row) => row.id === next.custom_tier_set?.fallback_tier_id)).toBe(true);
  });

  it("turns off a plan-mode floor whose row was removed, rather than leaving it pointing at nothing", async () => {
    const withFloor: ComplexityRouterConfigValue = {
      ...customValue,
      plan_mode_min_tier: "sec",
      custom_tier_set: {
        tiers: [
          ...customValue.custom_tier_set!.tiers,
          { id: "third", name: "BULK", definition: "d", models: ["gpt-4"] },
        ],
        fallback_tier_id: "CASUAL",
      },
    };
    const { committed } = renderEditor(withFloor);
    fireEvent.click(screen.getByRole("button", { name: "Remove the SECURITY_REVIEW tier" }));
    expect(committed().plan_mode_min_tier).toBeUndefined();
  });

  it("replaces the display-name inputs with the reason an edited tier set forbids them", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={customValue} onEditingTiersChange={vi.fn()} />);
    expect(screen.queryByLabelText("Display name for the Simple tier")).not.toBeInTheDocument();
    expect(screen.getByText("Display names rename the built-in tiers", { exact: false })).toBeInTheDocument();
    expect(screen.getByLabelText("Fallback tier")).toBeInTheDocument();
  });

  it("disables the once-per-session frequency and says why, rather than letting a stripped value look saved", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...customValue, session_affinity: true }}
        onEditingTiersChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("combobox", { name: "How often to classify" }));
    const sessionOption = screen.getByRole("option", { name: "Once per session" });
    expect(sessionOption).toHaveAttribute("aria-disabled", "true");
    expect(sessionOption).not.toBeChecked();
    expect(
      screen.getByText("Session pinning escalates along the built-in tier ladder", { exact: false }),
    ).toBeInTheDocument();
  });

  it("lets an edited tier set write its own opening instructions instead of refusing a prompt outright", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} value={customValue} onEditingTiersChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("your own calibration examples", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customize prompt" })).toBeInTheDocument();
    expect(screen.queryByText("A replacement prompt drops the tier bullets", { exact: false })).not.toBeInTheDocument();
  });

  it("gives built-in routers the opening-only editor, keeping the tier definitions derived", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{ ...defaultValue, classifier_type: "llm", classifier_llm_config: { model: "gpt-4", timeout_ms: 3000 } }}
        onEditingTiersChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByText("The base rubric supplies", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customize prompt" })).toBeInTheDocument();
    expect(screen.queryByText("Replace the built-in complexity rubric", { exact: false })).not.toBeInTheDocument();
  });

  it("keeps the legacy whole-prompt editor only on a router that already stored a replacement prompt", async () => {
    renderWithProviders(
      <ComplexityRouterConfig
        {...baseProps}
        value={{
          ...defaultValue,
          classifier_type: "llm",
          classifier_llm_config: { model: "gpt-4", timeout_ms: 3000, system_prompt: "Grade data sensitivity" },
        }}
        onEditingTiersChange={vi.fn()}
      />,
    );
    openAutoRouterAdvanced("Classification Method");
    expect(screen.getByRole("button", { name: "Edit custom prompt" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Customize prompt" })).not.toBeInTheDocument();
  });

  it("leaves built-in routers with their display-name inputs and no restriction copy", async () => {
    renderWithProviders(<ComplexityRouterConfig {...baseProps} onEditingTiersChange={vi.fn()} />);
    expect(screen.getByLabelText("Display name for the Simple tier")).toBeInTheDocument();
    expect(screen.queryByText("Display names rename the built-in tiers", { exact: false })).not.toBeInTheDocument();
  });
});

describe("classifier vision settings", () => {
  const llmValue: ComplexityRouterConfigValue = {
    ...defaultValue,
    classifier_type: "llm",
    classifier_llm_config: { model: "gpt-3.5-turbo", timeout_ms: 3000 },
  };

  const VisionFixture = ({ onChange = vi.fn() }: { onChange?: Mock }) => {
    const [value, setValue] = React.useState(llmValue);
    return (
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={value}
        onChange={(nextValue) => {
          setValue(nextValue);
          onChange(nextValue);
        }}
      />
    );
  };

  it("starts off and reveals the default cap when enabled", async () => {
    renderWithProviders(<VisionFixture />);
    openAutoRouterAdvanced("Classification Method");

    const vision = screen.getByRole("switch", { name: "Use images for classification" });
    expect(vision).not.toBeChecked();
    expect(screen.queryByLabelText("Maximum images per request")).not.toBeInTheDocument();

    fireEvent.click(vision);

    expect(screen.getByLabelText("Maximum images per request")).toHaveValue("1");
  });

  it("writes the switch and a clamped image cap into the classifier config", async () => {
    const onChange = vi.fn();
    renderWithProviders(<VisionFixture onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");

    fireEvent.click(screen.getByRole("switch", { name: "Use images for classification" }));
    expect(onChange).toHaveBeenLastCalledWith({
      ...llmValue,
      classifier_llm_config: { ...llmValue.classifier_llm_config, vision: { enabled: true, max_images: 1 } },
    });

    fireEvent.change(screen.getByLabelText("Maximum images per request"), { target: { value: "1.7" } });
    expect(onChange).toHaveBeenLastCalledWith({
      ...llmValue,
      classifier_llm_config: { ...llmValue.classifier_llm_config, vision: { enabled: true, max_images: 2 } },
    });
  });

  it("keeps the image cap draft empty until a valid value is entered", async () => {
    const onChange = vi.fn();
    renderWithProviders(<VisionFixture onChange={onChange} />);
    openAutoRouterAdvanced("Classification Method");
    fireEvent.click(screen.getByRole("switch", { name: "Use images for classification" }));
    onChange.mockClear();

    const input = screen.getByLabelText("Maximum images per request");
    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue("");
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: "0" } });
    expect(onChange).toHaveBeenLastCalledWith({
      ...llmValue,
      classifier_llm_config: { ...llmValue.classifier_llm_config, vision: { enabled: true, max_images: 1 } },
    });
  });

  it("is absent when the classifier is heuristic", async () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultValue} onChange={vi.fn()} />);
    openAutoRouterAdvanced("Classification Method");

    expect(screen.queryByText("Use images for classification")).not.toBeInTheDocument();
  });
});
