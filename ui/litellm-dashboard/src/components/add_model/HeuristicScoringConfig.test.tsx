import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import HeuristicScoringConfig from "./HeuristicScoringConfig";
import { ComplexityRouterConfigValue, DEFAULT_DIMENSION_WEIGHTS } from "./ComplexityRouterConfig";

const BASE: ComplexityRouterConfigValue = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: ["o3"], REASONING: ["o3"] },
  classifier_type: "heuristic",
};

const expandPanel = async () => {
  await userEvent.click(screen.getByText("Advanced scoring"));
};

describe("HeuristicScoringConfig", () => {
  it("starts collapsed with no override badge on an untouched router", () => {
    renderWithProviders(<HeuristicScoringConfig value={BASE} onChange={vi.fn()} />);

    expect(screen.getByText("Advanced scoring")).toBeInTheDocument();
    expect(screen.queryByTestId("advanced-scoring-override-count")).not.toBeInTheDocument();
  });

  it("counts the overridden groups in the collapsed header, so a tuned router is visible without expanding", () => {
    renderWithProviders(
      <HeuristicScoringConfig
        value={{
          ...BASE,
          token_thresholds: { simple: 25, complex: 900 },
          dimension_weights: DEFAULT_DIMENSION_WEIGHTS,
        }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("advanced-scoring-override-count")).toHaveTextContent("2 overrides");
  });

  it("shows the shipped defaults as the starting values", async () => {
    renderWithProviders(<HeuristicScoringConfig value={BASE} onChange={vi.fn()} />);
    await expandPanel();

    expect(screen.getByLabelText("Simple to Medium")).toHaveValue("0.15");
    expect(screen.getByLabelText("Long above")).toHaveValue("400");
    expect(screen.getByTestId("dimension-weight-total")).toHaveTextContent("total 1.00");
  });

  it("writes a whole boundary dict when one field is edited, never a partial one", async () => {
    const onChange = vi.fn();
    renderWithProviders(<HeuristicScoringConfig value={BASE} onChange={onChange} />);
    await expandPanel();

    fireEvent.change(screen.getByLabelText("Complex to Reasoning"), { target: { value: "0.8" } });

    const last = onChange.mock.calls.at(-1)?.[0] as ComplexityRouterConfigValue;
    expect(last.tier_boundaries).toEqual({ simple_medium: 0.15, medium_complex: 0.35, complex_reasoning: 0.8 });
  });

  it("commits nothing when a field is emptied, rather than writing NaN into the config", async () => {
    const onChange = vi.fn();
    renderWithProviders(<HeuristicScoringConfig value={BASE} onChange={onChange} />);
    await expandPanel();

    fireEvent.change(screen.getByLabelText("Short below"), { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalled();
  });

  it("lets a decimal be typed a character at a time without eating the point", async () => {
    // A plain controlled number input renders Number("0.") as "0", so "0.22" is untypeable. The field
    // shows the raw draft while it is being edited, and commits only what parses.
    const onChange = vi.fn();
    renderWithProviders(<HeuristicScoringConfig value={BASE} onChange={onChange} />);
    await expandPanel();

    const field = screen.getByLabelText("Simple to Medium");
    fireEvent.change(field, { target: { value: "0." } });
    expect(field).toHaveValue("0.");

    fireEvent.change(field, { target: { value: "0.22" } });
    const last = onChange.mock.calls.at(-1)?.[0] as ComplexityRouterConfigValue;
    expect(last.tier_boundaries?.simple_medium).toBe(0.22);
  });

  it("resets a group back to undefined so the router goes back to tracking the backend defaults", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <HeuristicScoringConfig
        value={{ ...BASE, token_thresholds: { simple: 25, complex: 900 } }}
        onChange={onChange}
      />,
    );
    await expandPanel();

    const [resetTokenThresholds] = screen.getAllByRole("button", { name: "Reset to defaults" });
    await userEvent.click(resetTokenThresholds);

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ token_thresholds: undefined }));
  });

  it("warns when boundaries are out of order, which strands the tiers between them", async () => {
    renderWithProviders(
      <HeuristicScoringConfig
        value={{ ...BASE, tier_boundaries: { simple_medium: 0.5, medium_complex: 0.2, complex_reasoning: 0.6 } }}
        onChange={vi.fn()}
      />,
    );
    await expandPanel();

    expect(screen.getByRole("alert")).toHaveTextContent(/every tier between them is unreachable/);
  });

  it("explains a weight total away from 1.00 instead of blocking it", async () => {
    renderWithProviders(
      <HeuristicScoringConfig
        value={{ ...BASE, dimension_weights: { ...DEFAULT_DIMENSION_WEIGHTS, codePresence: 0.5 } }}
        onChange={vi.fn()}
      />,
    );
    await expandPanel();

    expect(screen.getByTestId("dimension-weight-total")).toHaveTextContent("total 1.20");
    expect(screen.getByText(/absolute multipliers/)).toBeInTheDocument();
  });
});

describe("ClassificationMethodConfig scorer knob gating", () => {
  const classificationProps = {
    onChange: vi.fn(),
    modelOptions: [{ value: "gpt-4o-mini", label: "gpt-4o-mini" }],
    defaultModel: "gpt-4o-mini",
  };

  it("offers the knobs when the heuristic decides the tier", () => {
    renderWithProviders(<ClassificationMethodConfig {...classificationProps} value={BASE} />);
    expect(screen.getByText("Advanced scoring")).toBeInTheDocument();
  });

  it("offers the knobs to an LLM classifier that falls back to the heuristic", () => {
    renderWithProviders(
      <ClassificationMethodConfig
        {...classificationProps}
        value={{
          ...BASE,
          classifier_type: "llm",
          classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
          classifier_fallback: "heuristic",
        }}
      />,
    );
    expect(screen.getByText("Advanced scoring")).toBeInTheDocument();
  });

  it("hides them when the classifier falls back to the default model and nothing is ever scored", () => {
    renderWithProviders(
      <ClassificationMethodConfig
        {...classificationProps}
        value={{
          ...BASE,
          classifier_type: "llm",
          classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
          classifier_fallback: "default_model",
        }}
      />,
    );
    expect(screen.queryByText("Advanced scoring")).not.toBeInTheDocument();
  });

  it("describes the tier ranges from the configured boundaries rather than the shipped numbers", () => {
    renderWithProviders(
      <ClassificationMethodConfig
        {...classificationProps}
        value={{ ...BASE, tier_boundaries: { simple_medium: 0.22, medium_complex: 0.44, complex_reasoning: 0.66 } }}
      />,
    );

    expect(screen.getByText(/Score < 0.22/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.22 - 0.44/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.44 - 0.66/)).toBeInTheDocument();
    expect(screen.getByText(/Score > 0.66/)).toBeInTheDocument();
    expect(screen.queryByText(/0.15/)).not.toBeInTheDocument();
  });
});
