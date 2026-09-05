import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useComplexityScorerDefaults } from "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import HeuristicScoringConfig from "./HeuristicScoringConfig";
import { ClassifierFallback, ClassifierType, ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { DIMENSION_LABELS } from "./heuristic_scoring_knobs";
import { LOADED_SCORER_DEFAULTS_QUERY, SHIPPED_SCORER_DEFAULTS } from "../../../tests/mocks/complexityScorerDefaults";

vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

const BASE: ComplexityRouterConfigValue = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: ["o3"], REASONING: ["o3"] },
  classifier_type: "heuristic",
};

const render = async (value: ComplexityRouterConfigValue, onChange = vi.fn()) => {
  renderWithProviders(<HeuristicScoringConfig value={value} onChange={onChange} />);
  await userEvent.click(screen.getByText("Advanced scoring"));
  return onChange;
};

const commit = async (label: string, raw: string) => {
  const onChange = await render(BASE);
  fireEvent.change(screen.getByLabelText(label), { target: { value: raw } });
  return onChange.mock.calls.at(-1)?.[0] as ComplexityRouterConfigValue | undefined;
};

describe("HeuristicScoringConfig", () => {
  it("counts overridden groups on the collapsed header", () => {
    const tuned = { ...BASE, token_thresholds: { simple: 25, complex: 900 } };
    renderWithProviders(<HeuristicScoringConfig value={tuned} onChange={vi.fn()} />);

    expect(screen.getByTestId("advanced-scoring-override-count")).toHaveTextContent("1 override");
  });

  it("prefills the shipped defaults", async () => {
    await render(BASE);

    expect(screen.getByLabelText("Simple to Medium")).toHaveValue("0.15");
    expect(screen.getByLabelText("Long above")).toHaveValue("400");
    expect(screen.getByTestId("dimension-weight-total")).toHaveTextContent("total 1.00");
  });

  it("writes a whole dict, and keeps the decimal point typeable", async () => {
    // A plain controlled number input renders Number("0.") as "0", so "0.22" would be untypeable.
    const onChange = await render(BASE);
    const field = screen.getByLabelText("Simple to Medium");

    fireEvent.change(field, { target: { value: "0." } });
    expect(field).toHaveValue("0.");
    fireEvent.change(field, { target: { value: "0.22" } });

    expect((onChange.mock.calls.at(-1)?.[0] as ComplexityRouterConfigValue).tier_boundaries).toEqual({
      simple_medium: 0.22,
      medium_complex: 0.35,
      complex_reasoning: 0.6,
    });
  });

  it("commits nothing for an emptied field, rather than NaN", async () => {
    const onChange = await render(BASE);
    fireEvent.change(screen.getByLabelText("Short below"), { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalled();
  });

  // min and max are inert attributes on a text input, so without the explicit clamp these would persist
  // a weight of 999, or an infinite boundary, into the router config.
  it.each([
    ["Code presence", "999", 1],
    ["Code presence", "-2", 0],
    ["Long above", "100000", 100000],
  ])("clamps %s = %s to %s", async (label, raw, expected) => {
    const next = await commit(label, raw);
    expect(
      { ...next?.dimension_weights, ...next?.token_thresholds }[label === "Long above" ? "complex" : "codePresence"],
    ).toBe(expected);
  });

  it.each(["Infinity", "1e999"])("refuses to commit %s", async (raw) => {
    expect((await commit("Simple to Medium", raw))?.tier_boundaries).toBeUndefined();
  });

  it("resets a group back to undefined so it tracks the backend defaults again", async () => {
    const tuned = { ...BASE, token_thresholds: { simple: 25, complex: 900 } };
    const onChange = await render(tuned);

    await userEvent.click(screen.getByRole("button", { name: "Reset to defaults" }));

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ token_thresholds: undefined }));
  });

  it("shows the boundary an untouched override floor tracks, rather than a fixed number", async () => {
    await render(BASE);

    const field = screen.getByLabelText("Minimum score");
    expect(field).toHaveValue("");
    expect(field).toHaveAttribute("placeholder", SHIPPED_SCORER_DEFAULTS.tier_boundaries.simple_medium.toFixed(2));
  });

  it("tracks the operator's own Simple to Medium override, not the shipped boundary", async () => {
    await render({ ...BASE, tier_boundaries: { simple_medium: 0.42, medium_complex: 0.5, complex_reasoning: 0.7 } });

    expect(screen.getByLabelText("Minimum score")).toHaveAttribute("placeholder", "0.42");
  });

  // 0 restores an unconditional override, so it has to reach the config as 0 rather than as "untouched".
  it("commits an explicit 0 override floor", async () => {
    const onChange = await render(BASE);
    fireEvent.change(screen.getByLabelText("Minimum score"), { target: { value: "0" } });

    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({ reasoning_override_min_score: 0 });
  });

  it("renders a stored 0 as 0 rather than as an untouched field", async () => {
    await render({ ...BASE, reasoning_override_min_score: 0 });

    expect(screen.getByLabelText("Minimum score")).toHaveValue("0");
  });

  it("counts a set override floor among the overrides", () => {
    renderWithProviders(
      <HeuristicScoringConfig value={{ ...BASE, reasoning_override_min_score: 0 }} onChange={vi.fn()} />,
    );

    expect(screen.getByTestId("advanced-scoring-override-count")).toHaveTextContent("1 override");
  });

  it("clamps the override floor to the score range", async () => {
    const onChange = await render(BASE);
    fireEvent.change(screen.getByLabelText("Minimum score"), { target: { value: "9" } });

    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({ reasoning_override_min_score: 1 });
  });

  it("resets the override floor back to tracking the boundary", async () => {
    const onChange = await render({ ...BASE, reasoning_override_min_score: 0 });

    await userEvent.click(screen.getAllByRole("button", { name: "Reset to defaults" }).at(-1)!);

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ reasoning_override_min_score: undefined }));
  });

  it("flags decreasing boundaries as an error without blocking the save", async () => {
    const bad = { ...BASE, tier_boundaries: { simple_medium: 0.5, medium_complex: 0.2, complex_reasoning: 0.6 } };
    await render(bad);

    expect(screen.getByRole("alert")).toHaveTextContent(/unreachable/);
  });
});

describe("ClassificationMethodConfig scorer gating", () => {
  const props = { onChange: vi.fn(), modelOptions: [{ value: "gpt-4o-mini", label: "gpt-4o-mini" }] };
  const withClassifier = (type: ClassifierType, fallback?: ClassifierFallback): ComplexityRouterConfigValue => ({
    ...BASE,
    classifier_type: type,
    classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifier_fallback: fallback,
  });

  it.each([
    ["heuristic decides the tier", "heuristic" as ClassifierType, undefined, true],
    ["an LLM classifier falls back to the heuristic", "llm" as ClassifierType, "heuristic" as ClassifierFallback, true],
    [
      "an LLM classifier falls back to the default model",
      "llm" as ClassifierType,
      "default_model" as ClassifierFallback,
      false,
    ],
  ])("offers the knobs when %s: %s", async (_case, type, fallback, expected) => {
    renderWithProviders(<ClassificationMethodConfig {...props} value={withClassifier(type, fallback)} />);

    expect(screen.queryByText("Advanced scoring") !== null).toBe(expected);
  });

  it("describes the tier ranges from the configured boundaries, not the shipped numbers", () => {
    const tuned = { ...BASE, tier_boundaries: { simple_medium: 0.22, medium_complex: 0.44, complex_reasoning: 0.66 } };
    renderWithProviders(<ClassificationMethodConfig {...props} value={tuned} />);

    expect(screen.getByText(/Score < 0.22/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.44 - 0.66/)).toBeInTheDocument();
    expect(screen.queryByText(/0.15/)).not.toBeInTheDocument();
  });

  it("states the configured override floor in the reasoning-marker aside, not the boundary", () => {
    renderWithProviders(<ClassificationMethodConfig {...props} value={{ ...BASE, reasoning_override_min_score: 0 }} />);

    expect(screen.getByText(/2\+ reasoning markers with a score of at least 0\.00/)).toBeInTheDocument();
  });

  it("falls back to the Simple to Medium boundary when no override floor is set", () => {
    renderWithProviders(<ClassificationMethodConfig {...props} value={BASE} />);

    expect(screen.getByText(/2\+ reasoning markers with a score of at least 0\.15/)).toBeInTheDocument();
  });

  it("renders a row for every scored dimension", async () => {
    await render(BASE);

    for (const key of Object.keys(SHIPPED_SCORER_DEFAULTS.dimension_weights)) {
      expect(screen.getByLabelText(DIMENSION_LABELS[key])).toBeInTheDocument();
    }
  });
});

describe("HeuristicScoringConfig when the defaults request fails", () => {
  const failing = { data: undefined, isPending: false, isError: true, refetch: vi.fn() };
  const pending = { data: undefined, isPending: true, isError: false, refetch: vi.fn() };

  const renderWithQuery = async (query: unknown, value: ComplexityRouterConfigValue) => {
    vi.mocked(useComplexityScorerDefaults).mockReturnValue(query as never);
    renderWithProviders(<HeuristicScoringConfig value={value} onChange={vi.fn()} />);
    await userEvent.click(screen.getByText("Advanced scoring"));
  };

  afterEach(() => vi.mocked(useComplexityScorerDefaults).mockReturnValue(LOADED_SCORER_DEFAULTS_QUERY));

  it("says so instead of claiming to still be loading", async () => {
    await renderWithQuery(failing, BASE);

    expect(screen.queryByText(/Loading the shipped defaults/)).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/Could not load the shipped defaults/);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("still shows and edits the values this router already overrides", async () => {
    await renderWithQuery(failing, { ...BASE, token_thresholds: { simple: 25, complex: 900 } });

    expect(screen.getByLabelText("Short below")).toHaveValue("25");
    expect(screen.getByLabelText("Long above")).toHaveValue("900");
  });

  it("keeps saying loading while the request is genuinely in flight", async () => {
    await renderWithQuery(pending, BASE);

    expect(screen.getByText(/Loading the shipped defaults/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("HeuristicScoringConfig degraded states", () => {
  afterEach(() => vi.mocked(useComplexityScorerDefaults).mockReturnValue(LOADED_SCORER_DEFAULTS_QUERY));

  it("states no weight total when the dimension set is unknown, rather than one built from overrides alone", async () => {
    vi.mocked(useComplexityScorerDefaults).mockReturnValue({
      data: undefined,
      isPending: false,
      isError: true,
      refetch: vi.fn(),
    } as never);
    renderWithProviders(
      <HeuristicScoringConfig value={{ ...BASE, dimension_weights: { codePresence: 0.5 } }} onChange={vi.fn()} />,
    );
    await userEvent.click(screen.getByText("Advanced scoring"));

    expect(screen.getByLabelText("Code presence")).toHaveValue("0.5");
    expect(screen.queryByTestId("dimension-weight-total")).not.toBeInTheDocument();
  });
});
