import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useClassifierPlugins } from "@/app/(dashboard)/hooks/autoRouter/useClassifierPlugins";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { LOADED_CLASSIFIER_PLUGINS_QUERY } from "../../../tests/mocks/classifierPlugins";

vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useComplexityScorerDefaults",
  async () => await import("../../../tests/mocks/complexityScorerDefaults"),
);

vi.mock(
  "@/app/(dashboard)/hooks/autoRouter/useClassifierPlugins",
  async () => await import("../../../tests/mocks/classifierPlugins"),
);

const BASE: ComplexityRouterConfigValue = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: ["o3"], REASONING: ["o3"] },
  classifier_type: "heuristic",
};

const CUSTOM: ComplexityRouterConfigValue = {
  ...BASE,
  classifier_type: "custom",
  classifier_plugin: "tier-by-team",
  classifier_plugin_timeout_ms: 3000,
};

const MODEL_OPTIONS = [{ value: "gpt-4o-mini", label: "gpt-4o-mini" }];

const render = (value: ComplexityRouterConfigValue, onChange = vi.fn()) => {
  renderWithProviders(
    <ClassificationMethodConfig value={value} onChange={onChange} modelOptions={MODEL_OPTIONS} defaultModel="gpt-4o" />,
  );
  return onChange;
};

const lastValue = (onChange: ReturnType<typeof vi.fn>) =>
  onChange.mock.calls.at(-1)?.[0] as ComplexityRouterConfigValue | undefined;

describe("ClassificationMethodConfig classification method radios", () => {
  it("offers the custom classifier alongside the heuristic and the LLM classifier", () => {
    render(BASE);

    expect(screen.getByRole("radio", { name: /Heuristic/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /LLM Classifier/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Custom classifier/ })).toBeEnabled();
  });

  // Only the custom branch may carry a plugin: leaving one behind would have the backend reject the
  // save outright (classifier_plugin set with a non-custom classifier_type).
  it("clears the plugin when the operator switches back to the heuristic", async () => {
    const onChange = render(CUSTOM);

    await userEvent.click(screen.getByRole("radio", { name: /Heuristic/ }));

    expect(lastValue(onChange)).toMatchObject({
      classifier_type: "heuristic",
      classifier_plugin: undefined,
      classifier_plugin_timeout_ms: undefined,
      classifier_fallback: undefined,
    });
  });

  it("stamps the default plugin timeout when custom is selected", async () => {
    const onChange = render(BASE);

    await userEvent.click(screen.getByRole("radio", { name: /Custom classifier/ }));

    expect(lastValue(onChange)).toMatchObject({ classifier_type: "custom", classifier_plugin_timeout_ms: 3000 });
  });
});

describe("ClassificationMethodConfig custom classifier controls", () => {
  it("does not show the plugin controls until custom is the selected method", () => {
    render(BASE);

    expect(screen.queryByRole("combobox", { name: "Classifier Plugin" })).not.toBeInTheDocument();
  });

  it("offers the names the proxy registered", async () => {
    render({ ...CUSTOM, classifier_plugin: undefined });

    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Classifier Plugin" }));

    expect(await screen.findByTitle("tier-by-team")).toBeInTheDocument();
    expect(screen.getByTitle("spend-aware")).toBeInTheDocument();
  });

  it("records the plugin the operator picks", async () => {
    const onChange = render({ ...CUSTOM, classifier_plugin: undefined });

    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Classifier Plugin" }));
    await userEvent.click(await screen.findByTitle("spend-aware"));

    expect(lastValue(onChange)).toMatchObject({ classifier_plugin: "spend-aware", classifier_plugin_timeout_ms: 3000 });
  });

  it("emits an edited plugin timeout", () => {
    const onChange = render(CUSTOM);

    fireEvent.change(screen.getByRole("spinbutton", { name: "Plugin Timeout (ms)" }), { target: { value: "750" } });

    expect(lastValue(onChange)).toMatchObject({ classifier_plugin_timeout_ms: 750 });
  });

  // The backend applies classifier_fallback to a plugin exactly as it does to an LLM classifier, so
  // gating this picker on the LLM branch would leave the plugin's failure path unconfigurable.
  it("offers the fallback picker, which is not LLM-only", () => {
    render(CUSTOM);

    expect(screen.getByRole("radio", { name: /Score with the heuristic/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Route to the default model/ })).toBeInTheDocument();
  });

  it("says the score no longer decides the tier under a plugin", () => {
    render(CUSTOM);

    expect(screen.getByText(/classifies with your own classifier plugin/)).toBeInTheDocument();
  });

  it("reports a missing plugin once the form has been submitted", () => {
    renderWithProviders(
      <ClassificationMethodConfig
        value={{ ...CUSTOM, classifier_plugin: undefined }}
        onChange={vi.fn()}
        modelOptions={MODEL_OPTIONS}
        showValidationErrors
      />,
    );

    expect(screen.getByText("A classifier plugin is required")).toBeInTheDocument();
  });
});

describe("ClassificationMethodConfig when the proxy registered no plugins", () => {
  const empty = { data: [], isPending: false, isError: false, refetch: vi.fn() };
  const failing = { data: undefined, isPending: false, isError: true, refetch: vi.fn() };

  afterEach(() => vi.mocked(useClassifierPlugins).mockReturnValue(LOADED_CLASSIFIER_PLUGINS_QUERY));

  it("disables the custom radio and says how to enable it", () => {
    vi.mocked(useClassifierPlugins).mockReturnValue(empty as never);
    render(BASE);

    expect(screen.getByRole("radio", { name: /Custom classifier/ })).toBeDisabled();
    expect(
      screen.getByText("Declare classifier_plugins in the proxy config to enable custom classifiers"),
    ).toBeInTheDocument();
  });

  // A fetch that never landed leaves the registry unknown, not empty. Blanking the select or greying
  // out the radio here would clear an already-configured plugin on the operator's next save.
  it("keeps a stored plugin name selectable when the fetch failed", () => {
    vi.mocked(useClassifierPlugins).mockReturnValue(failing as never);
    render(CUSTOM);

    expect(screen.getByRole("radio", { name: /Custom classifier/ })).toBeEnabled();
    expect(screen.getByRole("combobox", { name: "Classifier Plugin" })).toBeInTheDocument();
    expect(screen.getByTitle("tier-by-team")).toBeInTheDocument();
    expect(
      screen.queryByText("Declare classifier_plugins in the proxy config to enable custom classifiers"),
    ).not.toBeInTheDocument();
  });
});
