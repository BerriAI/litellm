import { selectAutoRouterApproach } from "../../../tests/autoRouterSetup";
import React, { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { act, fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import ForecastClassifierConfig from "./ForecastClassifierConfig";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import { getForecastConfigError, isForecastClassifier } from "./forecast_classifier_config";
import { buildUpdatedComplexityRouterConfig } from "../edit_auto_router/edit_auto_router_modal";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  getComplexityScorerDefaults: vi.fn(async () => ({
    tier_boundaries: {},
    token_thresholds: {},
    dimension_weights: {},
  })),
}));

const initial: ComplexityRouterConfigValue = {
  classifier_type: "capability",
  classifier_llm_config: { model: "judge", timeout_ms: 20000 },
  tiers: { SIMPLE: ["efficient"], MEDIUM: [], COMPLEX: [], REASONING: ["capable"] },
  capability_classifier_config: { efficient_tier: "SIMPLE", capable_tier: "REASONING", base_threshold: 0.7 },
};
const fuseInitial: ComplexityRouterConfigValue = {
  ...initial,
  classifier_type: "llm_v2",
  capability_classifier_config: undefined,
  adaptive: false,
  llm_v2_config: {
    efficient_profile: "Small solver",
    capable_profile: "Larger solver",
    harness: "One attempt",
    max_quality_gap: 0.05,
  },
};
const options = ["judge", "efficient", "capable"].map((model) => ({ value: model, label: model }));
const catalog = {
  version: "catalog-v1",
  models: [
    {
      id: "efficient-v1",
      label: "Efficient preset",
      text: "Maintained efficient profile",
      sources: ["https://example.com/efficient"],
      model: "efficient-model",
    },
    {
      id: "capable-v1",
      label: "Capable preset",
      text: "Maintained capable profile",
      sources: ["https://example.com/capable"],
      model: "capable-model",
    },
  ],
  harnesses: [
    {
      id: "runtime-v1",
      label: "Runtime preset",
      text: "Maintained runtime profile",
      sources: ["https://example.com/runtime"],
    },
  ],
};
const presetConfig = {
  efficient_profile_preset: catalog.models[0].id,
  capable_profile_preset: catalog.models[1].id,
  harness_preset: catalog.harnesses[0].id,
  max_quality_gap: 0.05,
};
const presetInitial = { ...fuseInitial, llm_v2_config: presetConfig };

beforeEach(() => {
  testQueryClient.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockImplementation(async () => Response.json(catalog)),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
});

function Form({ initialValue = initial }: { initialValue?: ComplexityRouterConfigValue }) {
  const [value, setValue] = useState(initialValue);
  const [saved, setSaved] = useState("");
  return (
    <>
      <AutoRouterClassifierTabs value={value} onChange={setValue}>
        {isForecastClassifier(value.classifier_type) ? (
          <ForecastClassifierConfig
            value={value}
            onChange={setValue}
            modelOptions={options}
            effortOptionsByModel={{}}
          />
        ) : (
          <ClassificationMethodConfig
            value={value}
            onChange={setValue}
            modelOptions={options}
            effortOptionsByModel={{}}
          />
        )}
      </AutoRouterClassifierTabs>
      <button
        disabled={Boolean(getForecastConfigError(value))}
        onClick={() => setSaved(JSON.stringify(buildUpdatedComplexityRouterConfig({}, value)))}
      >
        Save configuration
      </button>
      <output aria-label="Saved configuration">{saved}</output>
    </>
  );
}

describe("forecast classifier form", () => {
  it("selects all three maintained presets, previews provenance, and saves only references", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Form initialValue={fuseInitial} />);
    await user.click(screen.getByRole("combobox", { name: "Efficient solver profile preset" }));
    await user.click(await screen.findByRole("option", { name: /^Efficient preset/ }));
    await user.click(screen.getByRole("combobox", { name: "Capable solver profile preset" }));
    await user.click(screen.getByRole("option", { name: /^Capable preset/ }));
    await user.click(screen.getByRole("combobox", { name: "Harness and budget preset" }));
    await user.click(screen.getByRole("option", { name: /^Runtime preset/ }));
    expect(screen.getByLabelText("Efficient solver profile")).toHaveValue(catalog.models[0].text);
    expect(screen.getByLabelText("Efficient solver profile")).toHaveAttribute("readonly");
    expect(screen.getByLabelText("Capable solver profile")).toHaveValue(catalog.models[1].text);
    expect(screen.getByLabelText("Harness and budget")).toHaveValue(catalog.harnesses[0].text);
    expect(screen.getAllByText(`Catalog version: ${catalog.version}`)).toHaveLength(3);
    expect(screen.getByText(`Model: ${catalog.models[0].model}`)).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Source 1" }).map((link) => link.getAttribute("href"))).toEqual([
      catalog.models[0].sources[0],
      catalog.models[1].sources[0],
      catalog.harnesses[0].sources[0],
    ]);
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    expect(JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config).toEqual(
      presetConfig,
    );
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      expect.objectContaining({ url: expect.stringMatching(/\/public\/complexity_router\/fuse_presets$/) }),
    );
  });

  it.each([undefined, null, "Explicit override"])(
    "copies effective text to Custom and clears only that reference, override=%s",
    async (override) => {
      const user = userEvent.setup();
      renderWithProviders(
        <Form initialValue={{ ...presetInitial, llm_v2_config: { ...presetConfig, efficient_profile: override } }} />,
      );
      const effectiveText = override ?? catalog.models[0].text;
      await waitFor(() => expect(screen.getByLabelText("Efficient solver profile")).toHaveValue(effectiveText));
      await user.click(screen.getByRole("combobox", { name: "Efficient solver profile preset" }));
      await user.click(screen.getByRole("option", { name: "Custom" }));
      expect(screen.getByLabelText("Efficient solver profile")).not.toHaveAttribute("readonly");
      expect(screen.getByLabelText("Efficient solver profile")).toHaveValue(effectiveText);
      fireEvent.change(screen.getByLabelText("Efficient solver profile"), { target: { value: "Custom budget" } });
      fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
      const { efficient_profile_preset: _preset, ...rest } = presetConfig;
      expect(
        JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config,
      ).toEqual({
        ...rest,
        efficient_profile: "Custom budget",
      });
    },
  );

  it.each([
    { ...fuseInitial.llm_v2_config!, efficient_profile: catalog.models[0].text },
    {
      ...presetConfig,
      efficient_profile: "Explicit override",
      capable_profile: "Capable override",
      harness: "Harness override",
    },
  ])("keeps existing custom ownership and references on an unchanged save: %j", async (settings) => {
    renderWithProviders(<Form initialValue={{ ...fuseInitial, llm_v2_config: settings }} />);
    await waitFor(() => expect(screen.queryByText(/Loading profile presets/)).not.toBeInTheDocument());
    expect(screen.getByRole("combobox", { name: "Efficient solver profile preset" })).toHaveValue("Custom");
    expect(screen.getByLabelText("Efficient solver profile")).toHaveValue(settings.efficient_profile);
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    expect(JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config).toEqual(
      settings,
    );
  });

  it.each([true, false])(
    "keeps edits and stored IDs while the pending catalog settles, success=%s",
    async (success) => {
      let resolveCatalog: (response: Response) => void = () => {};
      vi.mocked(fetch).mockReturnValue(
        new Promise<Response>((resolve) => {
          resolveCatalog = resolve;
        }),
      );
      const settings = { ...presetConfig, efficient_profile: "Original override" };
      renderWithProviders(<Form initialValue={{ ...fuseInitial, llm_v2_config: settings }} />);
      expect(screen.getByText(/Loading profile presets/)).toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("Efficient solver profile"), { target: { value: "Typed while loading" } });
      await act(async () =>
        resolveCatalog(success ? Response.json(catalog) : Response.json({ error: "unavailable" }, { status: 503 })),
      );
      if (success) await screen.findAllByText(`Catalog version: ${catalog.version}`);
      else expect(await screen.findByText(/Profile presets could not be loaded/)).toBeInTheDocument();
      expect(screen.getByLabelText("Efficient solver profile")).toHaveValue("Typed while loading");
      fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
      expect(
        JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config,
      ).toEqual({ ...settings, efficient_profile: "Typed while loading" });
    },
  );

  it.each([
    ["efficient_profile", "Efficient solver profile"],
    ["capable_profile", "Capable solver profile"],
    ["harness", "Harness and budget"],
  ] as const)(
    "preserves the saved %s reference during a catalog outage until Custom text replaces it",
    async (field, label) => {
      const user = userEvent.setup();
      vi.mocked(fetch).mockImplementation(async () => Response.json({ error: "unavailable" }, { status: 503 }));
      renderWithProviders(<Form initialValue={presetInitial} />);
      expect(await screen.findByText(/Profile presets could not be loaded/)).toBeInTheDocument();
      const save = screen.getByRole("button", { name: "Save configuration" });
      const output = screen.getByRole("status", { name: "Saved configuration" });
      expect(save).toBeEnabled();
      await user.click(save);
      expect(JSON.parse(output.textContent!).llm_v2_config).toEqual(presetConfig);

      await user.click(screen.getByRole("combobox", { name: `${label} preset` }));
      await user.click(screen.getByRole("option", { name: "Custom" }));
      expect(screen.getByLabelText(label)).toHaveValue("");
      expect(screen.getByLabelText(label)).not.toHaveAttribute("readonly");
      expect(screen.getByRole("combobox", { name: `${label} preset` })).toHaveValue("Custom");
      expect(save).toBeEnabled();
      await user.click(save);
      expect(JSON.parse(output.textContent!).llm_v2_config).toEqual(presetConfig);

      fireEvent.change(screen.getByLabelText(label), { target: { value: "   " } });
      expect(save).toBeDisabled();
      await user.click(screen.getByRole("button", { name: `Keep saved ${label.toLowerCase()} preset` }));
      expect(screen.getByLabelText(label)).toHaveAttribute("readonly");
      expect(save).toBeEnabled();
      await user.click(save);
      expect(JSON.parse(output.textContent!).llm_v2_config).toEqual(presetConfig);

      await user.click(screen.getByRole("combobox", { name: `${label} preset` }));
      await user.click(screen.getByRole("option", { name: "Custom" }));
      const replacement = "Manually authored replacement";
      fireEvent.change(screen.getByLabelText(label), { target: { value: replacement } });
      expect(save).toBeEnabled();
      await user.click(save);
      const referenceKey = `${field}_preset` as const;
      const { [referenceKey]: _reference, ...remaining } = presetConfig;
      expect(JSON.parse(output.textContent!).llm_v2_config).toEqual({ ...remaining, [field]: replacement });
    },
  );

  it.each([true, false])(
    "keeps a reference selected as Custom while the catalog settles, success=%s",
    async (success) => {
      const user = userEvent.setup();
      const response = Promise.withResolvers<Response>();
      vi.mocked(fetch).mockReturnValue(response.promise);
      renderWithProviders(<Form initialValue={presetInitial} />);
      await user.click(screen.getByRole("combobox", { name: "Efficient solver profile preset" }));
      await user.click(screen.getByRole("option", { name: "Custom" }));
      await act(async () => response.resolve(success ? Response.json(catalog) : Response.json({}, { status: 503 })));
      if (success) await screen.findAllByText(`Catalog version: ${catalog.version}`);
      else await screen.findByText(/Profile presets could not be loaded/);
      expect(screen.getByLabelText("Efficient solver profile")).toHaveValue(success ? catalog.models[0].text : "");
      await user.click(screen.getByRole("button", { name: "Save configuration" }));
      expect(
        JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config,
      ).toEqual(presetConfig);
      fireEvent.change(screen.getByLabelText("Efficient solver profile"), { target: { value: "Replacement" } });
      await user.click(screen.getByRole("button", { name: "Save configuration" }));
      const { efficient_profile_preset: _reference, ...remaining } = presetConfig;
      expect(
        JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config,
      ).toEqual({
        ...remaining,
        efficient_profile: "Replacement",
      });
    },
  );

  it("keeps unknown saved IDs visible with unavailable previews rather than replacing them", async () => {
    const settings = { ...presetConfig, efficient_profile_preset: "unavailable-v8" };
    renderWithProviders(<Form initialValue={{ ...fuseInitial, llm_v2_config: settings }} />);
    await screen.findAllByText(`Catalog version: ${catalog.version}`);
    expect(screen.getByRole("combobox", { name: "Efficient solver profile preset" })).toHaveValue("unavailable-v8");
    expect(screen.getByText("Preset preview unavailable. The saved reference is preserved")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    expect(JSON.parse(screen.getByRole("status", { name: "Saved configuration" }).textContent!).llm_v2_config).toEqual(
      settings,
    );
  });

  it("switches a populated standard router to Capability without saving hidden pools or their overrides", async () => {
    renderWithProviders(
      <Form
        initialValue={{
          classifier_type: "llm",
          classifier_llm_config: { model: "judge", timeout_ms: 20000, classification_rubric: "agentic" },
          adaptive: true,
          plan_mode_min_tier: "MEDIUM",
          tiers: {
            SIMPLE: ["efficient", "second-efficient"],
            MEDIUM: ["leftover-medium"],
            COMPLEX: ["leftover-complex"],
            REASONING: ["capable"],
          },
          tier_model_params: {
            SIMPLE: { efficient: { reasoning_effort: "low", speed: "fast", max_tokens: 1024 } },
            MEDIUM: { "leftover-medium": { speed: "fast" } },
            COMPLEX: { "leftover-complex": { max_tokens: 4096 } },
            REASONING: { capable: { reasoning_effort: "high" } },
          },
        }}
      />,
    );
    await selectAutoRouterApproach("Capability");
    fireEvent.change(screen.getByLabelText("Solve probability threshold"), { target: { value: "0.7" } });
    expect(screen.getByRole("button", { name: "Save configuration" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    const output = screen.getByRole("status", { name: "Saved configuration" });
    expect(output).toHaveTextContent('"classifier_type":"capability"');
    expect(output).toHaveTextContent('"SIMPLE":["efficient","second-efficient"]');
    expect(output).toHaveTextContent('"REASONING":["capable"]');
    expect(output).toHaveTextContent('"reasoning_effort":"low","speed":"fast","max_tokens":1024');
    expect(output).toHaveTextContent('"reasoning_effort":"high"');
    expect(output).toHaveTextContent('"adaptive":false');
    expect(output).not.toHaveTextContent("leftover-medium");
    expect(output).not.toHaveTextContent("leftover-complex");
    expect(output).not.toHaveTextContent('"plan_mode_min_tier"');
  });

  it.each(["capability", "llm_v2"] as const)(
    "carries non-default solver assignments when switching away from %s",
    async (source) => {
      const pair = { efficient_tier: "MEDIUM", capable_tier: "COMPLEX" };
      const previous: ComplexityRouterConfigValue = {
        ...(source === "capability" ? initial : fuseInitial),
        tiers: { SIMPLE: [], MEDIUM: ["efficient"], COMPLEX: ["capable"], REASONING: [] },
        capability_classifier_config:
          source === "capability" ? { ...initial.capability_classifier_config!, ...pair } : undefined,
        llm_v2_config: source === "llm_v2" ? { ...fuseInitial.llm_v2_config!, ...pair } : undefined,
        plan_mode_min_tier: "COMPLEX",
        tier_model_params: { MEDIUM: { efficient: { max_tokens: 128 } }, COMPLEX: { capable: { speed: "fast" } } },
      };
      renderWithProviders(<Form initialValue={previous} />);
      await selectAutoRouterApproach(source === "capability" ? "Fuse v2" : "Capability");
      if (source === "capability") {
        fireEvent.change(screen.getByLabelText("Efficient solver profile"), { target: { value: "Small solver" } });
        fireEvent.change(screen.getByLabelText("Capable solver profile"), { target: { value: "Large solver" } });
        fireEvent.change(screen.getByLabelText("Harness and budget"), { target: { value: "One attempt" } });
        fireEvent.change(screen.getByLabelText("Maximum quality gap"), { target: { value: "0.05" } });
      } else {
        fireEvent.change(screen.getByLabelText("Solve probability threshold"), { target: { value: "0.7" } });
      }
      expect(screen.getByRole("button", { name: "Save configuration" })).toBeEnabled();
      fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
      const output = screen.getByRole("status", { name: "Saved configuration" });
      expect(output).toHaveTextContent('"efficient_tier":"MEDIUM","capable_tier":"COMPLEX"');
      expect(output).toHaveTextContent('"tiers":{"MEDIUM":["efficient"],"COMPLEX":["capable"]}');
      expect(output).toHaveTextContent('"plan_mode_min_tier":"COMPLEX"');
      expect(output).toHaveTextContent('"max_tokens":128');
      expect(output).toHaveTextContent('"speed":"fast"');
    },
  );

  it("keeps decimal and negative numbers when entered one character at a time", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Form />);
    const threshold = screen.getByLabelText("Solve probability threshold");
    await user.clear(threshold);
    await user.type(threshold, "0.65");
    expect(threshold).toHaveValue(0.65);
    await user.click(screen.getByRole("button", { name: "Classifier options" }));
    await user.click(screen.getByRole("switch", { name: "Use fitted calibration" }));
    await user.type(screen.getByLabelText("Efficient intercept"), "-0.3");
    expect(screen.getByLabelText("Efficient intercept")).toHaveValue(-0.3);
  });

  it.each([
    ["capability", "LLM Classifier"],
    ["capability", "Heuristic first"],
    ["capability", "Hybrid"],
    ["llm_v2", "LLM Classifier"],
    ["llm_v2", "Heuristic first"],
    ["llm_v2", "Hybrid"],
  ] as const)("restores the current rubric when switching %s through Complexity to %s", async (source, target) => {
    const user = userEvent.setup();
    renderWithProviders(<Form initialValue={source === "capability" ? initial : fuseInitial} />);
    await selectAutoRouterApproach("Complexity");
    fireEvent.click(screen.getByRole("radio", { name: new RegExp(`^${target}`) }));
    await user.click(screen.getByRole("combobox", { name: "Judge model" }));
    await user.click(screen.getByRole("option", { name: "judge" }));
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    const output = screen.getByRole("status", { name: "Saved configuration" });
    expect(output).toHaveTextContent('"classification_rubric":"agentic"');
    expect(output).toHaveTextContent('"model":"judge"');
    expect(output).toHaveTextContent(
      `"timeout_ms":${(source === "capability" ? initial : fuseInitial).classifier_llm_config?.timeout_ms}`,
    );
    expect(output).not.toHaveTextContent('"capability_classifier_config"');
    expect(output).not.toHaveTextContent('"llm_v2_config"');
  });

  it("saves capability threshold edits together with fitted calibration", async () => {
    renderWithProviders(<Form />);
    fireEvent.change(screen.getByLabelText("Solve probability threshold"), { target: { value: "0.6" } });
    fireEvent.click(screen.getByRole("button", { name: "Classifier options" }));
    fireEvent.click(screen.getByRole("switch", { name: "Use fitted calibration" }));
    expect(screen.getByRole("button", { name: "Save configuration" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Calibration version"), { target: { value: "eval-a" } });
    fireEvent.change(screen.getByLabelText("Efficient slope"), { target: { value: "1.2" } });
    fireEvent.change(screen.getByLabelText("Efficient intercept"), { target: { value: "-0.3" } });
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    const output = screen.getByRole("status", { name: "Saved configuration" });
    expect(output).toHaveTextContent('"base_threshold":0.6');
    expect(output).toHaveTextContent('"calibration":{"version":"eval-a","slope":1.2,"intercept":-0.3}');
    fireEvent.change(screen.getByLabelText("Solve probability threshold"), { target: { value: "" } });
    expect(screen.getByRole("button", { name: "Save configuration" })).toBeDisabled();
  });

  it("switches to Fuse, requires solver context, and saves the filled fields", async () => {
    renderWithProviders(<Form />);
    await selectAutoRouterApproach("Fuse v2");
    expect(screen.queryByLabelText("Solve probability threshold")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save configuration" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Efficient solver profile"), {
      target: { value: "Short reasoning budget" },
    });
    fireEvent.change(screen.getByLabelText("Capable solver profile"), { target: { value: "Larger reasoning budget" } });
    fireEvent.change(screen.getByLabelText("Harness and budget"), {
      target: { value: "Shell and test runner, one attempt" },
    });
    fireEvent.change(screen.getByLabelText("Maximum quality gap"), { target: { value: "0.05" } });
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    const output = screen.getByRole("status", { name: "Saved configuration" });
    expect(output).toHaveTextContent('"classifier_type":"llm_v2"');
    expect(output).toHaveTextContent('"efficient_profile":"Short reasoning budget"');
    expect(output).toHaveTextContent('"capable_profile":"Larger reasoning budget"');
    expect(output).toHaveTextContent('"harness":"Shell and test runner, one attempt"');
    expect(output).toHaveTextContent('"max_quality_gap":0.05');
    expect(output).toHaveTextContent('"adaptive":false');
    expect(output).not.toHaveTextContent('"capability_classifier_config"');
  });
});
