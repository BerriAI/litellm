import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
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
  it("switches a populated standard router to Capability without saving hidden pools or their overrides", () => {
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
    fireEvent.click(screen.getByRole("tab", { name: "Capability" }));
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
    (source) => {
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
      fireEvent.click(screen.getByRole("tab", { name: source === "capability" ? "Fuse v2" : "Capability" }));
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
    fireEvent.click(screen.getByRole("tab", { name: "Complexity" }));
    fireEvent.click(screen.getByRole("radio", { name: new RegExp(`^${target}`) }));
    await user.click(screen.getByRole("combobox", { name: "Classifier Model" }));
    await user.click(screen.getByRole("option", { name: "judge", exact: true }));
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));
    const output = screen.getByRole("status", { name: "Saved configuration" });
    expect(output).toHaveTextContent('"classification_rubric":"agentic"');
    expect(output).toHaveTextContent('"model":"judge"');
    expect(output).toHaveTextContent('"timeout_ms":3000');
    expect(output).not.toHaveTextContent('"capability_classifier_config"');
    expect(output).not.toHaveTextContent('"llm_v2_config"');
  });

  it("saves capability threshold edits together with fitted calibration", () => {
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

  it("switches to Fuse, requires solver context, and saves the filled fields", () => {
    renderWithProviders(<Form />);
    fireEvent.click(screen.getByRole("tab", { name: "Fuse v2" }));
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
