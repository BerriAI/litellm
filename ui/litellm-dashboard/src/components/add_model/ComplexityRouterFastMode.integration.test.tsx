import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, within } from "../../../tests/test-utils";
import {
  buildUpdatedComplexityRouterConfig,
  hydrateComplexityRouterConfig,
} from "../edit_auto_router/edit_auto_router_modal";
import type { KeywordTierRule } from "./KeywordTierRules";
import type { ModelGroup } from "../llm_calls/fetch_models";
import ComplexityRouterConfig, { type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const modelInfo: ModelGroup[] = [
  { model_group: "primary", supported_reasoning_efforts: ["low", "high"], supports_fast_mode: true },
  { model_group: "secondary", supports_fast_mode: true },
  { model_group: "blocked", supported_reasoning_efforts: ["low"], supports_fast_mode: false },
  { model_group: "missing", supported_reasoning_efforts: ["low"] },
];

it.each([false, true])("edits and round-trips independent model settings with custom tiers=%s", async (custom) => {
  const user = userEvent.setup();
  const tier = custom ? "custom-a" : "COMPLEX";
  const otherTier = custom ? "custom-b" : "REASONING";
  const label = custom ? "Interactive" : "Complex";
  const models = ["primary", "secondary", "blocked", "missing"];
  const initial: ComplexityRouterConfigValue = {
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: models, REASONING: ["primary"] },
    classifier_type: "heuristic",
    ...(custom && {
      custom_tier_set: {
        tiers: [
          { id: tier, name: label, definition: "Interactive requests", models },
          { id: otherTier, name: "Deliberate", definition: "Careful requests", models: ["primary"] },
        ],
        fallback_tier_id: tier,
      },
    }),
    tier_model_params: {
      [tier]: {
        primary: { reasoning_effort: "high", max_tokens: 1024 },
        secondary: { speed: "fast" },
        blocked: { speed: "fast" },
      },
      [otherTier]: { primary: { speed: "fast", reasoning_effort: "low" } },
    },
  };
  const onChange = vi.fn<(value: ComplexityRouterConfigValue) => void>();
  const editor = (value: ComplexityRouterConfigValue) => (
    <ComplexityRouterConfig modelInfo={modelInfo} value={value} onChange={onChange} />
  );
  const view = renderWithProviders(editor(initial));
  const fast = () => screen.getByRole("switch", { name: `Fast mode for primary in the ${label} tier` });

  expect(screen.getAllByRole("switch", { name: /^Fast mode for/ })).toHaveLength(4);
  expect(screen.queryByRole("switch", { name: /^Fast mode for missing/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("combobox", { name: /^Reasoning effort for secondary/ })).not.toBeInTheDocument();
  expect(screen.getByRole("switch", { name: `Fast mode for secondary in the ${label} tier` })).toBeChecked();
  expect(fast()).not.toBeChecked();
  expect(onChange).not.toHaveBeenCalled();

  await user.click(fast());
  const enabled = onChange.mock.lastCall![0];
  expect(enabled.tier_model_params).toEqual({
    ...initial.tier_model_params,
    [tier]: {
      ...initial.tier_model_params![tier],
      primary: { reasoning_effort: "high", max_tokens: 1024, speed: "fast" },
    },
  });
  const saved = buildUpdatedComplexityRouterConfig({}, enabled);
  expect(saved.tier_model_configs).toEqual({
    [custom ? label : tier]: [
      { model_name: "primary", litellm_params: { reasoning_effort: "high", max_tokens: 1024, speed: "fast" } },
      { model_name: "secondary", litellm_params: { speed: "fast" } },
      { model_name: "blocked", litellm_params: { speed: "fast" } },
    ],
    [custom ? "Deliberate" : otherTier]: [
      { model_name: "primary", litellm_params: { speed: "fast", reasoning_effort: "low" } },
    ],
  });
  const reopened = hydrateComplexityRouterConfig(saved, undefined);
  const reopenedTier = custom ? reopened.custom_tier_set!.tiers[0].id : tier;
  view.rerender(editor(reopened));
  expect(fast()).toBeChecked();

  await user.click(screen.getByRole("combobox", { name: `Reasoning effort for primary in the ${label} tier` }));
  await user.click(await screen.findByRole("option", { name: "low" }));
  const effortChanged = onChange.mock.lastCall![0];
  expect(effortChanged.tier_model_params?.[reopenedTier].primary).toEqual({
    reasoning_effort: "low",
    max_tokens: 1024,
    speed: "fast",
  });
  view.rerender(editor(effortChanged));
  await user.click(fast());
  const disabled = onChange.mock.lastCall![0];
  expect(disabled.tier_model_params).toEqual({
    ...effortChanged.tier_model_params,
    [reopenedTier]: {
      ...effortChanged.tier_model_params![reopenedTier],
      primary: { reasoning_effort: "low", max_tokens: 1024 },
    },
  });
  view.rerender(editor(disabled));
  expect(fast()).not.toBeChecked();

  const picker = () => screen.getByRole("combobox", { name: `Select model(s) for ${label.toLowerCase()} queries` });
  await user.click(picker());
  await user.click(await screen.findByRole("option", { name: "primary" }));
  await user.keyboard("{Escape}");
  const deselected = onChange.mock.lastCall![0];
  expect(deselected.tier_model_params?.[reopenedTier]).toEqual({
    secondary: { speed: "fast" },
    blocked: { speed: "fast" },
  });
  view.rerender(editor(deselected));
  expect(screen.queryByRole("switch", { name: `Fast mode for primary in the ${label} tier` })).not.toBeInTheDocument();
  await user.click(picker());
  await user.click(await screen.findByRole("option", { name: "primary" }));
  await user.keyboard("{Escape}");
  const reselected = onChange.mock.lastCall![0];
  view.rerender(editor(reselected));
  expect(fast()).not.toBeChecked();
  expect(screen.getByRole("combobox", { name: `Reasoning effort for primary in the ${label} tier` })).toHaveTextContent(
    "Default",
  );
});

it.each(["capability", "llm_v2"] as const)("preserves Fast mode controls for %s solvers", async (classifierType) => {
  const user = userEvent.setup();
  const initial: ComplexityRouterConfigValue = {
    classifier_type: classifierType,
    classifier_llm_config: { model: "primary", timeout_ms: 3000 },
    tiers: { SIMPLE: ["primary"], MEDIUM: [], COMPLEX: [], REASONING: ["blocked"] },
    capability_classifier_config: { efficient_tier: "SIMPLE", capable_tier: "REASONING", base_threshold: 0.7 },
    llm_v2_config: {
      efficient_profile: "Small solver",
      capable_profile: "Large solver",
      harness: "One attempt",
      max_quality_gap: 0.05,
    },
    tier_model_params: { SIMPLE: { primary: { reasoning_effort: "high", max_tokens: 1024 } } },
  };
  const onChange = vi.fn<(value: ComplexityRouterConfigValue) => void>();
  const editor = (value: ComplexityRouterConfigValue) => (
    <ComplexityRouterConfig modelInfo={modelInfo} value={value} onChange={onChange} />
  );
  const view = renderWithProviders(editor(initial));
  const fast = () => screen.getByRole("switch", { name: "Fast mode for primary in the Efficient solver tier" });
  expect(screen.getAllByRole("switch", { name: /^Fast mode for/ })).toHaveLength(1);
  expect(fast()).not.toBeChecked();
  await user.click(fast());
  const enabled = onChange.mock.lastCall![0];
  expect(enabled.tier_model_params?.SIMPLE.primary).toEqual({
    reasoning_effort: "high",
    max_tokens: 1024,
    speed: "fast",
  });
  const saved = buildUpdatedComplexityRouterConfig({}, enabled);
  expect(saved.tier_model_configs).toEqual({
    SIMPLE: [{ model_name: "primary", litellm_params: { reasoning_effort: "high", max_tokens: 1024, speed: "fast" } }],
  });
  view.rerender(editor(hydrateComplexityRouterConfig(saved, undefined)));
  expect(fast()).toBeChecked();
  await user.click(fast());
  expect(onChange.mock.lastCall![0].tier_model_params?.SIMPLE.primary).toEqual({
    reasoning_effort: "high",
    max_tokens: 1024,
  });
});

describe("Fast mode metadata", () => {
  it.each(["heuristic", "capability", "llm_v2"] as const)(
    "can clear stored Fast mode without current capability metadata for %s",
    async (classifier_type) => {
      const user = userEvent.setup();
      const value: ComplexityRouterConfigValue = {
        tiers: { SIMPLE: ["primary"], MEDIUM: [], COMPLEX: [], REASONING: ["secondary"] },
        classifier_type,
        tier_model_params: { SIMPLE: { primary: { speed: "fast", max_tokens: 512 } } },
      };
      const onChange = vi.fn<(value: ComplexityRouterConfigValue) => void>();
      const editor = (current: ComplexityRouterConfigValue, info: ModelGroup[]) => (
        <ComplexityRouterConfig modelInfo={info} value={current} onChange={onChange} />
      );
      const view = renderWithProviders(editor(value, []));
      const fast = () => screen.getByRole("switch", { name: /^Fast mode for primary/ });
      expect(fast()).toBeChecked();
      expect(onChange).not.toHaveBeenCalled();
      view.rerender(editor(value, [{ model_group: "primary", supports_fast_mode: false }]));
      expect(fast()).toBeChecked();
      await user.click(fast());
      const cleared = onChange.mock.lastCall![0];
      expect(cleared.tier_model_params?.SIMPLE.primary).toEqual({ max_tokens: 512 });
      const saved = buildUpdatedComplexityRouterConfig({}, cleared);
      expect(saved.tier_model_configs).toEqual({
        SIMPLE: [{ model_name: "primary", litellm_params: { max_tokens: 512 } }],
      });
      view.rerender(editor(hydrateComplexityRouterConfig(saved, undefined), []));
      expect(screen.queryByRole("switch", { name: /^Fast mode for primary/ })).not.toBeInTheDocument();
      view.rerender(editor(cleared, modelInfo));
      expect(fast()).not.toBeChecked();
    },
  );
});

it.each(["MEDIUM", "REASONING"])("clears a legacy Capability pool while reconciling plan floor %s", async (floor) => {
  const user = userEvent.setup();
  const stored = {
    classifier_type: "capability" as const,
    plan_mode_min_tier: floor,
    tiers: { SIMPLE: ["primary"], MEDIUM: ["secondary"], COMPLEX: [], REASONING: ["blocked"] },
    tier_model_configs: { MEDIUM: [{ model_name: "secondary", litellm_params: { speed: "fast" } }] },
  };
  const value = hydrateComplexityRouterConfig(stored, undefined);
  const onChange = vi.fn<(value: ComplexityRouterConfigValue) => void>();
  renderWithProviders(<ComplexityRouterConfig modelInfo={modelInfo} value={value} onChange={onChange} />);
  await user.click(screen.getByRole("button", { name: "Advanced routing options" }));
  expect(screen.getByRole("switch", { name: "Fast mode for secondary in the Medium routing pool tier" })).toBeChecked();
  expect(onChange).not.toHaveBeenCalled();
  await user.click(screen.getByRole("combobox", { name: "Select medium routing pool models" }));
  await user.click(await screen.findByRole("option", { name: "secondary" }));
  await user.keyboard("{Escape}");
  const cleared = onChange.mock.lastCall![0];
  expect(cleared.tiers.MEDIUM).toEqual([]);
  expect(cleared.plan_mode_min_tier).toBe(floor === "MEDIUM" ? undefined : floor);
  expect(cleared.tier_model_params).toBeUndefined();
  expect(buildUpdatedComplexityRouterConfig(stored, cleared).tiers).toEqual({
    SIMPLE: ["primary"],
    REASONING: ["blocked"],
  });
});

it.each(["capability", "llm_v2"] as const)(
  "shows and clears a persisted default model in %s",
  async (classifier_type) => {
    const user = userEvent.setup();
    const stored = {
      classifier_type,
      default_model: "legacy-default",
      tiers: { SIMPLE: ["primary"], REASONING: ["secondary"] },
    };
    const onChange = vi.fn<(value: ComplexityRouterConfigValue) => void>();
    const editor = (value: ComplexityRouterConfigValue) => (
      <ComplexityRouterConfig modelInfo={modelInfo} value={value} onChange={onChange} />
    );
    const view = renderWithProviders(editor(hydrateComplexityRouterConfig(stored, undefined)));
    await user.click(screen.getByRole("button", { name: "Advanced routing options" }));
    const select = () => screen.getByRole("combobox", { name: "Default model" });
    expect(select()).toHaveValue("legacy-default");
    expect(onChange).not.toHaveBeenCalled();
    await user.click(select());
    await user.click(await screen.findByRole("option", { name: "blocked" }));
    const changed = onChange.mock.lastCall![0];
    expect(buildUpdatedComplexityRouterConfig(stored, changed).default_model).toBe("blocked");
    view.rerender(editor(changed));
    await user.click(
      within(screen.getByRole("group", { name: "Default model configuration" })).getByRole("button", { name: "Clear" }),
    );
    const cleared = onChange.mock.lastCall![0];
    expect(cleared.default_model).toBeUndefined();
    const saved = buildUpdatedComplexityRouterConfig(stored, cleared);
    expect(saved).not.toHaveProperty("default_model");
    view.rerender(editor(hydrateComplexityRouterConfig(saved, undefined)));
    expect(select()).toHaveValue("");
    expect(select()).toHaveAttribute("placeholder", expect.stringContaining("primary"));
  },
);

it.each(["capability", "llm_v2"] as const)("offers only populated keyword targets for %s", async (classifier_type) => {
  const user = userEvent.setup();
  const value: ComplexityRouterConfigValue = {
    classifier_type,
    tiers: { SIMPLE: ["primary"], MEDIUM: [], COMPLEX: [], REASONING: ["secondary"] },
  };
  const onRulesChange = vi.fn<(rules: KeywordTierRule[]) => void>();
  const editor = (rules: KeywordTierRule[]) => (
    <ComplexityRouterConfig
      value={value}
      onChange={vi.fn()}
      modelInfo={modelInfo}
      keywordTierRules={rules}
      onKeywordTierRulesChange={onRulesChange}
    />
  );
  const view = renderWithProviders(editor([]));
  await user.click(screen.getByRole("button", { name: "Advanced routing options" }));
  await user.click(screen.getByText("Advanced: Keyword/Semantic Matching"));
  await user.click(screen.getByRole("button", { name: "Add keyword rule" }));
  const rules = onRulesChange.mock.lastCall![0];
  expect(rules[0].tier).toBe("SIMPLE");
  view.rerender(editor(rules));
  await user.click(screen.getByRole("combobox", { name: "Route keyword rule 1 to tier" }));
  expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual(["Simple", "Reasoning"]);
});
