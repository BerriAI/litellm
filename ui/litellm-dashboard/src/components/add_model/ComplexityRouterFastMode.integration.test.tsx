import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import {
  buildUpdatedComplexityRouterConfig,
  hydrateComplexityRouterConfig,
} from "../edit_auto_router/edit_auto_router_modal";
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

  expect(screen.getAllByRole("switch", { name: /^Fast mode for/ })).toHaveLength(3);
  expect(screen.queryByRole("switch", { name: /^Fast mode for (blocked|missing)/ })).not.toBeInTheDocument();
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

describe("Fast mode metadata", () => {
  it("offers nothing before model capabilities load and leaves stored speed untouched", () => {
    const value: ComplexityRouterConfigValue = {
      tiers: { SIMPLE: ["primary"], MEDIUM: [], COMPLEX: [], REASONING: [] },
      classifier_type: "heuristic",
      tier_model_params: { SIMPLE: { primary: { speed: "fast" } } },
    };
    const onChange = vi.fn();
    renderWithProviders(<ComplexityRouterConfig modelInfo={[]} value={value} onChange={onChange} />);
    expect(screen.queryByRole("switch", { name: /^Fast mode for/ })).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    expect(buildUpdatedComplexityRouterConfig({}, value).tier_model_configs).toEqual({
      SIMPLE: [{ model_name: "primary", litellm_params: { speed: "fast" } }],
    });
  });
});
