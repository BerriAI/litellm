import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const initial: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  tiers: { SIMPLE: ["efficient"], MEDIUM: [], COMPLEX: [], REASONING: ["capable"] },
};

function Form({ initialValue = initial }: { initialValue?: ComplexityRouterConfigValue }) {
  const [value, setValue] = useState(initialValue);
  return (
    <AutoRouterClassifierTabs value={value} onChange={setValue}>
      <output aria-label="Classifier type">{value.classifier_type}</output>
    </AutoRouterClassifierTabs>
  );
}

describe("AutoRouterClassifierTabs", () => {
  it.each(["heuristic", "heuristic_v2", "llm", "heuristic_first", "hybrid"] as const)(
    "groups %s under Complexity without resetting its configuration",
    (classifier_type) => {
      const onChange = vi.fn();
      renderWithProviders(
        <AutoRouterClassifierTabs value={{ ...initial, classifier_type }} onChange={onChange}>
          Existing classifier settings
        </AutoRouterClassifierTabs>,
      );
      expect(screen.getByRole("tab", { name: "Complexity" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tabpanel", { name: "Complexity" })).toHaveTextContent("Existing classifier settings");
      fireEvent.click(screen.getByRole("tab", { name: "Complexity" }));
      expect(onChange).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["capability", "Capability"],
    ["llm_v2", "Fuse v2"],
  ] as const)("opens saved %s settings and switches back to local Complexity", (classifier_type, label) => {
    renderWithProviders(<Form initialValue={{ ...initial, classifier_type }} />);
    expect(screen.getByRole("tab", { name: label })).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByRole("tab", { name: "Complexity" }));
    expect(screen.getByRole("tab", { name: "Complexity" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent("heuristic");
  });

  it("keeps custom tiers editable under Complexity and explains why forecast tabs are disabled", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <AutoRouterClassifierTabs
        value={{
          ...initial,
          custom_tier_set: {
            tiers: [{ id: "review", name: "REVIEW", definition: "Code reviews", models: ["capable"] }],
            fallback_tier_id: "review",
          },
        }}
        onChange={onChange}
      >
        Custom tiers
      </AutoRouterClassifierTabs>,
    );
    expect(screen.getByRole("tabpanel", { name: "Complexity" })).toHaveTextContent("Custom tiers");
    for (const name of ["Capability", "Fuse v2"]) {
      const tab = screen.getByRole("tab", { name });
      expect(tab).toHaveAttribute("aria-disabled", "true");
      expect(tab).toHaveAccessibleDescription("Restore standard tiers to use Capability or Fuse v2.");
      fireEvent.click(tab);
    }
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByText("Restore standard tiers to use Capability or Fuse v2.")).toBeVisible();
  });
});
