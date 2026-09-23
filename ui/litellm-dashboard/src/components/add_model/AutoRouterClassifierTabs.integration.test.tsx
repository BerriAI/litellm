import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen, waitFor, within } from "../../../tests/test-utils";
import { selectAutoRouterOption } from "../../../tests/autoRouterSetup";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import { AutoRouterAllowanceNote, AutoRouterAvailabilityContext } from "./AutoRouterAvailability";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const initial: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  tiers: { SIMPLE: ["efficient"], MEDIUM: [], COMPLEX: [], REASONING: ["capable"] },
};

function Form({
  initialValue = initial,
  remaining = 1,
  limit = 1,
  ownedFeature,
  availabilityState,
}: {
  initialValue?: ComplexityRouterConfigValue;
  remaining?: number;
  limit?: number | null;
  ownedFeature?: string;
  availabilityState?: Partial<React.ContextType<typeof AutoRouterAvailabilityContext>>;
}) {
  const [value, setValue] = useState(initialValue);
  return (
    <AutoRouterAvailabilityContext.Provider
      value={{
        isPending: false,
        isError: false,
        data: {
          allowances: ["heuristic_v2", "capability", "llm_v2", "tier_or_classifier_prompt", "heuristic_tuning"].map(
            (key) => ({
              key,
              limit,
              remaining,
              available: true,
              used_by_this_router: key === ownedFeature,
            }),
          ),
          error: null,
        },
        ...availabilityState,
      }}
    >
      <AutoRouterClassifierTabs value={value} onChange={setValue}>
        <output aria-label="Classifier type">{value.classifier_type}</output>
      </AutoRouterClassifierTabs>
    </AutoRouterAvailabilityContext.Provider>
  );
}

describe("Auto-router classifier selection", () => {
  it.each(["heuristic", "heuristic_v2", "llm", "heuristic_first", "hybrid", "jev"] as const)(
    "shows saved %s without changing its configuration",
    async (classifier_type) => {
      const onChange = vi.fn();
      renderWithProviders(
        <AutoRouterClassifierTabs value={{ ...initial, classifier_type }} onChange={onChange}>
          Existing settings
        </AutoRouterClassifierTabs>,
      );
      const family = {
        heuristic: "Heuristics",
        heuristic_v2: "Heuristics",
        llm: "LLM",
        heuristic_first: "LLM",
        hybrid: "LLM",
        jev: "Jev",
      }[classifier_type];
      expect(screen.getByRole("radio", { name: new RegExp(`^${family}$`) })).toBeChecked();
      fireEvent.click(screen.getByRole("radio", { name: new RegExp(`^${family}$`) }));
      expect(onChange).not.toHaveBeenCalled();
    },
  );

  it.each([
    ["capability", "Capability"],
    ["llm_v2", "Fuse v2"],
  ] as const)(
    "opens saved %s and retains the LLM family when switching to Complexity",
    async (classifier_type, label) => {
      renderWithProviders(<Form initialValue={{ ...initial, classifier_type }} />);
      expect(screen.getByRole("button", { name: "Routing approach" })).toHaveTextContent(label);
      await selectAutoRouterOption("Routing approach", "Complexity");
      expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent("llm");
    },
  );

  it.each([
    [1, "heuristic"],
    [0, "heuristic"],
  ])("defaults to Rule-based when %s v2 slots remain", async (remaining, classifier) => {
    renderWithProviders(<Form remaining={Number(remaining)} />);
    fireEvent.click(screen.getByRole("radio", { name: /^Heuristics$/ }));
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent(String(classifier));
    fireEvent.click(screen.getByRole("button", { name: "Heuristic" }));
    expect(screen.getByRole("menuitemradio", { name: /^Heuristic v2/ })).toHaveTextContent(
      `${remaining} of 1 available`,
    );
  });

  it.each([
    { data: undefined },
    { isPending: true },
    { isError: true },
    { isChecking: true },
    { data: { allowances: [], error: null } },
    { data: { allowances: [{ key: "heuristic_v2", limit: 1, remaining: null, available: false }], error: null } },
  ])("uses Rule-based when v2 availability is unverified: %j", async (availabilityState) => {
    renderWithProviders(<Form availabilityState={availabilityState} />);
    fireEvent.click(screen.getByRole("radio", { name: "Heuristics" }));
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent(/^heuristic$/);
  });

  it("does not present Rule-based as having a classifier quota", () => {
    renderWithProviders(<Form initialValue={{ ...initial, classifier_type: "heuristic" }} remaining={0} />);
    expect(screen.getByRole("button", { name: "Heuristic" })).toHaveTextContent(/^Rule-based/);
    fireEvent.click(screen.getByRole("button", { name: "Heuristic" }));
    expect(screen.getAllByRole("menuitemradio")[0]).toHaveTextContent(/^Rule-based/);
    expect(screen.getByRole("menuitemradio", { name: /^Rule-based/ })).not.toHaveTextContent("of 1 available");
    expect(screen.getByRole("menuitemradio", { name: /^Heuristic v2/ })).toHaveTextContent("0 of 1 available");
  });

  it("omits allowance labels with an unlimited entitlement", async () => {
    renderWithProviders(<Form initialValue={{ ...initial, classifier_type: "heuristic_v2" }} limit={null} />);
    fireEvent.click(screen.getByRole("button", { name: "Heuristic" }));
    expect(screen.getByRole("menuitemradio", { name: /^Heuristic v2/ })).not.toHaveTextContent("available");
  });

  it.each([
    ["heuristic", "Heuristic", "Heuristic v2"],
    ["llm", "Routing approach", "Capability"],
    ["llm", "Routing approach", "Fuse v2"],
  ] as const)("blocks exhausted %s options: %s / %s", (classifier_type, field, option) => {
    renderWithProviders(<Form initialValue={{ ...initial, classifier_type }} remaining={0} />);
    fireEvent.click(screen.getByRole("button", { name: field }));
    const unavailable = screen.getByRole("menuitemradio", { name: new RegExp(`^${option}`) });
    expect(unavailable).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(unavailable);
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent(classifier_type);
  });

  it.each([
    ["heuristic_v2", "heuristic", "Heuristic", "Heuristic v2"],
    ["capability", "llm", "Routing approach", "Capability"],
    ["llm_v2", "llm", "Routing approach", "Fuse v2"],
  ] as const)("lets a saved router reselect its own %s allowance", async (feature, classifier_type, field, option) => {
    renderWithProviders(<Form initialValue={{ ...initial, classifier_type }} remaining={0} ownedFeature={feature} />);
    await selectAutoRouterOption(field, option);
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent(feature);
    expect(screen.getByRole("button", { name: field })).toHaveTextContent("Used by this router");
  });

  it("shows Jev's single Complexity approach without changing saved configuration", () => {
    const onChange = vi.fn();
    renderWithProviders(
      <AutoRouterClassifierTabs value={{ ...initial, classifier_type: "jev" }} onChange={onChange}>
        Existing settings
      </AutoRouterClassifierTabs>,
    );
    expect(screen.getByRole("button", { name: "Routing approach" })).toHaveTextContent("ComplexityUnlimited");
    fireEvent.click(screen.getByRole("button", { name: "Routing approach" }));
    expect(screen.getAllByRole("menuitemradio")).toHaveLength(1);
    fireEvent.click(screen.getByRole("menuitemradio", { name: /^Complexity/ }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("keeps custom tiers editable and disables incompatible choices", async () => {
    renderWithProviders(
      <Form
        initialValue={{
          ...initial,
          custom_tier_set: {
            tiers: [{ id: "review", name: "REVIEW", definition: "Code reviews", models: ["capable"] }],
            fallback_tier_id: "review",
          },
        }}
      />,
    );
    expect(screen.getByRole("radio", { name: /^Heuristics$/ })).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(screen.getByRole("button", { name: "Routing approach" }));
    for (const name of ["Capability", "Fuse v2"]) {
      expect(screen.getByRole("menuitemradio", { name: new RegExp(`^${name}`) })).toHaveAttribute(
        "aria-disabled",
        "true",
      );
    }
    expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent("llm");
  });
});

describe("Gated routing contact action", () => {
  it("offers a pricing discussion in View limits", async () => {
    renderWithProviders(<Form remaining={0} />);
    expect(screen.queryByRole("link", { name: "Talk to our team" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "View limits" }));
    const link = within(screen.getByRole("dialog")).getByRole("link", { name: "Talk to our team" });
    await waitFor(() => expect(link).toBeVisible());
    expect(link).toHaveAttribute("href", "https://calendly.com/tin-berri/litellm-auto-router-pricing-discussion");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it.each([
    ["heuristic", "Heuristic", "Heuristic v2"],
    ["llm", "Routing approach", "Capability"],
  ] as const)(
    "keeps the contact action available beside the disabled %s choice",
    async (classifier_type, field, option) => {
      renderWithProviders(<Form initialValue={{ ...initial, classifier_type }} remaining={0} />);
      expect(screen.queryByText(/Need more/)).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: field }));
      const disabled = screen.getByRole("menuitemradio", { name: new RegExp(`^${option}`) });
      expect(disabled).toHaveAttribute("aria-disabled", "true");
      const link = screen.getByRole("menuitem", { name: `Talk to our team about ${option}` });
      await waitFor(() => expect(link).toBeVisible());
      expect(link).toHaveAttribute("href", "https://calendly.com/tin-berri/litellm-auto-router-pricing-discussion");
      expect(link).toHaveAttribute("target", "_blank");
      fireEvent.click(link);
      expect(screen.getByRole("status", { name: "Classifier type" })).toHaveTextContent(classifier_type);
    },
  );

  it.each([
    { remaining: 1 },
    { remaining: 0, limit: null },
    { remaining: 0, availabilityState: { isPending: true } },
    { remaining: 0, availabilityState: { isError: true } },
    { remaining: 0, availabilityState: { isChecking: true } },
  ])("does not pitch an upgrade for a free or unverified option: %j", (props) => {
    renderWithProviders(<Form {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Routing approach" }));
    expect(screen.queryByRole("menuitem", { name: /Talk to our team/ })).not.toBeInTheDocument();
  });

  it("does not pitch an upgrade for the saved heuristic's own slot", () => {
    renderWithProviders(
      <Form initialValue={{ ...initial, classifier_type: "heuristic_v2" }} remaining={0} ownedFeature="heuristic_v2" />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Heuristic" }));
    expect(screen.queryByRole("menuitem", { name: /Talk to our team/ })).not.toBeInTheDocument();
  });

  it("includes the sales action beside customization limits and blocked changes", () => {
    const allowance = { key: "tier_or_classifier_prompt", limit: 1, remaining: 0, available: true };
    const state = {
      isPending: false,
      isError: false,
      data: { allowances: [allowance], error: "Custom tiers have no available allowance" },
    };
    renderWithProviders(
      <AutoRouterAvailabilityContext.Provider value={state}>
        <AutoRouterClassifierTabs value={initial} onChange={vi.fn()}>
          <AutoRouterAllowanceNote feature="tier_or_classifier_prompt" label="Custom tiers" />
        </AutoRouterClassifierTabs>
      </AutoRouterAvailabilityContext.Provider>,
    );
    expect(screen.getByText(/Custom tiers: 0 of 1 available/)).toHaveTextContent("Talk to our team");
    expect(within(screen.getByRole("alert")).getByRole("link", { name: "Talk to our team" })).toBeVisible();
  });
});
