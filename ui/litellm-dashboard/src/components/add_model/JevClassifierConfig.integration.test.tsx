import React, { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ClassificationMethodConfig from "./ClassificationMethodConfig";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import JevEditor from "./JevClassifierConfig";
import { type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import {
  buildUpdatedComplexityRouterConfig,
  hydrateComplexityRouterConfig,
} from "../edit_auto_router/edit_auto_router_modal";
import { applyTierSetAction } from "./tier_set_actions";
import { testAutoRouterRouting } from "../networking";
import { JEV_CONNECTION_TEST_PROMPT } from "./build_auto_router_routing_test_request";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    isLoading: false,
    isAuthorized: true,
    token: "token",
    accessToken: "token",
    userId: "user",
    userEmail: "user@example.com",
    userRole: "Admin",
    userRoleLabel: "Admin",
    isViewOnly: false,
    premiumUser: false,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  })),
}));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  getComplexityScorerDefaults: vi.fn(async () => ({
    tier_boundaries: {},
    token_thresholds: {},
    dimension_weights: {},
  })),
  testAutoRouterRouting: vi.fn(async () => ({ status: "error", error: "fixture" })),
}));

const initial: ComplexityRouterConfigValue = {
  classifier_type: "llm",
  classifier_llm_config: { model: "judge", timeout_ms: 1000 },
  tiers: { SIMPLE: ["fast"], MEDIUM: ["mid"], COMPLEX: ["strong"], REASONING: ["reasoner"] },
};

function Form() {
  const [value, setValue] = useState(initial);
  return (
    <AutoRouterClassifierTabs value={value} onChange={setValue}>
      <ClassificationMethodConfig
        value={value}
        onChange={setValue}
        modelOptions={[{ value: "judge", label: "judge" }]}
        effortOptionsByModel={{ judge: ["low"] }}
        customTechnicalKeywords={[]}
        onCustomTechnicalKeywordsChange={() => {}}
      />
      <button
        onClick={() =>
          setValue(
            applyTierSetAction(value, [], {
              kind: "patch",
              id: "SIMPLE",
              patch: { name: "QUICK", definition: "Quick tasks" },
            }).value,
          )
        }
      >
        Customize tiers
      </button>
      <button
        onClick={() =>
          setValue(hydrateComplexityRouterConfig(buildUpdatedComplexityRouterConfig({}, value), undefined))
        }
      >
        Save and reload
      </button>
      <button
        onClick={() => {
          const request = {
            prompt: JEV_CONNECTION_TEST_PROMPT,
            complexity_router_config: buildUpdatedComplexityRouterConfig({}, value),
          };
          void testAutoRouterRouting("token", request);
        }}
      >
        Probe current config
      </button>
    </AutoRouterClassifierTabs>
  );
}

describe("JEV classifier editor", () => {
  afterEach(() => vi.mocked(useAuthorized).mockReset());
  it("creates Nimble connection settings and retains the provider after save and reload", () => {
    renderWithProviders(<Form />);
    fireEvent.click(screen.getByRole("radio", { name: "Decision Model" }));
    fireEvent.click(screen.getByRole("radio", { name: "Nimble (open source)" }));
    expect(screen.getByLabelText("Nimble Model")).toHaveValue("nimble-latest");
    fireEvent.change(screen.getByLabelText("API Base"), { target: { value: "https://nimble.example.com" } });
    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "new-nimble-key" } });
    fireEvent.change(screen.getByLabelText("Nimble Timeout (ms)"), { target: { value: "20000" } });
    fireEvent.click(screen.getByRole("button", { name: "Probe current config" }));
    expect(testAutoRouterRouting).toHaveBeenLastCalledWith(
      "token",
      expect.objectContaining({
        complexity_router_config: expect.objectContaining({
          classifier_type: "jev",
          jev_classifier_config: {
            provider: "bespoke_nimble",
            model: "nimble-latest",
            timeout_ms: 20000,
            api_base: "https://nimble.example.com",
            api_key: "new-nimble-key",
          },
        }),
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Save and reload" }));
    expect(screen.getByRole("radio", { name: "Nimble (open source)" })).toBeChecked();
    expect(screen.getByLabelText("Nimble Timeout (ms)")).toHaveValue(20000);
    expect(screen.getByLabelText("API Key")).toHaveValue("");
    fireEvent.click(screen.getByText("Connection settings"));
    fireEvent.click(screen.getByRole("button", { name: "Clear saved API key" }));
    fireEvent.click(screen.getByRole("button", { name: "Probe current config" }));
    expect(testAutoRouterRouting).toHaveBeenLastCalledWith(
      "token",
      expect.objectContaining({
        complexity_router_config: expect.objectContaining({
          jev_classifier_config: {
            provider: "bespoke_nimble",
            model: "nimble-latest",
            timeout_ms: 20000,
            api_key: null,
          },
        }),
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Use gateway connection" }));
    fireEvent.click(screen.getByRole("button", { name: "Probe current config" }));
    expect(testAutoRouterRouting).toHaveBeenLastCalledWith(
      "token",
      expect.objectContaining({
        complexity_router_config: expect.objectContaining({
          jev_classifier_config: {
            provider: "bespoke_nimble",
            model: "nimble-latest",
            timeout_ms: 20000,
            api_key: null,
            api_base: null,
          },
        }),
      }),
    );
  });

  it.each(["Internal User", "Admin Viewer", "org_admin"])("hides connection overrides for %s", (userRole) => {
    vi.mocked(useAuthorized).mockReturnValue({ ...useAuthorized(), userRole });
    renderWithProviders(<Form />);
    fireEvent.click(screen.getByRole("radio", { name: "Decision Model" }));
    fireEvent.click(screen.getByRole("radio", { name: "Nimble (open source)" }));
    expect(screen.getByLabelText("Nimble Model")).toHaveValue("nimble-latest");
    expect(screen.queryByText("Connection settings")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("API Base")).not.toBeInTheDocument();
  });

  it("uses built-in JEV without a license and preserves custom tiers and context through reload", () => {
    renderWithProviders(<Form />);
    expect(screen.getByLabelText("Judge model")).toBeInTheDocument();
    expect(screen.getByText("Reasoning Effort")).toBeInTheDocument();
    expect(screen.getByText("Classifier Prompt")).toBeInTheDocument();
    expect(screen.getByRole("switch", { name: "Use images for classification" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Decision Model" }));
    expect(screen.getByRole("radio", { name: "Decision Model" })).toBeChecked();
    expect(screen.getByLabelText("Jev Model")).toHaveValue("jev-latest");
    expect(screen.getByLabelText("Jev Instructions")).toBeEnabled();
    expect(screen.queryByLabelText("Judge model")).not.toBeInTheDocument();
    expect(screen.queryByText("Reasoning Effort")).not.toBeInTheDocument();
    expect(screen.queryByText("Classifier Prompt")).not.toBeInTheDocument();
    expect(screen.queryByRole("switch", { name: "Use images for classification" })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Jev Model"), { target: { value: "jev-test" } });
    fireEvent.change(screen.getByLabelText("Jev Timeout (ms)"), { target: { value: "4200" } });
    fireEvent.change(screen.getByLabelText("Context Window Size"), { target: { value: "6" } });
    fireEvent.change(screen.getByLabelText("Circuit breaker cooldown (seconds)"), { target: { value: "50" } });
    fireEvent.click(screen.getByRole("switch", { name: "Classifier circuit breaker" }));
    fireEvent.click(screen.getByRole("button", { name: "Customize tiers" }));
    fireEvent.click(screen.getByRole("button", { name: "Save and reload" }));
    expect(screen.getByRole("radio", { name: "Decision Model" })).toBeChecked();
    expect(screen.getByLabelText("Jev Model")).toHaveValue("jev-test");
    expect(screen.getByLabelText("Jev Timeout (ms)")).toHaveValue(4200);
    expect(screen.getByLabelText("Context Window Size")).toHaveValue("6");
    expect(screen.getByRole("switch", { name: "Classifier circuit breaker" })).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Probe current config" }));
    expect(testAutoRouterRouting).toHaveBeenCalledWith(
      "token",
      expect.objectContaining({
        complexity_router_config: expect.objectContaining({
          classifier_type: "jev",
          jev_classifier_config: {
            model: "jev-test",
            timeout_ms: 4200,
            circuit_breaker_enabled: false,
            circuit_breaker_cooldown_seconds: 50,
          },
          tiers: expect.objectContaining({ QUICK: ["fast"] }),
        }),
      }),
    );
  });

  it("allows licensed instructions and can restore built-in instructions", () => {
    const authorized = useAuthorized();
    vi.mocked(useAuthorized).mockReturnValue({ ...authorized, premiumUser: true });
    const LicensedForm = () => {
      const [value, setValue] = useState<ComplexityRouterConfigValue>({
        ...initial,
        classifier_type: "jev",
        jev_classifier_config: { model: "jev-latest", timeout_ms: 3000, instructions: "Existing instructions" },
      });
      return <JevEditor value={value} onChange={setValue} />;
    };
    renderWithProviders(<LicensedForm />);
    expect(screen.getByLabelText("Jev Instructions")).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Jev Instructions"), { target: { value: "New instructions" } });
    expect(screen.getByLabelText("Jev Instructions")).toHaveValue("New instructions");
    fireEvent.click(screen.getByRole("button", { name: "Restore built-in Jev instructions" }));
    expect(screen.getByLabelText("Jev Instructions")).toHaveValue("");
  });
});
