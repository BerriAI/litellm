import { renderWithProviders, screen } from "../../tests/test-utils";
import { describe, expect, it } from "vitest";
import GuardrailSettingsView from "./GuardrailSettingsView";

describe("GuardrailSettingsView", () => {
  it("should render", () => {
    renderWithProviders(<GuardrailSettingsView globalGuardrailNames={new Set()} />);

    expect(screen.getByText("Guardrails Settings")).toBeInTheDocument();
  });

  it("should separate active global and team-specific guardrails", () => {
    renderWithProviders(
      <GuardrailSettingsView
        globalGuardrailNames={new Set(["global-one", "global-two"])}
        teamGuardrails={["global-one", "team-one"]}
        optedOutGlobalGuardrails={["global-two"]}
      />,
    );

    expect(screen.getByText("global-one")).toBeInTheDocument();
    expect(screen.getByText("team-one")).toBeInTheDocument();
    expect(screen.queryByText("global-two")).not.toBeInTheDocument();
  });

  it("should show when global guardrails are bypassed", () => {
    renderWithProviders(<GuardrailSettingsView globalGuardrailNames={new Set(["global-one"])} killSwitchOn />);

    expect(screen.getByText("Bypassed for this team")).toBeInTheDocument();
  });
});
