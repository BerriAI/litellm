import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

import GuardrailTable from "./guardrail_table";
import { Guardrail, GuardrailDefinitionLocation } from "@/components/guardrails/types";

const baseProps = {
  isLoading: false,
  onDeleteClick: vi.fn(),
  onGuardrailClick: vi.fn(),
  onToggleEnabled: vi.fn(),
};

const makeGuardrail = (overrides: Partial<Guardrail> = {}): Guardrail => ({
  guardrail_id: "gr-1",
  guardrail_name: "PII Redaction",
  litellm_params: { guardrail: "presidio", mode: "pre_call", default_on: true },
  guardrail_info: null,
  created_at: "2021-01-01",
  updated_at: "2021-01-02",
  guardrail_definition_location: GuardrailDefinitionLocation.DB,
  enabled: true,
  ...overrides,
});

describe("GuardrailTable", () => {
  it("renders every column header", () => {
    render(<GuardrailTable guardrailsList={[]} {...baseProps} />);
    for (const header of [
      "Guardrail ID",
      "Name",
      "Provider",
      "Mode",
      "Default On",
      "Enabled",
      "Created At",
      "Updated At",
    ]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("renders the provider logo from the bundled guardrail logo map", () => {
    render(<GuardrailTable guardrailsList={[makeGuardrail()]} {...baseProps} />);
    const logo = screen.getByAltText("Presidio PII logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("microsoft_azure.svg"));
  });

  it("falls back to a letter avatar for an unknown provider slug", () => {
    const guardrail = makeGuardrail({
      litellm_params: { guardrail: "mystery_guard", mode: "pre_call", default_on: false },
    });
    render(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} />);
    expect(screen.getByText("mystery_guard")).toBeInTheDocument();
    expect(screen.queryByAltText("mystery_guard logo")).not.toBeInTheDocument();
    expect(screen.getByText("m")).toBeInTheDocument();
  });

  it("renders a tag-based mode object instead of crashing the table", () => {
    const guardrail = makeGuardrail({
      litellm_params: {
        guardrail: "bedrock",
        mode: { tags: { "Service-Type: internal-service": "post_call" }, default: ["pre_call", "post_call"] },
        default_on: true,
      },
    });
    render(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} />);
    expect(screen.getByText("pre_call, post_call (tag-based)")).toBeInTheDocument();
  });

  it("deletes a DB guardrail through the actions menu", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const guardrail = makeGuardrail({ guardrail_id: "gr-9", guardrail_name: "Toxicity Filter" });
    render(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("guardrail-actions-gr-9"));
    await user.click(await screen.findByTestId("guardrail-action-delete"));

    expect(onDeleteClick).toHaveBeenCalledWith("gr-9", "Toxicity Filter");
  });

  it("disables deletion for config guardrails so they cannot be removed from the dashboard", async () => {
    const user = userEvent.setup();
    const onDeleteClick = vi.fn();
    const guardrail = makeGuardrail({
      guardrail_id: "cfg-1",
      guardrail_name: "Config Guardrail",
      guardrail_definition_location: GuardrailDefinitionLocation.CONFIG,
    });
    render(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} onDeleteClick={onDeleteClick} />);

    await user.click(screen.getByTestId("guardrail-actions-cfg-1"));
    const deleteItem = await screen.findByTestId("guardrail-action-delete");

    expect(deleteItem).toHaveAttribute("data-disabled");
    expect(onDeleteClick).not.toHaveBeenCalled();
  });

  it("reflects the enabled state in the row switch", () => {
    render(
      <GuardrailTable
        guardrailsList={[
          makeGuardrail({ guardrail_id: "on", guardrail_name: "On", enabled: true }),
          makeGuardrail({ guardrail_id: "off", guardrail_name: "Off", enabled: false }),
        ]}
        {...baseProps}
      />,
    );

    expect(screen.getByRole("switch", { name: "Enable On" })).toBeChecked();
    expect(screen.getByRole("switch", { name: "Enable Off" })).not.toBeChecked();
  });

  it("toggles a DB guardrail off from the row switch", async () => {
    const user = userEvent.setup();
    const onToggleEnabled = vi.fn();
    render(
      <GuardrailTable
        guardrailsList={[makeGuardrail({ guardrail_id: "gr-2", guardrail_name: "Headroom" })]}
        {...baseProps}
        onToggleEnabled={onToggleEnabled}
      />,
    );

    await user.click(screen.getByRole("switch", { name: "Enable Headroom" }));

    expect(onToggleEnabled).toHaveBeenCalledWith("gr-2", false);
  });

  it("lets config guardrails be toggled even though they cannot be deleted", async () => {
    const user = userEvent.setup();
    const onToggleEnabled = vi.fn();
    const disabledConfigOverrides: Partial<Guardrail> = {
      guardrail_id: "cfg-2",
      guardrail_name: "Config Headroom",
      guardrail_definition_location: GuardrailDefinitionLocation.CONFIG,
      enabled: false,
    };
    const guardrail = makeGuardrail(disabledConfigOverrides);
    render(<GuardrailTable guardrailsList={[guardrail]} {...baseProps} onToggleEnabled={onToggleEnabled} />);

    const toggle = screen.getByRole("switch", { name: "Enable Config Headroom" });
    expect(toggle).toBeEnabled();
    await user.click(toggle);

    expect(onToggleEnabled).toHaveBeenCalledWith("cfg-2", true);
  });
});
