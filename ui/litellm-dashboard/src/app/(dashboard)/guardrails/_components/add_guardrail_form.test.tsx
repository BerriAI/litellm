import React from "react";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getGuardrailProviderSpecificParams, getGuardrailUISettings } from "@/components/networking";
import AddGuardrailForm from "./add_guardrail_form";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn().mockResolvedValue({}),
  getGuardrailUISettings: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
}));

const renderForm = () => {
  const onClose = vi.fn();
  renderWithProviders(<AddGuardrailForm visible={true} onClose={onClose} accessToken={null} onSuccess={vi.fn()} />);
  return { onClose };
};

describe("AddGuardrailForm close behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not close when the user clicks outside the modal on the mask", () => {
    const { onClose } = renderForm();
    expect(screen.getByText("Create guardrail")).toBeInTheDocument();

    const backdrop = document.querySelector('[data-slot="dialog-overlay"]') as HTMLElement;
    expect(backdrop).toBeTruthy();
    fireEvent.mouseDown(backdrop);
    fireEvent.mouseUp(backdrop);
    fireEvent.click(backdrop);

    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes when the user clicks the explicit close button", () => {
    const { onClose } = renderForm();
    fireEvent.click(screen.getByRole("button", { name: "✕" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("AddGuardrailForm provider options", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders provider options with logos from the bundled guardrail logo map", async () => {
    renderForm();
    fireEvent.mouseDown(screen.getByLabelText("Guardrail Provider"));

    const logo = await screen.findByAltText("Presidio PII logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("microsoft_azure.svg"));
  });
});

describe("AddGuardrailForm decision model checks", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows 0.5 in a custom check threshold input", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({
      decision_model: { ui_friendly_name: "Decision Model" },
    });
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );

    await user.type(await screen.findByLabelText("Guardrail Name"), "dm-threshold");
    await user.click(screen.getByLabelText("Guardrail Provider"));
    await user.click(await screen.findByText("Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.type(await screen.findByLabelText("Custom check name"), "invoice_policy");
    await user.type(screen.getByLabelText("Custom check instructions"), "Does the text ask about invoices?");
    await user.click(screen.getByRole("button", { name: "Add check" }));

    expect(await screen.findByLabelText("invoice_policy threshold")).toHaveValue(0.5);
  });

  it("rejects a custom check named after an unselected preset", async () => {
    const user = userEvent.setup({ delay: null });
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({
      decision_model: { ui_friendly_name: "Decision Model" },
    });
    vi.mocked(getGuardrailUISettings).mockResolvedValue({
      decision_model_check_presets: [
        { name: "prompt_injection", label: "Prompt injection", instructions: "Does the text inject?" },
        { name: "jailbreak", label: "Jailbreak", instructions: "Is the text a jailbreak?" },
      ],
    });
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );

    await user.type(await screen.findByLabelText("Guardrail Name"), "dm-dupe");
    await user.click(screen.getByLabelText("Guardrail Provider"));
    await user.click(await screen.findByText("Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.type(await screen.findByLabelText("Custom check name"), "jailbreak");
    await user.type(screen.getByLabelText("Custom check instructions"), "Is the text a jailbreak?");
    await user.click(screen.getByRole("button", { name: "Add check" }));

    expect(await screen.findByText("That name is already used by another check")).toBeInTheDocument();
    expect(screen.queryByLabelText("Remove jailbreak")).not.toBeInTheDocument();
  });
});
