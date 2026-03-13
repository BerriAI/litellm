import React from "react";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailSelectionModal from "./guardrail_selection_modal";

const makeGuardrailDef = (name: string, description = "A guardrail description") => ({
  guardrail_name: name,
  guardrail_info: { description },
  litellm_params: { guardrail: "presidio", mode: "pre_call" },
});

const makeTemplate = (guardrailDefs: any[] = [], overrides: any = {}) => ({
  title: "Test Template",
  guardrailDefinitions: guardrailDefs,
  ...overrides,
});

const defaultProps = {
  visible: true,
  template: makeTemplate([makeGuardrailDef("guardrail-new-1"), makeGuardrailDef("guardrail-new-2")]),
  existingGuardrails: new Set<string>(),
  onConfirm: vi.fn(),
  onCancel: vi.fn(),
};

describe("GuardrailSelectionModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render guardrail names from the template", async () => {
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    expect(await screen.findByText("guardrail-new-1")).toBeInTheDocument();
    expect(screen.getByText("guardrail-new-2")).toBeInTheDocument();
  });

  it("should pre-select only new guardrails when the modal opens", async () => {
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    await screen.findByText("guardrail-new-1");
    const checkboxes = screen.getAllByRole("checkbox");
    checkboxes.forEach((cb) => expect(cb).toBeChecked());
  });

  it("should not show a checkbox for guardrails that already exist", async () => {
    const props = {
      ...defaultProps,
      template: makeTemplate([makeGuardrailDef("existing-g"), makeGuardrailDef("new-g")]),
      existingGuardrails: new Set(["existing-g"]),
    };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    await screen.findByText("existing-g");
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  });

  it("should show an 'Already exists' tag for guardrails that exist in the system", async () => {
    const props = {
      ...defaultProps,
      template: makeTemplate([makeGuardrailDef("existing-g")]),
      existingGuardrails: new Set(["existing-g"]),
    };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    expect(await screen.findByText("Already exists")).toBeInTheDocument();
  });

  it("should show 'Create N Guardrails & Use Template' on the confirm button when N guardrails are selected", async () => {
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    expect(await screen.findByRole("button", { name: /create 2 guardrails & use template/i })).toBeInTheDocument();
  });

  it("should show 'Use Template' on the confirm button when no new guardrails are selected", async () => {
    const props = {
      ...defaultProps,
      template: makeTemplate([makeGuardrailDef("existing-g")]),
      existingGuardrails: new Set(["existing-g"]),
    };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    expect(await screen.findByRole("button", { name: /^use template$/i })).toBeInTheDocument();
  });

  it("should call onConfirm with the definitions of selected guardrails when confirmed", async () => {
    const user = userEvent.setup();
    const def = makeGuardrailDef("my-guardrail");
    const props = { ...defaultProps, template: makeTemplate([def]) };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    await user.click(await screen.findByRole("button", { name: /create 1 guardrail/i }));
    expect(defaultProps.onConfirm).toHaveBeenCalledWith([def]);
  });

  it("should deselect all guardrails when 'Deselect All' is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    await screen.findByText("guardrail-new-1");
    await user.click(screen.getByRole("button", { name: /deselect all/i }));
    screen.getAllByRole("checkbox").forEach((cb) => expect(cb).not.toBeChecked());
  });

  it("should re-select all new guardrails when 'Select All New' is clicked after deselecting", async () => {
    const user = userEvent.setup();
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    await screen.findByText("guardrail-new-1");
    await user.click(screen.getByRole("button", { name: /deselect all/i }));
    await user.click(screen.getByRole("button", { name: /select all new/i }));
    screen.getAllByRole("checkbox").forEach((cb) => expect(cb).toBeChecked());
  });

  it("should show 'No guardrails defined' when the template has no guardrail definitions", async () => {
    const props = { ...defaultProps, template: makeTemplate([]) };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    expect(await screen.findByText(/no guardrails defined for this template/i)).toBeInTheDocument();
  });

  it("should show a progress badge when progressInfo is provided", async () => {
    const props = { ...defaultProps, progressInfo: { current: 2, total: 5 } };
    renderWithProviders(<GuardrailSelectionModal {...props} />);
    expect(await screen.findByText(/template 2 of 5/i)).toBeInTheDocument();
  });

  it("should call onCancel when the Cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<GuardrailSelectionModal {...defaultProps} />);
    await screen.findByText("guardrail-new-1");
    await user.click(screen.getByRole("button", { name: /^cancel$/i }));
    expect(defaultProps.onCancel).toHaveBeenCalled();
  });
});
