import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

import TeamAdminSettingsForm from "./TeamAdminSettingsForm";

const renderForm = (editableFields: ReadonlySet<string>, overrides: { isSaving?: boolean } = {}) => {
  const onSave = vi.fn().mockResolvedValue(undefined);
  const onCancel = vi.fn();
  renderWithProviders(
    <TeamAdminSettingsForm
      initialValues={{ tpm_limit: 1000, rpm_limit: 50, max_budget: 20 }}
      editableFields={editableFields}
      isSaving={overrides.isSaving ?? false}
      onCancel={onCancel}
      onSave={onSave}
    />,
  );
  return { onSave, onCancel };
};

describe("TeamAdminSettingsForm", () => {
  it("shows the team's current values for every field the proxy lets team admins edit", () => {
    renderForm(new Set(["tpm_limit", "rpm_limit", "max_budget"]));

    expect(screen.getByLabelText("Tokens per minute Limit (TPM)")).toHaveValue(1000);
    expect(screen.getByLabelText("Requests per minute Limit (RPM)")).toHaveValue(50);
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(20);
  });

  it("hides the fields the proxy has not enabled for team admins", () => {
    renderForm(new Set(["rpm_limit"]));

    expect(screen.getByLabelText("Requests per minute Limit (RPM)")).toBeInTheDocument();
    expect(screen.queryByLabelText("Tokens per minute Limit (TPM)")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Max Budget (USD)")).not.toBeInTheDocument();
  });

  it("saves the new TPM limit and nothing else", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "5000" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: 5000 }));
  });

  it("saves a lowered budget and a new RPM limit without resending the unchanged TPM limit", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit", "rpm_limit", "max_budget"]));

    fireEvent.change(screen.getByLabelText("Requests per minute Limit (RPM)"), { target: { value: "80" } });
    fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "12.5" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ rpm_limit: 80, max_budget: 12.5 }));
  });

  it("saves a cleared TPM limit as no limit", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: null }));
  });

  it("keeps Save disabled until the TPM limit differs from the team's", () => {
    renderForm(new Set(["tpm_limit"]));
    const tpmInput = screen.getByLabelText("Tokens per minute Limit (TPM)");
    const save = screen.getByRole("button", { name: /save changes/i });

    expect(save).toBeDisabled();
    fireEvent.change(tpmInput, { target: { value: "5000" } });
    expect(save).toBeEnabled();
    fireEvent.change(tpmInput, { target: { value: "1000" } });
    expect(save).toBeDisabled();
  });

  it("closes without saving on cancel", async () => {
    const user = userEvent.setup();
    const { onSave, onCancel } = renderForm(new Set(["tpm_limit"]));

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onSave).not.toHaveBeenCalled();
  });

  it("locks both buttons while a save is in flight", () => {
    renderForm(new Set(["tpm_limit"]), { isSaving: true });
    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "5000" } });

    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
  });
});
