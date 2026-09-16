import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

import TeamAdminSettingsForm from "./TeamAdminSettingsForm";

const renderForm = (editableFields: ReadonlySet<string>, overrides: { isSaving?: boolean } = {}) => {
  const onSave = vi.fn().mockResolvedValue(undefined);
  const onCancel = vi.fn();
  renderWithProviders(
    <TeamAdminSettingsForm
      initialValues={{ tpm_limit: 1000 }}
      editableFields={editableFields}
      isSaving={overrides.isSaving ?? false}
      onCancel={onCancel}
      onSave={onSave}
    />,
  );
  return { onSave, onCancel };
};

describe("TeamAdminSettingsForm", () => {
  it("shows the team's current TPM limit when the proxy lets team admins edit it", () => {
    renderForm(new Set(["tpm_limit"]));

    expect(screen.getByLabelText("Tokens per minute Limit (TPM)")).toHaveValue(1000);
  });

  it("hides the TPM limit when the proxy has not enabled it for team admins", () => {
    renderForm(new Set(["max_budget"]));

    expect(screen.queryByLabelText("Tokens per minute Limit (TPM)")).not.toBeInTheDocument();
  });

  it("saves the new TPM limit and nothing else", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "5000" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: 5000 }));
  });

  it("saves a cleared TPM limit as no limit", async () => {
    const user = userEvent.setup();
    const { onSave } = renderForm(new Set(["tpm_limit"]));

    fireEvent.change(screen.getByLabelText("Tokens per minute Limit (TPM)"), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({ tpm_limit: null }));
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

    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /save changes/i })).toBeDisabled();
  });
});
