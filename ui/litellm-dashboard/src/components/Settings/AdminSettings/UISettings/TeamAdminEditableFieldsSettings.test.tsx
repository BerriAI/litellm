import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { renderWithProviders, screen } from "@/../tests/test-utils";

import TeamAdminEditableFieldsSettings from "./TeamAdminEditableFieldsSettings";

describe("TeamAdminEditableFieldsSettings", () => {
  it("explains that nothing can be enabled when the proxy supports no fields", () => {
    renderWithProviders(
      <TeamAdminEditableFieldsSettings
        editableFields={[]}
        supportedFields={[]}
        isUpdating={false}
        onUpdate={vi.fn()}
      />,
    );

    expect(screen.getByText("Team admins cannot edit team settings")).toBeInTheDocument();
    expect(screen.getByText(/does not support enabling any team settings fields/)).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("renders one checkbox per supported field, checked for the enabled ones", () => {
    renderWithProviders(
      <TeamAdminEditableFieldsSettings
        editableFields={["tpm_limit"]}
        supportedFields={["max_budget", "tpm_limit"]}
        description="Fields a team admin may change"
        isUpdating={false}
        onUpdate={vi.fn()}
      />,
    );

    expect(screen.getByText("1 field enabled")).toBeInTheDocument();
    expect(screen.getByText("Fields a team admin may change")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "max_budget" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "tpm_limit" })).toBeChecked();
  });

  it("saves the list with the field added when an unchecked field is ticked", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <TeamAdminEditableFieldsSettings
        editableFields={["tpm_limit"]}
        supportedFields={["max_budget", "tpm_limit"]}
        isUpdating={false}
        onUpdate={onUpdate}
      />,
    );

    await user.click(screen.getByRole("checkbox", { name: "max_budget" }));

    expect(onUpdate).toHaveBeenCalledWith({ team_admin_editable_team_fields: ["tpm_limit", "max_budget"] });
  });

  it("saves the list with the field removed when a checked field is unticked", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <TeamAdminEditableFieldsSettings
        editableFields={["max_budget", "tpm_limit"]}
        supportedFields={["max_budget", "tpm_limit"]}
        isUpdating={false}
        onUpdate={onUpdate}
      />,
    );

    await user.click(screen.getByRole("checkbox", { name: "tpm_limit" }));

    expect(onUpdate).toHaveBeenCalledWith({ team_admin_editable_team_fields: ["max_budget"] });
  });

  it("blocks toggling while a save is in flight", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <TeamAdminEditableFieldsSettings
        editableFields={[]}
        supportedFields={["tpm_limit"]}
        isUpdating={true}
        onUpdate={onUpdate}
      />,
    );

    await user.click(screen.getByRole("checkbox", { name: "tpm_limit" }));

    expect(onUpdate).not.toHaveBeenCalled();
  });
});
