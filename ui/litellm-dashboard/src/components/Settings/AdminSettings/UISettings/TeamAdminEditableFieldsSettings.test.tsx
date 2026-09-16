import { beforeEach, describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import { toast } from "@/lib/toast";

import TeamAdminEditableFieldsSettings from "./TeamAdminEditableFieldsSettings";

const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockUseUpdateUISettings = vi.hoisted(() => vi.fn());

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token" }),
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings", () => ({
  useUpdateUISettings: mockUseUpdateUISettings,
}));

const TPM_LABEL = "Tokens per minute Limit (TPM)";

const mockSettings = (supported: readonly string[], enabled: readonly string[]) =>
  mockUseUISettings.mockReturnValue({
    isLoading: false,
    data: {
      field_schema: {
        properties: {
          team_admin_editable_team_fields: {
            description: "Fields a team admin may change",
            items: { type: "string", enum: supported },
          },
        },
      },
      values: { team_admin_editable_team_fields: enabled },
    },
  });

const mockSave = ({
  isPending = false,
  outcome = "success",
}: {
  isPending?: boolean;
  outcome?: "success" | "error";
}) => {
  const mutate = vi.fn((_settings: unknown, options: { onSuccess: () => void; onError: (error: Error) => void }) =>
    outcome === "success" ? options.onSuccess() : options.onError(new Error("save failed")),
  );
  mockUseUpdateUISettings.mockReturnValue({ mutate, isPending });
  return mutate;
};

const saveButton = () => screen.getByRole("button", { name: "Save" });

describe("TeamAdminEditableFieldsSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("explains that nothing can be enabled when the proxy supports no fields", () => {
    mockSettings([], []);
    mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);

    expect(screen.getByText("Team admins cannot edit team settings")).toBeInTheDocument();
    expect(screen.getByText(/does not support enabling any team settings fields/)).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save" })).not.toBeInTheDocument();
  });

  it("renders one checkbox per supported field, checked for the saved ones, with Save disabled until something changes", () => {
    mockSettings(["max_budget", "tpm_limit"], ["tpm_limit"]);
    mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);

    expect(screen.getByText("Team admin editable fields")).toBeInTheDocument();
    expect(screen.getByText("1 field enabled")).toBeInTheDocument();
    expect(screen.getByText("Fields a team admin may change")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "max_budget" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: TPM_LABEL })).toBeChecked();
    expect(saveButton()).toBeDisabled();
  });

  it("only saves a ticked field once Save is clicked", async () => {
    mockSettings(["max_budget", "tpm_limit"], ["tpm_limit"]);
    const mutate = mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: "max_budget" }));

    expect(screen.getByRole("checkbox", { name: "max_budget" })).toBeChecked();
    expect(mutate).not.toHaveBeenCalled();

    fireEvent.click(saveButton());

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Team admin editable fields updated successfully"));
    expect(mutate).toHaveBeenCalledWith(
      { team_admin_editable_team_fields: ["max_budget", "tpm_limit"] },
      expect.anything(),
    );
    expect(saveButton()).toBeDisabled();
  });

  it("saves the list without an unticked field", async () => {
    mockSettings(["max_budget", "tpm_limit"], ["max_budget", "tpm_limit"]);
    const mutate = mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));
    fireEvent.click(saveButton());

    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    expect(mutate).toHaveBeenCalledWith({ team_admin_editable_team_fields: ["max_budget"] }, expect.anything());
  });

  it("disables Save again when the draft is ticked back to the saved list", () => {
    mockSettings(["tpm_limit"], []);
    mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));

    expect(saveButton()).toBeEnabled();

    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));

    expect(screen.getByRole("checkbox", { name: TPM_LABEL })).not.toBeChecked();
    expect(saveButton()).toBeDisabled();
  });

  it("treats a saved list in another order, or with fields this proxy dropped, as the same selection", () => {
    mockSettings(["max_budget", "tpm_limit"], ["tpm_limit", "retired_field", "max_budget"]);
    mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);

    expect(screen.getByText("2 fields enabled")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));
    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));

    expect(saveButton()).toBeDisabled();
  });

  it("keeps the draft and shows the error when the save fails", async () => {
    mockSettings(["tpm_limit"], []);
    const mutate = mockSave({ outcome: "error" });

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));
    fireEvent.click(saveButton());

    await waitFor(() => expect(toast.fromError).toHaveBeenCalledTimes(1));
    expect(mutate).toHaveBeenCalledTimes(1);
    expect(toast.success).not.toHaveBeenCalled();
    expect(screen.getByRole("checkbox", { name: TPM_LABEL })).toBeChecked();
    expect(saveButton()).toBeEnabled();
  });

  it("blocks ticking and saving while a save is in flight", () => {
    mockSettings(["tpm_limit"], []);
    const mutate = mockSave({ isPending: true });

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: TPM_LABEL }));

    expect(screen.getByRole("checkbox", { name: TPM_LABEL })).not.toBeChecked();
    expect(screen.getByRole("button", { name: "Saving..." })).toBeDisabled();
    expect(mutate).not.toHaveBeenCalled();
  });
});
