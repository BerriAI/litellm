import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

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
const MAX_BUDGET_LABEL = "Max Budget (USD)";
const RAISE_MAX_BUDGET_LABEL = "Raise the team's max budget";
const RAISE_MAX_BUDGET_HELP_LABEL = "About raising the team's max budget";
const RAISE_MAX_BUDGET_TOOLTIP =
  "Lets team admins raise the budget too, capped by the organization's budget when the team has one. Only proxy admins can remove it.";
const RAISE_MAX_BUDGET_DOCS_URL =
  "https://docs.litellm.ai/docs/proxy/access_control#choosing-what-team-admins-can-edit";

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
const maxBudgetCheckbox = () => screen.getByRole("checkbox", { name: MAX_BUDGET_LABEL });
const raiseMaxBudgetCheckbox = () => screen.getByRole("checkbox", { name: RAISE_MAX_BUDGET_LABEL });

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
    expect(screen.getByRole("checkbox", { name: MAX_BUDGET_LABEL })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: TPM_LABEL })).toBeChecked();
    expect(saveButton()).toBeDisabled();
  });

  it("only saves a ticked field once Save is clicked", async () => {
    mockSettings(["max_budget", "tpm_limit"], ["tpm_limit"]);
    const mutate = mockSave({});

    renderWithProviders(<TeamAdminEditableFieldsSettings />);
    fireEvent.click(screen.getByRole("checkbox", { name: MAX_BUDGET_LABEL }));

    expect(screen.getByRole("checkbox", { name: MAX_BUDGET_LABEL })).toBeChecked();
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

  describe("raise_max_budget nested under max_budget", () => {
    const supported = ["tpm_limit", "rpm_limit", "max_budget", "raise_max_budget"];

    it("keeps the Raise checkbox disabled, and unticked when clicked, until Max Budget is ticked", () => {
      mockSettings(supported, []);
      mockSave({});

      renderWithProviders(<TeamAdminEditableFieldsSettings />);

      expect(raiseMaxBudgetCheckbox()).toHaveAttribute("aria-disabled", "true");
      fireEvent.click(raiseMaxBudgetCheckbox());
      expect(raiseMaxBudgetCheckbox()).not.toBeChecked();
      expect(saveButton()).toBeDisabled();

      fireEvent.click(maxBudgetCheckbox());

      expect(raiseMaxBudgetCheckbox()).not.toHaveAttribute("aria-disabled", "true");
      fireEvent.click(raiseMaxBudgetCheckbox());
      expect(raiseMaxBudgetCheckbox()).toBeChecked();
    });

    it("saves Max Budget together with Raise in the proxy's field order", async () => {
      mockSettings(supported, []);
      const mutate = mockSave({});

      renderWithProviders(<TeamAdminEditableFieldsSettings />);
      fireEvent.click(maxBudgetCheckbox());
      fireEvent.click(raiseMaxBudgetCheckbox());

      expect(raiseMaxBudgetCheckbox()).toBeChecked();

      fireEvent.click(saveButton());

      await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
      expect(mutate).toHaveBeenCalledWith(
        { team_admin_editable_team_fields: ["max_budget", "raise_max_budget"] },
        expect.anything(),
      );
    });

    it("drops Raise from the saved list when Max Budget is unticked", async () => {
      mockSettings(supported, ["tpm_limit", "max_budget", "raise_max_budget"]);
      const mutate = mockSave({});

      renderWithProviders(<TeamAdminEditableFieldsSettings />);

      expect(screen.getByText("3 fields enabled")).toBeInTheDocument();
      expect(raiseMaxBudgetCheckbox()).toBeChecked();

      fireEvent.click(maxBudgetCheckbox());

      expect(raiseMaxBudgetCheckbox()).not.toBeChecked();
      expect(raiseMaxBudgetCheckbox()).toHaveAttribute("aria-disabled", "true");

      fireEvent.click(saveButton());

      await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
      expect(mutate).toHaveBeenCalledWith({ team_admin_editable_team_fields: ["tpm_limit"] }, expect.anything());
    });

    it("explains what Raise allows in a tooltip on its help icon, with a link to the docs", async () => {
      const user = userEvent.setup();
      mockSettings(supported, []);
      mockSave({});

      renderWithProviders(<TeamAdminEditableFieldsSettings />);

      expect(screen.queryByText(RAISE_MAX_BUDGET_TOOLTIP)).not.toBeInTheDocument();

      await user.hover(screen.getByRole("button", { name: RAISE_MAX_BUDGET_HELP_LABEL }));

      expect(await screen.findByText(RAISE_MAX_BUDGET_TOOLTIP)).toBeInTheDocument();
      expect(screen.getByRole("link", { name: "Learn more" })).toHaveAttribute("href", RAISE_MAX_BUDGET_DOCS_URL);
    });

    it("renders nothing nested when the proxy does not support Raise", () => {
      mockSettings(["max_budget", "tpm_limit"], ["max_budget"]);
      mockSave({});

      renderWithProviders(<TeamAdminEditableFieldsSettings />);

      expect(screen.queryByRole("checkbox", { name: RAISE_MAX_BUDGET_LABEL })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: RAISE_MAX_BUDGET_HELP_LABEL })).not.toBeInTheDocument();
      expect(screen.getAllByRole("checkbox")).toHaveLength(2);
    });
  });
});
