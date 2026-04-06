import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import TeamSSOSettings from "./TeamSSOSettings";
import * as networking from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

vi.mock("./networking");

vi.mock("./common_components/budget_duration_dropdown", () => {
  const BudgetDurationDropdown = ({ value, onChange }: { value: string | null; onChange: (value: string) => void }) => (
    <select
      data-testid="budget-duration-dropdown"
      value={value || ""}
      onChange={(e) => onChange(e.target.value)}
      aria-label="Budget duration"
    >
      <option value="">Select duration</option>
      <option value="24h">Daily</option>
      <option value="7d">Weekly</option>
      <option value="30d">Monthly</option>
    </select>
  );
  BudgetDurationDropdown.displayName = "BudgetDurationDropdown";
  return {
    default: BudgetDurationDropdown,
    getBudgetDurationLabel: vi.fn((value: string) => {
      const map: Record<string, string> = { "24h": "daily", "7d": "weekly", "30d": "monthly" };
      return map[value] || value;
    }),
  };
});

vi.mock("./key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: vi.fn((model: string) => model),
}));

vi.mock("./ModelSelect/ModelSelect", () => {
  const ModelSelect = ({ value, onChange }: { value: string[]; onChange: (value: string[]) => void }) => (
    <select
      data-testid="model-select"
      multiple
      value={value || []}
      onChange={(e) => {
        const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
        onChange(selectedValues);
      }}
      aria-label="Models"
    >
      <option value="gpt-4">gpt-4</option>
      <option value="claude-3">claude-3</option>
    </select>
  );
  ModelSelect.displayName = "ModelSelect";
  return { ModelSelect };
});

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const React = await import("react");

  const SelectComponent = ({
    value,
    onChange,
    mode,
    children,
    className,
  }: {
    value: any;
    onChange: (value: any) => void;
    mode?: string;
    children: React.ReactNode;
    className?: string;
  }) => {
    const isMultiple = mode === "multiple";
    const selectValue = isMultiple ? (Array.isArray(value) ? value : []) : value || "";
    return React.createElement(
      "select",
      {
        multiple: isMultiple,
        value: selectValue,
        onChange: (e: React.ChangeEvent<HTMLSelectElement>) => {
          const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
          onChange(isMultiple ? selectedValues : selectedValues[0] || undefined);
        },
        className,
        "aria-label": "Select",
        role: "listbox",
      },
      children,
    );
  };
  SelectComponent.displayName = "Select";

  const SelectOption = ({ value: optionValue, children: optionChildren }: { value: string; children: React.ReactNode }) =>
    React.createElement("option", { value: optionValue }, optionChildren);
  SelectOption.displayName = "SelectOption";
  SelectComponent.Option = SelectOption;

  const Spin = ({ size }: { size?: string }) =>
    React.createElement("div", { "data-testid": "spinner", "data-size": size });
  Spin.displayName = "Spin";

  const InputNumber = ({
    value,
    onChange,
    placeholder,
    prefix,
  }: {
    value: number | null;
    onChange: (value: number | null) => void;
    placeholder?: string;
    prefix?: string;
    min?: number;
    className?: string;
    style?: React.CSSProperties;
  }) =>
    React.createElement("input", {
      type: "number",
      value: value ?? "",
      onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
        const v = e.target.value === "" ? null : Number(e.target.value);
        onChange(v);
      },
      placeholder,
      "data-prefix": prefix,
      "aria-label": "number input",
    });
  InputNumber.displayName = "InputNumber";

  return {
    ...actual,
    Spin,
    Select: SelectComponent,
    InputNumber,
  };
});

const mockGetDefaultTeamSettings = vi.mocked(networking.getDefaultTeamSettings);
const mockUpdateDefaultTeamSettings = vi.mocked(networking.updateDefaultTeamSettings);
const mockNotificationsManager = vi.mocked(NotificationsManager);

describe("TeamSSOSettings", () => {
  const defaultProps = {
    accessToken: "test-token",
    userID: "test-user",
    userRole: "admin",
  };

  const mockSettingsResponse = {
    values: {
      max_budget: 1000,
      budget_duration: "30d",
      tpm_limit: 500,
      rpm_limit: 100,
      models: ["gpt-4"],
      team_member_permissions: ["/key/generate", "/key/update"],
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // --- Loading & Error States ---

  it("should show loading spinner while fetching settings", () => {
    mockGetDefaultTeamSettings.mockImplementation(() => new Promise(() => {}));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    expect(screen.getByTestId("spinner")).toBeInTheDocument();
  });

  it("should display error message when fetch fails", async () => {
    mockGetDefaultTeamSettings.mockRejectedValue(new Error("Fetch failed"));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(
        screen.getByText("No team settings available or you do not have permission to view them."),
      ).toBeInTheDocument();
    });
    expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to fetch team settings");
  });

  it("should not fetch settings when access token is null", async () => {
    renderWithProviders(<TeamSSOSettings accessToken={null} userID="test-user" userRole="admin" />);

    await waitFor(() => {
      expect(mockGetDefaultTeamSettings).not.toHaveBeenCalled();
    });
  });

  // --- View Mode ---

  it("should render title and subtitle", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Default Team Settings")).toBeInTheDocument();
      expect(screen.getByText("These settings will be applied by default when creating new teams.")).toBeInTheDocument();
    });
  });

  it("should render section headers", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Budget & Rate Limits")).toBeInTheDocument();
      expect(screen.getByText("Access & Permissions")).toBeInTheDocument();
    });
  });

  it("should display all field labels and descriptions", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Max Budget")).toBeInTheDocument();
      expect(screen.getByText("Budget Duration")).toBeInTheDocument();
      expect(screen.getByText("TPM Limit")).toBeInTheDocument();
      expect(screen.getByText("RPM Limit")).toBeInTheDocument();
      expect(screen.getByText("Models")).toBeInTheDocument();
      expect(screen.getByText("Team Member Permissions")).toBeInTheDocument();
    });

    // Descriptions
    expect(screen.getByText("Maximum budget (in USD) for new automatically created teams.")).toBeInTheDocument();
    expect(screen.getByText("How frequently the team's budget resets.")).toBeInTheDocument();
  });

  it("should display formatted values in view mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      // max_budget displayed with $
      expect(screen.getByText("$1,000")).toBeInTheDocument();
      // budget_duration through getBudgetDurationLabel
      expect(screen.getByText("monthly")).toBeInTheDocument();
      // tpm_limit formatted
      expect(screen.getByText("500")).toBeInTheDocument();
      // rpm_limit formatted
      expect(screen.getByText("100")).toBeInTheDocument();
    });
  });

  it("should display models as tags in view mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });
  });

  it("should display permissions as tags in view mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("/key/generate")).toBeInTheDocument();
      expect(screen.getByText("/key/update")).toBeInTheDocument();
    });
  });

  it("should display 'Not set' for null values", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue({
      values: {
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        models: [],
        team_member_permissions: [],
      },
    });

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      const notSetElements = screen.getAllByText("Not set");
      // max_budget, budget_duration, tpm_limit, rpm_limit, models (empty), permissions (empty)
      expect(notSetElements.length).toBeGreaterThanOrEqual(4);
    });
  });

  // --- Edit Mode Toggle ---

  it("should toggle to edit mode when Edit Settings is clicked", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));

    expect(screen.getByRole("button", { name: /Cancel/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save Changes/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Edit Settings/i })).not.toBeInTheDocument();
  });

  it("should cancel edit mode and reset values", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));
    await userEvent.click(screen.getByRole("button", { name: /Cancel/i }));

    expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Cancel/i })).not.toBeInTheDocument();
  });

  // --- Edit Mode Fields ---

  it("should show budget duration dropdown in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));

    await waitFor(() => {
      expect(screen.getByTestId("budget-duration-dropdown")).toBeInTheDocument();
    });
  });

  it("should show ModelSelect in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });
  });

  it("should show number inputs for budget and rate limits in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));

    await waitFor(() => {
      const numberInputs = screen.getAllByLabelText("number input");
      // max_budget, tpm_limit, rpm_limit
      expect(numberInputs.length).toBe(3);
    });
  });

  it("should show permissions multi-select in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));

    await waitFor(() => {
      const listboxes = screen.getAllByRole("listbox");
      expect(listboxes.length).toBeGreaterThan(0);
    });
  });

  // --- Save ---

  it("should save settings and show success notification", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);
    mockUpdateDefaultTeamSettings.mockResolvedValue({
      settings: mockSettingsResponse.values,
    });

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));
    await userEvent.click(screen.getByRole("button", { name: /Save Changes/i }));

    await waitFor(() => {
      expect(mockUpdateDefaultTeamSettings).toHaveBeenCalledWith("test-token", expect.any(Object));
    });

    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Default team settings updated successfully");

    // Should exit edit mode after save
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });
  });

  it("should show error notification when save fails", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);
    mockUpdateDefaultTeamSettings.mockRejectedValue(new Error("Save failed"));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));
    await userEvent.click(screen.getByRole("button", { name: /Save Changes/i }));

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to update team settings");
    });
  });

  it("should disable cancel button while saving", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettingsResponse);
    mockUpdateDefaultTeamSettings.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve({ settings: mockSettingsResponse.values }), 100)),
    );

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Edit Settings/i })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole("button", { name: /Edit Settings/i }));
    await userEvent.click(screen.getByRole("button", { name: /Save Changes/i }));

    expect(screen.getByRole("button", { name: /Cancel/i })).toBeDisabled();
  });
});
