import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import TeamSSOSettings from "./TeamSSOSettings";
import * as networking from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

vi.mock("./networking");

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  const React = await import("react");
  const Card = ({ children }: { children: React.ReactNode }) => React.createElement("div", { "data-testid": "card" }, children);
  Card.displayName = "Card";
  const Title = ({ children }: { children: React.ReactNode }) => React.createElement("h2", {}, children);
  Title.displayName = "Title";
  const Text = ({ children }: { children: React.ReactNode }) => React.createElement("span", {}, children);
  Text.displayName = "Text";
  const Divider = () => React.createElement("hr", {});
  Divider.displayName = "Divider";
  const TextInput = ({ value, onChange, placeholder, className }: any) =>
    React.createElement("input", {
      type: "text",
      value: value || "",
      onChange,
      placeholder,
      className,
    });
  TextInput.displayName = "TextInput";
  return {
    ...actual,
    Card,
    Title,
    Text,
    Divider,
    TextInput,
  };
});

vi.mock("./common_components/budget_duration_dropdown", () => {
  const BudgetDurationDropdown = ({ value, onChange }: { value: string | null; onChange: (value: string) => void }) => (
    <select
      data-testid="budget-duration-dropdown"
      value={value || ""}
      onChange={(e) => onChange(e.target.value)}
      aria-label="Budget duration"
    >
      <option value="">Select duration</option>
      <option value="daily">Daily</option>
      <option value="monthly">Monthly</option>
    </select>
  );
  BudgetDurationDropdown.displayName = "BudgetDurationDropdown";
  return {
    default: BudgetDurationDropdown,
    getBudgetDurationLabel: vi.fn((value: string) => `Budget: ${value}`),
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
  const Spin = ({ size }: { size?: string }) => React.createElement("div", { "data-testid": "spinner", "data-size": size });
  Spin.displayName = "Spin";
  const Switch = ({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) =>
    React.createElement("input", {
      type: "checkbox",
      role: "switch",
      checked: checked,
      onChange: (e) => onChange(e.target.checked),
      "aria-label": "Toggle switch",
    });
  Switch.displayName = "Switch";
  const Paragraph = ({ children }: { children: React.ReactNode }) => React.createElement("p", {}, children);
  Paragraph.displayName = "Paragraph";
  return {
    ...actual,
    Spin,
    Switch,
    Select: SelectComponent,
    Typography: {
      Paragraph,
    },
  };
});

const mockGetDefaultTeamSettings = vi.mocked(networking.getDefaultTeamSettings);
const mockUpdateDefaultTeamSettings = vi.mocked(networking.updateDefaultTeamSettings);
const mockModelAvailableCall = vi.mocked(networking.modelAvailableCall);
const mockNotificationsManager = vi.mocked(NotificationsManager);

describe("TeamSSOSettings", () => {
  const defaultProps = {
    accessToken: "test-token",
    userID: "test-user",
    userRole: "admin",
  };

  const mockSettings = {
    values: {
      budget_duration: "monthly",
      max_budget: 1000,
      enabled: true,
      allowed_models: ["gpt-4", "claude-3"],
      models: ["gpt-4"],
      status: "active",
    },
    field_schema: {
      description: "Default team settings schema",
      properties: {
        budget_duration: {
          type: "string",
          description: "Budget duration setting",
        },
        max_budget: {
          type: "number",
          description: "Maximum budget amount",
        },
        enabled: {
          type: "boolean",
          description: "Enable feature",
        },
        allowed_models: {
          type: "array",
          items: {
            enum: ["gpt-4", "claude-3", "gpt-3.5-turbo"],
          },
          description: "Allowed models",
        },
        models: {
          type: "array",
          description: "Selected models",
        },
        status: {
          type: "string",
          enum: ["active", "inactive", "pending"],
          description: "Status",
        },
      },
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockModelAvailableCall.mockResolvedValue({
      data: [{ id: "gpt-4" }, { id: "claude-3" }],
    });
  });

  it("should render", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Default Team Settings")).toBeInTheDocument();
    });
  });

  it("should show loading spinner while fetching settings", () => {
    mockGetDefaultTeamSettings.mockImplementation(() => new Promise(() => { }));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    expect(screen.getByTestId("spinner")).toBeInTheDocument();
  });

  it("should display message when no settings are available", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(null as any);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(
        screen.getByText("No team settings available or you do not have permission to view them."),
      ).toBeInTheDocument();
    });
  });

  it("should not fetch settings when access token is null", async () => {
    renderWithProviders(<TeamSSOSettings accessToken={null} userID="test-user" userRole="admin" />);

    await waitFor(() => {
      expect(mockGetDefaultTeamSettings).not.toHaveBeenCalled();
    });
  });

  it("should display settings fields with correct values", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Budget Duration")).toBeInTheDocument();
      expect(screen.getByText("Max Budget")).toBeInTheDocument();
    });

    expect(screen.getByText("Budget: monthly")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    const enabledTexts = screen.getAllByText("Enabled");
    expect(enabledTexts.length).toBeGreaterThan(0);
  });

  it("should display 'Not set' for null values", async () => {
    const settingsWithNulls = {
      ...mockSettings,
      values: {
        ...mockSettings.values,
        max_budget: null,
      },
    };
    mockGetDefaultTeamSettings.mockResolvedValue(settingsWithNulls);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Not set")).toBeInTheDocument();
    });
  });

  it("should toggle edit mode when edit button is clicked", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
  });

  it("should cancel edit mode and reset values", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await userEvent.click(cancelButton);

    expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
  });

  it("should save settings when save button is clicked", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);
    mockUpdateDefaultTeamSettings.mockResolvedValue({
      settings: mockSettings.values,
    });

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: "Save Changes" });
    await userEvent.click(saveButton);

    await waitFor(() => {
      expect(mockUpdateDefaultTeamSettings).toHaveBeenCalledWith("test-token", mockSettings.values);
    });

    expect(mockNotificationsManager.success).toHaveBeenCalledWith("Default team settings updated successfully");
  });

  it("should show error notification when save fails", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);
    mockUpdateDefaultTeamSettings.mockRejectedValue(new Error("Save failed"));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: "Save Changes" });
    await userEvent.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to update team settings");
    });
  });

  it("should render boolean field as switch in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      const switchElement = screen.getByRole("switch");
      expect(switchElement).toBeInTheDocument();
      expect(switchElement).toBeChecked();
    });
  });

  it("should update boolean value when switch is toggled", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("switch")).toBeInTheDocument();
    });

    const switchElement = screen.getByRole("switch");
    await userEvent.click(switchElement);

    expect(switchElement).not.toBeChecked();
  });

  it("should render budget duration dropdown in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByLabelText("Budget duration")).toBeInTheDocument();
    });
  });

  it("should update budget duration when dropdown value changes", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByLabelText("Budget duration")).toBeInTheDocument();
    });

    const dropdown = screen.getByLabelText("Budget duration");
    await userEvent.selectOptions(dropdown, "daily");

    expect(dropdown).toHaveValue("daily");
  });

  it("should render text input for string fields in edit mode", async () => {
    const settingsWithString = {
      ...mockSettings,
      field_schema: {
        ...mockSettings.field_schema,
        properties: {
          ...mockSettings.field_schema.properties,
          team_name: {
            type: "string",
            description: "Team name",
          },
        },
      },
      values: {
        ...mockSettings.values,
        team_name: "Test Team",
      },
    };
    mockGetDefaultTeamSettings.mockResolvedValue(settingsWithString);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      const textInput = screen.getByDisplayValue("Test Team");
      expect(textInput).toBeInTheDocument();
    });
  });

  it("should render enum select for string enum fields in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      const statusSelect = screen.getAllByRole("listbox")[0];
      expect(statusSelect).toBeInTheDocument();
    });
  });

  it("should render multi-select for array enum fields in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      const multiSelects = screen.getAllByRole("listbox");
      expect(multiSelects.length).toBeGreaterThan(0);
    });
  });

  it("should render ModelSelect for models field in edit mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByTestId("model-select")).toBeInTheDocument();
    });
  });

  it("should display models as badges in view mode", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      const gpt4Elements = screen.getAllByText("gpt-4");
      expect(gpt4Elements.length).toBeGreaterThan(0);
    });
  });

  it("should display 'None' for empty arrays in view mode", async () => {
    const settingsWithEmptyArray = {
      ...mockSettings,
      values: {
        ...mockSettings.values,
        models: [],
      },
    };
    mockGetDefaultTeamSettings.mockResolvedValue(settingsWithEmptyArray);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      const noneTexts = screen.getAllByText("None");
      expect(noneTexts.length).toBeGreaterThan(0);
    });
  });

  it("should display schema description when available", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Default team settings schema")).toBeInTheDocument();
    });
  });

  it("should show error notification when fetching settings fails", async () => {
    mockGetDefaultTeamSettings.mockRejectedValue(new Error("Fetch failed"));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to fetch team settings");
    });
  });

  it("should handle model fetch error gracefully", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);
    mockModelAvailableCall.mockRejectedValue(new Error("Model fetch failed"));

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Default Team Settings")).toBeInTheDocument();
    });
  });

  it("should disable cancel button while saving", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);
    mockUpdateDefaultTeamSettings.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve({ settings: mockSettings.values }), 100)),
    );

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    });

    const editButton = screen.getByRole("button", { name: "Edit Settings" });
    await userEvent.click(editButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save Changes" })).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: "Save Changes" });
    await userEvent.click(saveButton);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should display field descriptions", async () => {
    mockGetDefaultTeamSettings.mockResolvedValue(mockSettings);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Budget duration setting")).toBeInTheDocument();
      expect(screen.getByText("Maximum budget amount")).toBeInTheDocument();
    });
  });

  it("should format field names by replacing underscores and capitalizing", async () => {
    const settingsWithUnderscores = {
      ...mockSettings,
      field_schema: {
        ...mockSettings.field_schema,
        properties: {
          ...mockSettings.field_schema.properties,
          max_budget_per_user: {
            type: "number",
            description: "Max budget per user",
          },
        },
      },
      values: {
        ...mockSettings.values,
        max_budget_per_user: 500,
      },
    };
    mockGetDefaultTeamSettings.mockResolvedValue(settingsWithUnderscores);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Max Budget Per User")).toBeInTheDocument();
    });
  });

  it("should display 'No schema information available' when schema is missing", async () => {
    const settingsWithoutSchema = {
      values: {},
      field_schema: null,
    };
    mockGetDefaultTeamSettings.mockResolvedValue(settingsWithoutSchema);

    renderWithProviders(<TeamSSOSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("No schema information available")).toBeInTheDocument();
    });
  });
});
