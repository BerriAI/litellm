import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DefaultUserSettings from "./DefaultUserSettings";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getInternalUserSettings: vi.fn(),
  updateInternalUserSettings: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("@/components/common_components/budget_duration_dropdown", () => ({
  default: ({ value, onChange }: { value: string | null; onChange: (value: string | null) => void }) => (
    <select data-testid="budget-duration" value={value || ""} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">Select duration</option>
      <option value="daily">Daily</option>
      <option value="monthly">Monthly</option>
    </select>
  ),
  getBudgetDurationLabel: (value: string) => value,
}));

vi.mock("@/components/key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: (model: string) => model,
}));

describe("DefaultUserSettings", () => {
  const mockGetInternalUserSettings = vi.mocked(networking.getInternalUserSettings);
  const mockUpdateInternalUserSettings = vi.mocked(networking.updateInternalUserSettings);
  const mockModelAvailableCall = vi.mocked(networking.modelAvailableCall);

  const defaultProps = {
    accessToken: "test-token",
    userID: "user-123",
    userRole: "Admin",
    possibleUIRoles: {
      internal_user_admin: {
        ui_label: "Admin",
        description: "Full access",
      },
      internal_user_viewer: {
        ui_label: "Viewer",
        description: "Read-only access",
      },
    },
  };

  const mockSettings = {
    values: {
      user_role: "internal_user_admin",
      budget_duration: "monthly",
      max_budget: 1000,
      teams: [],
    },
    field_schema: {
      description: "Default user settings",
      properties: {
        user_role: {
          type: "string",
          description: "User role",
        },
        budget_duration: {
          type: "string",
          description: "Budget duration",
        },
        max_budget: {
          type: "number",
          description: "Maximum budget",
        },
        teams: {
          type: "array",
          description: "Teams",
        },
      },
    },
  };

  beforeEach(() => {
    mockGetInternalUserSettings.mockClear();
    mockUpdateInternalUserSettings.mockClear();
    mockModelAvailableCall.mockClear();
    mockModelAvailableCall.mockResolvedValue({
      data: [{ id: "gpt-4" }, { id: "gpt-3.5-turbo" }],
    });
  });

  it("should render", async () => {
    mockGetInternalUserSettings.mockResolvedValue(mockSettings);

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(mockGetInternalUserSettings).toHaveBeenCalled();
    });

    expect(screen.getByText("Default User Settings")).toBeInTheDocument();
  });

  it("should toggle edit mode when edit button is clicked", async () => {
    mockGetInternalUserSettings.mockResolvedValue(mockSettings);

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Edit Settings")).toBeInTheDocument();
    });

    const editButton = screen.getByText("Edit Settings");
    act(() => {
      fireEvent.click(editButton);
    });

    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getByText("Save Changes")).toBeInTheDocument();
    expect(screen.queryByText("Edit Settings")).not.toBeInTheDocument();
  });

  it("should save settings when save button is clicked", async () => {
    mockGetInternalUserSettings.mockResolvedValue(mockSettings);
    mockUpdateInternalUserSettings.mockResolvedValue({
      settings: {
        ...mockSettings.values,
        max_budget: 2000,
      },
    });

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Edit Settings")).toBeInTheDocument();
    });

    const editButton = screen.getByText("Edit Settings");
    act(() => {
      fireEvent.click(editButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Save Changes")).toBeInTheDocument();
    });

    const saveButton = screen.getByText("Save Changes");
    act(() => {
      fireEvent.click(saveButton);
    });

    await waitFor(() => {
      expect(mockUpdateInternalUserSettings).toHaveBeenCalled();
    });

    expect(screen.getByText("Edit Settings")).toBeInTheDocument();
  });

  describe("editable fields", () => {
    const teamsOnlySettings = {
      values: {
        teams: [],
      },
      field_schema: {
        description: "Default user settings",
        properties: {
          teams: {
            type: "array",
            description: "Teams",
          },
        },
      },
    };

    const enterEditMode = async () => {
      mockGetInternalUserSettings.mockResolvedValue(teamsOnlySettings);
      mockUpdateInternalUserSettings.mockResolvedValue({ settings: {} });

      render(<DefaultUserSettings {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Edit Settings")).toBeInTheDocument();
      });

      act(() => {
        fireEvent.click(screen.getByText("Edit Settings"));
      });
    };

    const savedPayload = () => mockUpdateInternalUserSettings.mock.calls[0][1] as Record<string, unknown>;

    const selectUserRole = async (optionText: string) => {
      act(() => {
        fireEvent.mouseDown(document.querySelector(".ant-select-selector")!);
      });

      await waitFor(() => {
        expect(document.querySelectorAll(".ant-select-item-option").length).toBeGreaterThan(0);
      });

      const option = Array.from(document.querySelectorAll(".ant-select-item-option")).find((el) =>
        el.textContent?.includes(optionText),
      );
      expect(option).toBeTruthy();

      act(() => {
        fireEvent.click(option!);
      });
    };

    it("keeps the stored role and max budget of a team saved without an ID", async () => {
      mockGetInternalUserSettings.mockResolvedValue({
        ...teamsOnlySettings,
        values: { teams: [{ team_id: "", max_budget_in_team: 25, user_role: "admin" }] },
      });
      mockUpdateInternalUserSettings.mockResolvedValue({ settings: {} });

      render(<DefaultUserSettings {...defaultProps} />);

      await waitFor(() => {
        expect(screen.getByText("Edit Settings")).toBeInTheDocument();
      });
      act(() => {
        fireEvent.click(screen.getByText("Edit Settings"));
      });

      expect(screen.getByPlaceholderText("Enter team ID")).toHaveValue("");
      expect(screen.getByPlaceholderText("Optional")).toHaveValue("25.00");
      expect(document.querySelector(".ant-select-selection-item")).toHaveTextContent("Admin");
    });

    it("keeps showing the selected role of a team whose ID has not been typed yet", async () => {
      await enterEditMode();

      act(() => {
        fireEvent.click(screen.getByText("Add Team"));
      });
      await selectUserRole("Admin");

      expect(document.querySelector(".ant-select-selection-item")).toHaveTextContent("Admin");
    });

    it("keeps the role and max budget of a team whose ID has not been typed yet", async () => {
      await enterEditMode();

      act(() => {
        fireEvent.click(screen.getByText("Add Team"));
      });

      act(() => {
        fireEvent.change(screen.getByPlaceholderText("Optional"), { target: { value: "25" } });
      });
      await selectUserRole("Admin");

      act(() => {
        fireEvent.click(screen.getByText("Save Changes"));
      });

      await waitFor(() => {
        expect(mockUpdateInternalUserSettings).toHaveBeenCalled();
      });
      expect(savedPayload().teams).toEqual([{ team_id: "", max_budget_in_team: 25, user_role: "admin" }]);
    });
  });
});
