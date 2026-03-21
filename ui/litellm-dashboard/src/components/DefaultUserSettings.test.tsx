import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import DefaultUserSettings, { computeSettingsDiff } from "./DefaultUserSettings";
import * as networking from "./networking";

vi.mock("./networking", () => ({
  getInternalUserSettings: vi.fn(),
  updateInternalUserSettings: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

vi.mock("./common_components/budget_duration_dropdown", () => ({
  default: ({ value, onChange }: { value: string | null; onChange: (value: string | null) => void }) => (
    <select data-testid="budget-duration" value={value || ""} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">Select duration</option>
      <option value="daily">Daily</option>
      <option value="monthly">Monthly</option>
    </select>
  ),
  getBudgetDurationLabel: (value: string) => value,
}));

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: (n: number) => String(n),
}));

vi.mock("./key_team_helpers/fetch_available_models_team_key", () => ({
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

  it("should save settings directly when no destructive changes", async () => {
    // No changes at all → direct save (no modal)
    mockGetInternalUserSettings.mockResolvedValue(mockSettings);
    mockUpdateInternalUserSettings.mockResolvedValue({
      settings: mockSettings.values,
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

    // No modal should appear
    expect(screen.queryByText("Review Changes")).not.toBeInTheDocument();
    expect(screen.getByText("Edit Settings")).toBeInTheDocument();
  });

  it("should show confirmation modal when a team is removed", async () => {
    const settingsWithTeams = {
      ...mockSettings,
      values: {
        ...mockSettings.values,
        teams: [
          { team_id: "team-alpha", max_budget_in_team: 50, user_role: "user" },
          { team_id: "team-beta", max_budget_in_team: 25, user_role: "admin" },
        ],
      },
    };
    mockGetInternalUserSettings.mockResolvedValue(settingsWithTeams);

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Edit Settings")).toBeInTheDocument();
    });

    // Enter edit mode
    act(() => {
      fireEvent.click(screen.getByText("Edit Settings"));
    });

    // Remove the first team
    const removeButtons = screen.getAllByText("Remove");
    act(() => {
      fireEvent.click(removeButtons[0]);
    });

    // Click Save Changes
    act(() => {
      fireEvent.click(screen.getByText("Save Changes"));
    });

    // Modal should appear
    await waitFor(() => {
      expect(screen.getByText("Review Changes")).toBeInTheDocument();
    });

    // Should mention the removed team
    expect(screen.getByText(/team-alpha/)).toBeInTheDocument();

    // Save should NOT have been called yet
    expect(mockUpdateInternalUserSettings).not.toHaveBeenCalled();
  });

  it("should save after confirming in the modal", async () => {
    const settingsWithTeams = {
      ...mockSettings,
      values: {
        ...mockSettings.values,
        teams: [
          { team_id: "team-alpha", max_budget_in_team: 50, user_role: "user" },
          { team_id: "team-beta", max_budget_in_team: 25, user_role: "admin" },
        ],
      },
    };
    mockGetInternalUserSettings.mockResolvedValue(settingsWithTeams);
    mockUpdateInternalUserSettings.mockResolvedValue({
      settings: {
        ...settingsWithTeams.values,
        teams: [{ team_id: "team-beta", max_budget_in_team: 25, user_role: "admin" }],
      },
    });

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Edit Settings")).toBeInTheDocument();
    });

    // Enter edit mode, remove first team, click Save
    act(() => {
      fireEvent.click(screen.getByText("Edit Settings"));
    });
    const removeButtons = screen.getAllByText("Remove");
    act(() => {
      fireEvent.click(removeButtons[0]);
    });
    act(() => {
      fireEvent.click(screen.getByText("Save Changes"));
    });

    // Wait for modal
    await waitFor(() => {
      expect(screen.getByText("Review Changes")).toBeInTheDocument();
    });

    // Click Confirm
    await act(async () => {
      fireEvent.click(screen.getByText("Confirm Changes"));
    });

    // Save should have been called
    await waitFor(() => {
      expect(mockUpdateInternalUserSettings).toHaveBeenCalledTimes(1);
    });

    // The saved teams should not include team-alpha
    const sentValues = mockUpdateInternalUserSettings.mock.calls[0][1];
    const sentTeamIds = sentValues.teams.map((t: any) => t.team_id);
    expect(sentTeamIds).not.toContain("team-alpha");
    expect(sentTeamIds).toContain("team-beta");
  });

  it("should NOT save when modal is cancelled", async () => {
    const settingsWithTeams = {
      ...mockSettings,
      values: {
        ...mockSettings.values,
        teams: [
          { team_id: "team-alpha", max_budget_in_team: 50, user_role: "user" },
        ],
      },
    };
    mockGetInternalUserSettings.mockResolvedValue(settingsWithTeams);

    render(<DefaultUserSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Edit Settings")).toBeInTheDocument();
    });

    // Enter edit mode, remove team, click Save
    act(() => {
      fireEvent.click(screen.getByText("Edit Settings"));
    });
    const removeButtons = screen.getAllByText("Remove");
    act(() => {
      fireEvent.click(removeButtons[0]);
    });
    act(() => {
      fireEvent.click(screen.getByText("Save Changes"));
    });

    // Wait for modal
    await waitFor(() => {
      expect(screen.getByText("Review Changes")).toBeInTheDocument();
    });

    // Click Cancel inside the modal (there are two Cancel buttons: one in the
    // edit header and one in the modal footer). The modal's Cancel is the last
    // one rendered.
    const cancelButtons = screen.getAllByText("Cancel");
    await act(async () => {
      fireEvent.click(cancelButtons[cancelButtons.length - 1]);
    });

    // Save should NOT have been called
    expect(mockUpdateInternalUserSettings).not.toHaveBeenCalled();

    // Should still be in edit mode (Save Changes button visible)
    expect(screen.getByText("Save Changes")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Tests: computeSettingsDiff (pure function)
// ---------------------------------------------------------------------------

describe("computeSettingsDiff", () => {
  it("returns no changes when values are identical", () => {
    const values = { max_budget: 100, models: ["gpt-4"], teams: [] };
    const { changes, hasDestructiveChanges } = computeSettingsDiff(values, { ...values });
    expect(changes).toHaveLength(0);
    expect(hasDestructiveChanges).toBe(false);
  });

  it("detects team removal as destructive", () => {
    const original = {
      teams: [
        { team_id: "team-alpha", user_role: "user" },
        { team_id: "team-beta", user_role: "admin" },
      ],
    };
    const edited = {
      teams: [{ team_id: "team-alpha", user_role: "user" }],
    };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "removed",
          details: expect.stringContaining("team-beta"),
        }),
      ]),
    );
  });

  it("detects team addition as non-destructive", () => {
    const original = { teams: [] };
    const edited = { teams: [{ team_id: "new-team", user_role: "user" }] };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(false);
    expect(changes).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "added" })]),
    );
  });

  it("detects team budget change as destructive", () => {
    const original = { teams: [{ team_id: "t1", max_budget_in_team: 100, user_role: "user" }] };
    const edited = { teams: [{ team_id: "t1", max_budget_in_team: 50, user_role: "user" }] };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes[0].type).toBe("changed");
    expect(changes[0].details).toContain("$100");
    expect(changes[0].details).toContain("$50");
  });

  it("detects model removal as destructive", () => {
    const original = { models: ["gpt-4", "gpt-3.5-turbo"] };
    const edited = { models: ["gpt-4"] };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "removed", details: expect.stringContaining("gpt-3.5-turbo") }),
      ]),
    );
  });

  it("detects model addition as non-destructive", () => {
    const original = { models: ["gpt-4"] };
    const edited = { models: ["gpt-4", "claude-3"] };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(false);
    expect(changes).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "added" })]),
    );
  });

  it("detects scalar field cleared as destructive", () => {
    const original = { max_budget: 100 };
    const edited = { max_budget: null };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes[0].type).toBe("removed");
    expect(changes[0].details).toContain("Cleared");
  });

  it("detects scalar field set as non-destructive", () => {
    const original = { max_budget: null };
    const edited = { max_budget: 200 };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(false);
    expect(changes[0].type).toBe("added");
  });

  it("detects scalar value change as destructive", () => {
    const original = { user_role: "internal_user" };
    const edited = { user_role: "internal_user_view_only" };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes[0].type).toBe("changed");
  });

  it("handles string team_ids in original", () => {
    const original = { teams: ["team-alpha", "team-beta"] };
    const edited = { teams: [{ team_id: "team-alpha", user_role: "user" }] };

    const { changes, hasDestructiveChanges } = computeSettingsDiff(original, edited);
    expect(hasDestructiveChanges).toBe(true);
    expect(changes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "removed", details: expect.stringContaining("team-beta") }),
      ]),
    );
  });

  it("treats empty string same as null for scalars", () => {
    const original = { max_budget: "" };
    const edited = { max_budget: null };

    const { changes } = computeSettingsDiff(original, edited);
    expect(changes).toHaveLength(0);
  });
});
