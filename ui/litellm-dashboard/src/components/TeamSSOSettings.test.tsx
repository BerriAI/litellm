import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import TeamSSOSettings from "./TeamSSOSettings";
import * as networking from "./networking";

// Mock the networking functions
vi.mock("./networking");

// Mock the budget duration dropdown
vi.mock("./common_components/budget_duration_dropdown", () => ({
  default: ({ value, onChange }: { value: string | null; onChange: (value: string) => void }) => (
    <select data-testid="budget-duration-dropdown" value={value || ""} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select duration</option>
      <option value="daily">Daily</option>
      <option value="monthly">Monthly</option>
    </select>
  ),
  getBudgetDurationLabel: vi.fn((value: string) => value),
}));

// Mock the model display name helper
vi.mock("./key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: vi.fn((model: string) => model),
}));

describe("TeamSSOSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the component", async () => {
    // Mock successful API responses
    vi.mocked(networking.getDefaultTeamSettings).mockResolvedValue({
      values: {
        budget_duration: "monthly",
        max_budget: 1000,
      },
      field_schema: {
        description: "Default team settings",
        properties: {
          budget_duration: {
            type: "string",
            description: "Budget duration",
          },
          max_budget: {
            type: "number",
            description: "Maximum budget",
          },
        },
      },
    });

    vi.mocked(networking.modelAvailableCall).mockResolvedValue({
      data: [{ id: "gpt-4" }, { id: "claude-3" }],
    });

    renderWithProviders(<TeamSSOSettings accessToken="test-token" userID="test-user" userRole="admin" />);

    const container = await screen.findByText("Default Team Settings");
    expect(container).toBeInTheDocument();
  });
});
