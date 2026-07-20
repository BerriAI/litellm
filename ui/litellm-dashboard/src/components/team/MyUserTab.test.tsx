import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MyUserTab from "./MyUserTab";

vi.mock("./useMyTeamMember", () => ({
  useMyTeamMember: vi.fn(),
}));

import { useMyTeamMember } from "./useMyTeamMember";

const mockUseMyTeamMember = vi.mocked(useMyTeamMember);

describe("MyUserTab", () => {
  it("shows progress toward the per-user budget and model usage", () => {
    mockUseMyTeamMember.mockReturnValue({
      data: {
        user_id: "alice@example.com",
        team_id: "team-1",
        role: "user",
        user_email: "alice@example.com",
        spend: 40,
        total_spend: 120,
        using_team_default_budget: true,
        litellm_budget_table: {
          max_budget: 100,
          budget_duration: "30d",
          tpm_limit: null,
          rpm_limit: null,
        },
        model_max_budget_usage: {
          "claude-opus-4-8": {
            current_spend: 5,
            budget_limit: 20,
            time_period: "1d",
            scope: "team",
            percent_used: 25,
          },
        },
      },
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useMyTeamMember>);

    render(<MyUserTab teamId="team-1" />);

    expect(screen.getByText("Your Budget This Cycle")).toBeInTheDocument();
    expect(screen.getByText(/of \$100/)).toBeInTheDocument();
    expect(screen.getByText("Team default")).toBeInTheDocument();
    expect(screen.getByText(/\$60(\.00)? remaining/)).toBeInTheDocument();
    expect(screen.getByText("claude-opus-4-8")).toBeInTheDocument();
    expect(screen.getByText(/\$5(\.00)? \/ \$20(\.00)?/)).toBeInTheDocument();
  });
});
