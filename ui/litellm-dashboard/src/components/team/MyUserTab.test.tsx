import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import MyUserTab from "./MyUserTab";
import { useMyTeamMember } from "./useMyTeamMember";

vi.mock("./useMyTeamMember", () => ({
  useMyTeamMember: vi.fn(),
}));

describe("MyUserTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({ isLoading: true } as ReturnType<typeof useMyTeamMember>);

    renderWithProviders(<MyUserTab teamId="team-1" />);

    expect(screen.getByText("Loading your membership info…")).toBeInTheDocument();
  });

  it("should display the current member budget and model scope", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({
      data: {
        user_id: "user-1",
        user_email: "member@example.com",
        team_id: "team-1",
        role: "admin",
        spend: 12.5,
        total_spend: 30,
        litellm_budget_table: {
          max_budget: 100,
          tpm_limit: 1000,
          rpm_limit: 10,
          allowed_models: ["model-one"],
        },
      },
      isLoading: false,
      error: null,
    } as ReturnType<typeof useMyTeamMember>);

    renderWithProviders(<MyUserTab teamId="team-1" />);

    expect(screen.getByText("member@example.com")).toBeInTheDocument();
    expect(screen.getByText("model-one")).toBeInTheDocument();
    expect(screen.getByText("TPM: 1,000")).toBeInTheDocument();
  });

  describe("temporary budget increase", () => {
    const renderWithTempBudget = (temp: {
      temp_budget_increase?: number | null;
      temp_budget_expiry?: string | null;
    }) => {
      vi.mocked(useMyTeamMember).mockReturnValue({
        data: {
          user_id: "user-1",
          team_id: "team-1",
          role: "user",
          spend: 12.5,
          total_spend: 30,
          litellm_budget_table: { max_budget: 100, ...temp },
        },
        isLoading: false,
        error: null,
      } as ReturnType<typeof useMyTeamMember>);

      renderWithProviders(<MyUserTab teamId="team-1" />);
    };

    it("shows an active increase and its expiry date under the budget", () => {
      renderWithTempBudget({ temp_budget_increase: 150, temp_budget_expiry: "2099-11-01T12:00:00Z" });

      expect(screen.getByTestId("my-temp-budget")).toHaveTextContent("+ $150.00 temporary increase until Nov 1, 2099");
    });

    it("treats an expiry without a timezone as UTC, as the API returns it", () => {
      renderWithTempBudget({ temp_budget_increase: 150, temp_budget_expiry: "2099-11-01T12:00:00" });

      expect(screen.getByTestId("my-temp-budget")).toBeInTheDocument();
    });

    it("shows nothing once the increase has expired", () => {
      renderWithTempBudget({ temp_budget_increase: 150, temp_budget_expiry: "2020-01-11T00:00:00Z" });

      expect(screen.queryByTestId("my-temp-budget")).not.toBeInTheDocument();
    });

    it("shows nothing without an increase, with a 0 increase, or without an expiry", () => {
      renderWithTempBudget({ temp_budget_increase: 0, temp_budget_expiry: "2099-11-01T12:00:00Z" });
      expect(screen.queryByTestId("my-temp-budget")).not.toBeInTheDocument();
    });

    it("shows nothing when the expiry is missing", () => {
      renderWithTempBudget({ temp_budget_increase: 150, temp_budget_expiry: null });
      expect(screen.queryByTestId("my-temp-budget")).not.toBeInTheDocument();
    });
  });
});
