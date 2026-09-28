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
});
