import { fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import MyUserTab from "./MyUserTab";
import { useMyTeamMember, useUpdateMySelfBudget } from "./useMyTeamMember";

vi.mock("./useMyTeamMember", () => ({
  useMyTeamMember: vi.fn(),
  useUpdateMySelfBudget: vi.fn(),
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

  const memberInfo = (overrides: Record<string, unknown> = {}) => ({
    user_id: "user-1",
    team_id: "team-1",
    role: "user",
    spend: 20,
    total_spend: 30,
    litellm_budget_table: { max_budget: 100 },
    ...overrides,
  });

  const mockMutation = () => {
    const mutate = vi.fn();
    vi.mocked(useUpdateMySelfBudget).mockReturnValue({
      mutate,
      isPending: false,
      isError: false,
      reset: vi.fn(),
    } as unknown as ReturnType<typeof useUpdateMySelfBudget>);
    return mutate;
  };

  it("shows the effective budget and its source badge from the API", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({
      data: memberInfo({ effective_budget: 80, self_max_budget: 80, budget_source: "self" }),
      isLoading: false,
      error: null,
    } as ReturnType<typeof useMyTeamMember>);
    mockMutation();

    renderWithProviders(<MyUserTab teamId="team-1" />);

    expect(screen.getByText("of $80.0000")).toBeInTheDocument();
    expect(screen.getByTestId("budget-source-badge")).toHaveTextContent("Set by you");
    expect(screen.getByTestId("my-limit-value")).toHaveTextContent("$80.0000");
  });

  it("saves a typed limit through the PATCH mutation", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({
      data: memberInfo({ effective_budget: 100, budget_source: "custom", self_max_budget: null }),
      isLoading: false,
      error: null,
    } as ReturnType<typeof useMyTeamMember>);
    const mutate = mockMutation();

    renderWithProviders(<MyUserTab teamId="team-1" />);
    expect(screen.getByTestId("my-limit-value")).toHaveTextContent("Not set");

    fireEvent.click(screen.getByTestId("edit-my-limit"));
    fireEvent.change(screen.getByTestId("my-limit-input"), { target: { value: "80" } });
    fireEvent.click(screen.getByTestId("save-my-limit"));

    expect(mutate).toHaveBeenCalledWith(80, expect.anything());
  });

  it("clears a set limit by sending null", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({
      data: memberInfo({ effective_budget: 80, self_max_budget: 80, budget_source: "self" }),
      isLoading: false,
      error: null,
    } as ReturnType<typeof useMyTeamMember>);
    const mutate = mockMutation();

    renderWithProviders(<MyUserTab teamId="team-1" />);
    fireEvent.click(screen.getByTestId("clear-my-limit"));

    expect(mutate).toHaveBeenCalledWith(null, expect.anything());
  });

  it("warns when the typed limit is below the current cycle spend", () => {
    vi.mocked(useMyTeamMember).mockReturnValue({
      data: memberInfo({ effective_budget: 100, spend: 50, self_max_budget: null }),
      isLoading: false,
      error: null,
    } as ReturnType<typeof useMyTeamMember>);
    mockMutation();

    renderWithProviders(<MyUserTab teamId="team-1" />);
    fireEvent.click(screen.getByTestId("edit-my-limit"));
    fireEvent.change(screen.getByTestId("my-limit-input"), { target: { value: "40" } });

    expect(screen.getByTestId("below-spend-warning")).toHaveTextContent(
      "This is below your current spend of $50.0000. New requests will be blocked until you raise or clear your limit.",
    );

    fireEvent.change(screen.getByTestId("my-limit-input"), { target: { value: "60" } });
    expect(screen.queryByTestId("below-spend-warning")).not.toBeInTheDocument();
  });
});
