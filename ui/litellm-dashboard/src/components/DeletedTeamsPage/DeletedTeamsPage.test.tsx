import { screen } from "@testing-library/react";
import { vi, it, expect, beforeEach, MockedFunction } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import DeletedTeamsPage from "./DeletedTeamsPage";
import { useDeletedTeams, DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useDeletedTeams: vi.fn(),
}));

const mockUseDeletedTeams = useDeletedTeams as MockedFunction<typeof useDeletedTeams>;

const mockDeletedTeam: DeletedTeam = {
  team_id: "team-1",
  team_alias: "Test Team",
  models: ["gpt-3.5-turbo", "gpt-4"],
  max_budget: 500,
  budget_duration: "1m",
  tpm_limit: 5000,
  rpm_limit: 500,
  organization_id: "org-1",
  created_at: "2024-10-01T10:00:00Z",
  keys: [],
  members_with_roles: [],
  deleted_at: "2024-11-15T10:00:00Z",
  deleted_by: "user-1",
  spend: 100.5,
};

beforeEach(() => {
  vi.clearAllMocks();

  mockUseDeletedTeams.mockReturnValue({
    data: [mockDeletedTeam],
    isPending: false,
    isFetching: false,
  } as any);
});

it("should render DeletedTeamsPage component", () => {
  renderWithProviders(<DeletedTeamsPage />);

  expect(screen.getByText("Test Team")).toBeInTheDocument();
});

it("should handle loading state", () => {
  mockUseDeletedTeams.mockReturnValue({
    data: undefined,
    isPending: true,
    isFetching: false,
  } as any);

  renderWithProviders(<DeletedTeamsPage />);

  expect(screen.getByText("🚅 Loading teams...")).toBeInTheDocument();
});
