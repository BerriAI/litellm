import { screen, within } from "@testing-library/react";
import { vi, it, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { DeletedTeamsTable } from "./DeletedTeamsTable";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

const makeDeletedTeam = (overrides: Partial<DeletedTeam> = {}): DeletedTeam => ({
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
  ...overrides,
});

const paginationProps = {
  pagination: { pageIndex: 0, pageSize: 25 },
  onPaginationChange: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
});

it("should display team information", () => {
  renderWithProviders(
    <DeletedTeamsTable teams={[makeDeletedTeam()]} isLoading={false} rowCount={1} {...paginationProps} />,
  );

  expect(screen.getByText("Test Team")).toBeInTheDocument();
  expect(screen.getByText("team-1")).toBeInTheDocument();
  expect(screen.getByText("org-1")).toBeInTheDocument();
});

it("should sort teams by deleted_at descending by default", () => {
  const teams = [
    makeDeletedTeam({ team_id: "team-old", team_alias: "older-team", deleted_at: "2024-01-01T10:00:00Z" }),
    makeDeletedTeam({ team_id: "team-new", team_alias: "newer-team", deleted_at: "2024-06-01T10:00:00Z" }),
  ];
  renderWithProviders(<DeletedTeamsTable teams={teams} isLoading={false} rowCount={2} {...paginationProps} />);

  const rows = screen.getAllByRole("row").slice(1);
  expect(within(rows[0]).getByText("newer-team")).toBeInTheDocument();
  expect(within(rows[1]).getByText("older-team")).toBeInTheDocument();
});

it("should show skeleton rows when loading", () => {
  renderWithProviders(<DeletedTeamsTable teams={[]} isLoading rowCount={0} {...paginationProps} />);

  expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
});

it("should show the empty state when there are no deleted teams", () => {
  renderWithProviders(<DeletedTeamsTable teams={[]} isLoading={false} rowCount={0} {...paginationProps} />);

  expect(screen.getByText("No deleted teams found")).toBeInTheDocument();
});

it("renders the shared pagination footer with the server row count", () => {
  renderWithProviders(
    <DeletedTeamsTable
      teams={[makeDeletedTeam()]}
      isLoading={false}
      rowCount={137}
      pagination={{ pageIndex: 2, pageSize: 50 }}
      onPaginationChange={vi.fn()}
    />,
  );

  expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 101-137 of 137");
  expect(screen.getByTestId("pagination-page-size")).toHaveTextContent("50");
  expect(screen.getByTestId("pagination-prev")).toBeEnabled();
  expect(screen.getByTestId("pagination-next")).toBeDisabled();
});
