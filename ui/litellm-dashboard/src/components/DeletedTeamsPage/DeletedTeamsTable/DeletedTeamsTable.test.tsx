import { screen } from "@testing-library/react";
import { vi, it, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { DeletedTeamsTable } from "./DeletedTeamsTable";
import { DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

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
});

it("should render DeletedTeamsTable component", () => {
  renderWithProviders(
    <DeletedTeamsTable teams={[mockDeletedTeam]} isLoading={false} isFetching={false} />,
  );

  expect(screen.getByText("Test Team")).toBeInTheDocument();
});

it("should display team information correctly", () => {
  renderWithProviders(
    <DeletedTeamsTable teams={[mockDeletedTeam]} isLoading={false} isFetching={false} />,
  );

  expect(screen.getByText("Test Team")).toBeInTheDocument();
  expect(screen.getByText("team-1")).toBeInTheDocument();
  expect(screen.getByText("Showing 1 team")).toBeInTheDocument();
});
