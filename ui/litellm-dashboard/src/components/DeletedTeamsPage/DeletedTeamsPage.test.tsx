import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
    data: { teams: [mockDeletedTeam], total: 1 },
    isLoading: false,
  } as unknown as ReturnType<typeof useDeletedTeams>);
});

it("should render DeletedTeamsPage component", () => {
  renderWithProviders(<DeletedTeamsPage />);

  expect(screen.getByText("Test Team")).toBeInTheDocument();
});

it("requests the first page of 25 deleted teams and shows the server total in the footer", () => {
  mockUseDeletedTeams.mockReturnValue({
    data: { teams: [mockDeletedTeam], total: 137 },
    isLoading: false,
  } as unknown as ReturnType<typeof useDeletedTeams>);

  renderWithProviders(<DeletedTeamsPage />);

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(1, 25);
  expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-25 of 137");
  expect(screen.getByTestId("pagination-next")).toBeEnabled();
});

it("requests the next page from the server when Next is clicked", () => {
  mockUseDeletedTeams.mockReturnValue({
    data: { teams: [mockDeletedTeam], total: 137 },
    isLoading: false,
  } as unknown as ReturnType<typeof useDeletedTeams>);

  renderWithProviders(<DeletedTeamsPage />);
  fireEvent.click(screen.getByTestId("pagination-next"));

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(2, 25);
});

it("offers the shared page sizes and refetches with the selected one", async () => {
  const user = userEvent.setup();
  mockUseDeletedTeams.mockReturnValue({
    data: { teams: [mockDeletedTeam], total: 137 },
    isLoading: false,
  } as unknown as ReturnType<typeof useDeletedTeams>);

  renderWithProviders(<DeletedTeamsPage />);
  await user.click(screen.getByTestId("pagination-page-size"));

  const options = await screen.findAllByRole("option");
  expect(options.map((option) => option.textContent)).toEqual(["25", "50", "100"]);

  await user.click(screen.getByRole("option", { name: "100" }));

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(1, 100);
});

it("should show the enterprise notice for a non-premium user", () => {
  renderWithProviders(<DeletedTeamsPage />);

  expect(screen.getByText("Coming soon to Enterprise")).toBeInTheDocument();
  expect(
    screen.getByText("Deleted team auditing is graduating from beta into our Enterprise audit & compliance suite."),
  ).toBeInTheDocument();
});

it("should show skeleton rows while the initial load is pending", () => {
  mockUseDeletedTeams.mockReturnValue({
    data: undefined,
    isLoading: true,
  } as unknown as ReturnType<typeof useDeletedTeams>);

  renderWithProviders(<DeletedTeamsPage />);

  expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
});
