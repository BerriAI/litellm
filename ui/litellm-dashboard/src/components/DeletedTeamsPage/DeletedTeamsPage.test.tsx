import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactElement } from "react";
import { vi, it, expect, beforeEach, MockedFunction, type Mock } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import DeletedTeamsPage from "./DeletedTeamsPage";
import { useDeletedTeams, DeletedTeam } from "@/app/(dashboard)/hooks/teams/useTeams";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

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

it("offers the shared page sizes, refetches with the selected one and writes deleted_teams_page_size", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  mockUseDeletedTeams.mockReturnValue({
    data: { teams: [mockDeletedTeam], total: 137 },
    isLoading: false,
  } as unknown as ReturnType<typeof useDeletedTeams>);

  renderWithProviders(<DeletedTeamsPage />, { onUrlUpdate });
  await user.click(screen.getByTestId("pagination-page-size"));

  const options = await screen.findAllByRole("option");
  expect(options.map((option) => option.textContent)).toEqual(["25", "50", "100"]);

  await user.click(screen.getByRole("option", { name: "100" }));

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(1, 100);
  await waitFor(() =>
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("deleted_teams_page_size")).toBe("100"),
  );
  expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("page_size")).toBe(false);
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

const deletedTeamsResult = (teams: DeletedTeam[], total: number) =>
  ({ data: { teams, total }, isLoading: false }) as unknown as ReturnType<typeof useDeletedTeams>;

const SORT_TEAMS: DeletedTeam[] = [
  { ...mockDeletedTeam, team_id: "team-cheap", team_alias: "cheap-team", spend: 1, deleted_at: "2024-06-01T10:00:00Z" },
  {
    ...mockDeletedTeam,
    team_id: "team-pricey",
    team_alias: "pricey-team",
    spend: 9,
    deleted_at: "2024-01-01T10:00:00Z",
  },
];

const rowAliases = (aliases: string[]) =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => aliases.find((alias) => within(row).queryByText(alias) !== null));

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

const renderKeepingUrlWrites = (ui: ReactElement, searchParams: string, onUrlUpdate: OnUrlUpdateFunction) =>
  render(
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>{ui}</QueryClientProvider>
    </NuqsTestingAdapter>,
  );

const flushUrlWrites = () => new Promise((resolve) => setTimeout(resolve, 150));

it("requests the page and page size named by the deleted_teams_ URL keys", () => {
  renderWithProviders(<DeletedTeamsPage />, {
    searchParams: { deleted_teams_page: "3", deleted_teams_page_size: "50" },
  });

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(3, 50);
});

it("ignores the unprefixed and deleted_keys_ page keys", () => {
  renderWithProviders(<DeletedTeamsPage />, { searchParams: { page: "3", deleted_keys_page: "4" } });

  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(1, 25);
});

it("writes deleted_teams_page when the next page is requested", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  mockUseDeletedTeams.mockReturnValue(deletedTeamsResult([mockDeletedTeam], 137));
  renderWithProviders(<DeletedTeamsPage />, { onUrlUpdate });

  await user.click(screen.getByTestId("pagination-next"));

  await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("deleted_teams_page")).toBe("2"));
  expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(2, 25);
});

it("keeps ?deleted_teams_page= when the deleted teams request fails", async () => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  mockUseDeletedTeams.mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: true,
  } as unknown as ReturnType<typeof useDeletedTeams>);
  renderKeepingUrlWrites(<DeletedTeamsPage />, "?deleted_teams_page=3", onUrlUpdate);

  await flushUrlWrites();

  expect(onUrlUpdate).not.toHaveBeenCalled();
  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(3, 25);
});

it.each([
  ["asc", ["cheap-team", "pricey-team"]],
  ["desc", ["pricey-team", "cheap-team"]],
])("sorts the loaded page by deleted_teams_sort_by=spend in %s order", (order, expected) => {
  mockUseDeletedTeams.mockReturnValue(deletedTeamsResult(SORT_TEAMS, 2));
  renderWithProviders(<DeletedTeamsPage />, {
    searchParams: { deleted_teams_sort_by: "spend", deleted_teams_sort_order: order },
  });

  expect(rowAliases(expected)).toEqual(expected);
});

it("falls back to deleted_at descending when deleted_teams_sort_by is not a sortable column", () => {
  mockUseDeletedTeams.mockReturnValue(deletedTeamsResult([...SORT_TEAMS].reverse(), 2));
  renderWithProviders(<DeletedTeamsPage />, { searchParams: { deleted_teams_sort_by: "team_alias" } });

  expect(rowAliases(["cheap-team", "pricey-team"])).toEqual(["cheap-team", "pricey-team"]);
});

it("writes deleted_teams_sort_by and deleted_teams_sort_order when a sorted header is clicked again", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  mockUseDeletedTeams.mockReturnValue(deletedTeamsResult(SORT_TEAMS, 2));
  renderWithProviders(<DeletedTeamsPage />, { searchParams: { deleted_teams_sort_by: "spend" }, onUrlUpdate });
  expect(rowAliases(["cheap-team", "pricey-team"])).toEqual(["pricey-team", "cheap-team"]);

  await user.click(screen.getByTestId("sort-header-spend"));

  await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("deleted_teams_sort_order")).toBe("asc"));
  expect(lastUrl(onUrlUpdate)?.get("deleted_teams_sort_by")).toBe("spend");
  expect(lastUrl(onUrlUpdate)?.has("sort_by")).toBe(false);
  expect(rowAliases(["cheap-team", "pricey-team"])).toEqual(["cheap-team", "pricey-team"]);
});

it("stays on the current page when a header is clicked, since sorting only reorders the loaded page", async () => {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  mockUseDeletedTeams.mockReturnValue(deletedTeamsResult(SORT_TEAMS, 137));
  renderWithProviders(<DeletedTeamsPage />, { searchParams: { deleted_teams_page: "3" }, onUrlUpdate });

  await user.click(screen.getByTestId("sort-header-spend"));

  await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("deleted_teams_sort_by")).toBe("spend"));
  expect(lastUrl(onUrlUpdate)?.get("deleted_teams_page")).toBe("3");
  expect(mockUseDeletedTeams).toHaveBeenLastCalledWith(3, 25);
});
