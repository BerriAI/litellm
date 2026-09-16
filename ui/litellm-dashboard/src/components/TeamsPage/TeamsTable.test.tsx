import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, Mock, MockedFunction, vi } from "vitest";

import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { Team } from "../key_team_helpers/key_list";
import { TeamsResponse, useTeamsTable } from "@/app/(dashboard)/hooks/teams/useTeams";
import { TeamsTable } from "./TeamsTable";

// Resolve debounced values synchronously so an applied filter lands in the useTeamsTable query within the test tick.
vi.mock("@tanstack/react-pacer/debouncer", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  return {
    useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
    useDebouncedState: (initial: unknown) => {
      const [value, setValue] = React.useState(initial);
      return [value, setValue, { cancel: vi.fn(), flush: vi.fn() }];
    },
  };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token",
    userId: "test-user",
    userRole: "Admin",
    premiumUser: true,
    token: "test-token",
  })),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeamsTable: vi.fn(),
  teamsTableKeys: { all: ["teamsTable"] },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: vi.fn().mockReturnValue({
    data: [{ organization_id: "org-1", organization_alias: "Test Organization" }],
  }),
}));

const mockTeam: Team = {
  team_id: "team-1",
  team_alias: "Acme Team",
  models: ["gpt-4", "gpt-3.5-turbo", "claude-3", "claude-3-5-sonnet"],
  max_budget: 100,
  budget_duration: "1mo",
  tpm_limit: 5000,
  rpm_limit: 500,
  organization_id: "org-1",
  created_at: "2024-10-01T10:00:00Z",
  updated_at: "2024-11-01T10:00:00Z",
  keys: [],
  keys_count: 3,
  members_with_roles: [
    { user_id: "u1", user_email: "a@x.com", role: "admin" },
    { user_id: "u2", user_email: "b@x.com", role: "user" },
  ] as unknown as Team["members_with_roles"],
  spend: 42.5,
};

const mockUseTeamsTable = useTeamsTable as MockedFunction<typeof useTeamsTable>;

const teamsResult = (teams: Team[], data: Partial<TeamsResponse> = {}, extra: Record<string, unknown> = {}) =>
  ({
    data: {
      teams,
      total: teams.length,
      page: 1,
      page_size: 50,
      total_pages: 1,
      ...data,
    } as TeamsResponse,
    isPending: false,
    isFetching: false,
    isError: false,
    refetch: vi.fn(),
    ...extra,
  }) as any;

const noop = () => {};

const adminTable = (props: Partial<React.ComponentProps<typeof TeamsTable>> = {}) => (
  <TeamsTable userRole="Admin" userID="admin-1" onSelectTeam={noop} onEditTeam={noop} onDeleteTeam={noop} {...props} />
);

interface UrlOptions {
  searchParams?: Record<string, string>;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderTable = (props: Partial<React.ComponentProps<typeof TeamsTable>> = {}, urlOptions: UrlOptions = {}) =>
  renderWithProviders(adminTable(props), urlOptions);

const renderTableKeepingMountUpdates = ({ searchParams, onUrlUpdate }: UrlOptions) =>
  render(adminTable(), {
    wrapper: ({ children }: { children: ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
      </NuqsTestingAdapter>
    ),
  });

const lastUrlParams = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event.searchParams;
};

const COLUMN_STORAGE_KEY = "litellm_table_columns_teams";

const openFilters = () => fireEvent.click(screen.getByRole("button", { name: "Filters" }));
const lastOptions = () => mockUseTeamsTable.mock.calls[mockUseTeamsTable.mock.calls.length - 1][2] ?? {};

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.removeItem(COLUMN_STORAGE_KEY);
  mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam]));
});

it("renders a team row with alias, organization, and spend/budget", async () => {
  renderTable();

  await waitFor(() => {
    expect(screen.getByText("Acme Team")).toBeInTheDocument();
    expect(screen.getByText("Test Organization")).toBeInTheDocument();
    expect(screen.getByText("$42.50")).toBeInTheDocument();
    expect(screen.getByText("of $100.00")).toBeInTheDocument();
  });
});

it("renders the Resources cell with member, model, and key counts", () => {
  renderTable();

  expect(screen.getByTitle("2 members")).toBeInTheDocument();
  expect(screen.getByTitle("4 models")).toBeInTheDocument();
  expect(screen.getByTitle("3 keys")).toBeInTheDocument();
});

it("shows 'No teams found' when the list is empty", () => {
  mockUseTeamsTable.mockReturnValue(teamsResult([]));
  renderTable();
  expect(screen.getByText("No teams found")).toBeInTheDocument();
});

it("shows a loading state on initial load and hides the data", () => {
  mockUseTeamsTable.mockReturnValue(teamsResult([], {}, { data: null, isPending: true, isFetching: true }));
  renderTable();

  expect(screen.getByText("Loading teams...")).toBeInTheDocument();
  expect(screen.queryByText("Acme Team")).not.toBeInTheDocument();
});

it("replaces the previous rows with the loading state while a new search is pending", () => {
  mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], {}, { isPlaceholderData: true, isFetching: true }));
  renderTable();

  expect(screen.getByText("Loading teams...")).toBeInTheDocument();
  expect(screen.queryByText("Acme Team")).not.toBeInTheDocument();
});

describe("sort contract – only backend-sortable columns are sortable", () => {
  it("requests the default created_at descending sort on first render", () => {
    renderTable();
    expect(lastOptions()).toMatchObject({ sortBy: "created_at", sortOrder: "desc" });
  });

  it("sorts by the backend team_alias field (not the label) when the Team header is clicked", async () => {
    renderTable();
    fireEvent.click(screen.getByText("Team").closest("button") as HTMLElement);

    await waitFor(() => {
      expect(mockUseTeamsTable).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ sortBy: "team_alias" }));
    });
  });

  it("does not make Spend / Budget sortable (the backend rejects sort_by=spend)", () => {
    renderTable();
    expect(screen.getByText("Spend / Budget")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Spend / Budget" })).not.toBeInTheDocument();
    // Team and Created are the only sortable headers.
    expect(screen.getByText("Team").closest("button")).not.toBeNull();
    expect(screen.getByText("Created").closest("button")).not.toBeNull();
  });
});

describe("server-side filtering maps controls to the right query params", () => {
  it("sends no filter params when nothing is applied", () => {
    renderTable();
    expect(lastOptions()).toMatchObject({ organizationID: undefined, team_alias: undefined, teamID: undefined });
  });

  it("threads an applied Team alias filter into the query", async () => {
    renderTable();
    openFilters();

    fireEvent.change(await screen.findByPlaceholderText(/Enter team alias/), { target: { value: "acme" } });
    fireEvent.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockUseTeamsTable).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ team_alias: "acme" }));
    });
  });

  it("threads an applied Team ID filter into the query", async () => {
    renderTable();
    openFilters();

    fireEvent.change(await screen.findByPlaceholderText(/Enter team ID/), { target: { value: "team-xyz" } });
    fireEvent.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => {
      expect(mockUseTeamsTable).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ teamID: "team-xyz" }));
    });
  });

  it("threads the toolbar search into the search param", async () => {
    renderTable();
    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "platform" } });

    await waitFor(() => {
      expect(mockUseTeamsTable).toHaveBeenLastCalledWith(1, 50, expect.objectContaining({ search: "platform" }));
    });
  });

  it("opts into team id prefix matching so a partial id from a proxy error finds the team", async () => {
    renderTable();
    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "66c432fa" } });

    await waitFor(() => {
      expect(mockUseTeamsTable).toHaveBeenLastCalledWith(
        1,
        50,
        expect.objectContaining({ search: "66c432fa", searchTeamIdMatch: "prefix" }),
      );
    });
  });
});

describe("non-admin scoping", () => {
  it("scopes the list to the current user when the role is not an admin role", () => {
    renderTable({ userRole: "Internal User", userID: "user-42" });
    expect(lastOptions()).toMatchObject({ userID: "user-42" });
  });

  it("does not scope by user for the Admin role", () => {
    renderTable({ userRole: "Admin", userID: "admin-1" });
    expect(lastOptions()).toMatchObject({ userID: undefined });
  });
});

describe("row actions", () => {
  it("opens the team detail when the team cell is clicked", () => {
    const onSelectTeam = vi.fn();
    renderTable({ onSelectTeam });

    fireEvent.click(screen.getByText("Acme Team"));
    expect(onSelectTeam).toHaveBeenCalledWith(expect.objectContaining({ team_id: "team-1" }));
  });

  it("offers Edit and Delete to an Admin and wires them to the callbacks", async () => {
    const onEditTeam = vi.fn();
    const onDeleteTeam = vi.fn();
    const user = userEvent.setup();
    renderTable({ userRole: "Admin", onEditTeam, onDeleteTeam });

    await user.click(screen.getByTestId("team-actions-team-1"));

    await user.click(await screen.findByText("Edit team"));
    expect(onEditTeam).toHaveBeenCalledWith(expect.objectContaining({ team_id: "team-1" }));

    await chooseSelectOption(user, screen.getByTestId("team-actions-team-1"), "Delete team", "menuitem");
    expect(onDeleteTeam).toHaveBeenCalledWith(expect.objectContaining({ team_id: "team-1" }));
  });

  it("hides Edit and Delete from a non-admin, leaving only Copy team ID", async () => {
    const user = userEvent.setup();
    renderTable({ userRole: "Internal User" });

    await user.click(screen.getByTestId("team-actions-team-1"));

    expect(await screen.findByText("Copy team ID")).toBeInTheDocument();
    expect(screen.queryByText("Edit team")).not.toBeInTheDocument();
    expect(screen.queryByText("Delete team")).not.toBeInTheDocument();
  });
});

describe("pagination total comes from the query response", () => {
  it("shows the total count and page count from the response", async () => {
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], { total: 137, total_pages: 3 }));
    renderTable();

    await waitFor(() => {
      expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 137");
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 3");
    });
  });
});

describe("refresh control", () => {
  it("calls refetch when clicked", () => {
    const refetch = vi.fn();
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], {}, { refetch }));
    renderTable();

    fireEvent.click(screen.getByTestId("datatable-refresh"));
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it("keeps rows visible but disables refresh while a background fetch is in flight", () => {
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], {}, { isFetching: true }));
    renderTable();

    expect(screen.getByTestId("datatable-refresh")).toBeDisabled();
    expect(screen.getByText("Acme Team")).toBeInTheDocument();
  });
});

describe("column rendering details", () => {
  it("shows the organization alias when the id resolves, and the raw id when it does not", async () => {
    mockUseTeamsTable.mockReturnValue(
      teamsResult([
        { ...mockTeam, team_id: "a", organization_id: "org-1" },
        { ...mockTeam, team_id: "b", team_alias: "Orphan Team", organization_id: "org-unknown" },
      ]),
    );
    renderTable();

    await waitFor(() => {
      expect(screen.getByText("Test Organization")).toBeInTheDocument();
      expect(screen.getByText("org-unknown")).toBeInTheDocument();
    });
  });

  it("points the organization cell at the org's detail page, aliased or not", async () => {
    mockUseTeamsTable.mockReturnValue(
      teamsResult([
        { ...mockTeam, team_id: "a", organization_id: "org-1" },
        { ...mockTeam, team_id: "b", team_alias: "Orphan Team", organization_id: "org-unknown" },
      ]),
    );
    renderTable();

    expect(await screen.findByRole("link", { name: "Test Organization" })).toHaveAttribute(
      "href",
      "/ui/organizations?org=org-1",
    );
    expect(screen.getByRole("link", { name: "org-unknown" })).toHaveAttribute(
      "href",
      "/ui/organizations?org=org-unknown",
    );
  });

  it("renders an em dash for a team with no organization", () => {
    mockUseTeamsTable.mockReturnValue(teamsResult([{ ...mockTeam, organization_id: null as unknown as string }]));
    renderTable();
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /organization/i })).not.toBeInTheDocument();
  });

  it("falls back to keys.length when keys_count is absent", () => {
    mockUseTeamsTable.mockReturnValue(
      teamsResult([
        {
          ...mockTeam,
          keys_count: undefined,
          keys: [{ token: "t1" }, { token: "t2" }] as unknown as Team["keys"],
        },
      ]),
    );
    renderTable();
    expect(screen.getByTitle("2 keys")).toBeInTheDocument();
  });
});

describe("hidden-by-default columns", () => {
  it("hides Members, Models, Rate Limits, and Updated until toggled on", async () => {
    const user = userEvent.setup();
    renderTable();

    expect(screen.queryByText("Rate Limits")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Columns" }));
    const menu = await screen.findByRole("menu");
    expect(within(menu).getByText("Rate Limits")).toBeInTheDocument();
    expect(within(menu).getByText("Updated")).toBeInTheDocument();
  });

  it("remembers toggled columns across mounts", async () => {
    const user = userEvent.setup();
    const { unmount } = renderTable();

    await user.click(screen.getByRole("button", { name: "Columns" }));
    await user.click(await screen.findByTestId("view-option-rate_limits"));

    expect(await screen.findByRole("columnheader", { name: /Rate Limits/ })).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem(COLUMN_STORAGE_KEY) ?? "{}")).toMatchObject({ rate_limits: true });

    unmount();
    renderTable();
    expect(screen.getByRole("columnheader", { name: /Rate Limits/ })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: /Updated/ })).not.toBeInTheDocument();
  });
});

describe("URL state", () => {
  it("drives the query from the URL keys and ignores the generic ones", () => {
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], { total: 100 }));

    renderTable(
      {},
      {
        searchParams: {
          team_search: "platform",
          page: "2",
          page_size: "25",
          sort_by: "team_alias",
          sort_order: "asc",
          filter_org: "org-1",
          filter_alias: "acme",
          filter_team_id: "team-xyz",
          search: "ignored",
          filter_org_id: "org-ignored",
        },
      },
    );

    const expectedQuery = {
      search: "platform",
      sortBy: "team_alias",
      sortOrder: "asc",
      organizationID: "org-1",
      team_alias: "acme",
      teamID: "team-xyz",
    };
    expect(mockUseTeamsTable).toHaveBeenLastCalledWith(2, 25, expect.objectContaining(expectedQuery));
    expect(screen.getByTestId("datatable-search")).toHaveValue("platform");
    expect(screen.getByTestId("filter-chip-org_id")).toHaveTextContent("Test Organization");
  });

  it("falls back to the default sort for a column the backend cannot sort and caps page_size", () => {
    renderTable({}, { searchParams: { sort_by: "spend", sort_order: "asc", page_size: "500" } });

    expect(mockUseTeamsTable).toHaveBeenLastCalledWith(
      1,
      100,
      expect.objectContaining({ sortBy: "created_at", sortOrder: "asc" }),
    );
  });

  it("writes the search box to team_search and resets the page", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], { total: 100 }));
    renderTable({}, { searchParams: { page: "2" }, onUrlUpdate });

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "platform" } });

    await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("team_search")).toBe("platform"));
    expect(lastUrlParams(onUrlUpdate).has("page")).toBe(false);
    expect(lastUrlParams(onUrlUpdate).has("search")).toBe(false);
  });

  it("writes sort and page changes to sort_by, sort_order and page", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    mockUseTeamsTable.mockReturnValue(teamsResult([mockTeam], { total: 100 }));
    renderTable({}, { onUrlUpdate });

    await user.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("page")).toBe("2"));

    await user.click(screen.getByTestId("sort-header-team_alias"));
    await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("sort_by")).toBe("team_alias"));
    const params = lastUrlParams(onUrlUpdate);
    expect(params.get("sort_order")).toMatch(/^(asc|desc)$/);
    expect(params.has("page")).toBe(false);
  });

  it("writes applied filters to filter_org, filter_alias and filter_team_id", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderTable({}, { onUrlUpdate });

    openFilters();
    const drawer = await screen.findByTestId("filter-drawer-body");
    await chooseSelectOption(user, within(drawer).getByRole("combobox"), /Test Organization/);
    fireEvent.change(within(drawer).getByPlaceholderText(/Enter team alias/), { target: { value: " acme " } });
    fireEvent.change(within(drawer).getByPlaceholderText(/Enter team ID/), { target: { value: "team-xyz" } });
    fireEvent.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("filter_org")).toBe("org-1"));
    const params = lastUrlParams(onUrlUpdate);
    expect(params.get("filter_alias")).toBe("acme");
    expect(params.get("filter_team_id")).toBe("team-xyz");
    expect(params.has("filter_org_id")).toBe(false);
  });

  it("pulls an out-of-range page back once the list has loaded", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    mockUseTeamsTable.mockReturnValue(teamsResult([], { total: 0 }));

    renderTableKeepingMountUpdates({ searchParams: { page: "3" }, onUrlUpdate });

    await waitFor(() => expect(mockUseTeamsTable).toHaveBeenLastCalledWith(1, 50, expect.anything()));
    expect(lastUrlParams(onUrlUpdate).has("page")).toBe(false);
  });

  it("keeps the linked page while the list request is failing", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    mockUseTeamsTable.mockReturnValue(teamsResult([], {}, { data: undefined, isError: true }));

    renderTableKeepingMountUpdates({ searchParams: { page: "3" }, onUrlUpdate });

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(mockUseTeamsTable).toHaveBeenLastCalledWith(3, 50, expect.anything());
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
