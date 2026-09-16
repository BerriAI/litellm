import * as networking from "@/components/networking";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders } from "../../../tests/test-utils";
import { afterEach, describe, expect, it, Mock, vi } from "vitest";
import AvailableTeamsPanel from "./AvailableTeamsPanel";
import type { AvailableTeam } from "./AvailableTeamsTableColumns";

vi.mock("@/components/networking", () => ({
  availableTeamListCall: vi.fn(),
  teamMemberAddCall: vi.fn(),
}));

const team = (overrides: Partial<AvailableTeam> = {}): AvailableTeam => ({
  team_id: "team-1",
  team_alias: "Test Team 1",
  description: "Test Description 1",
  models: ["gpt-4"],
  members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
  ...overrides,
});

const teamWithMembers = (alias: string, memberCount: number): AvailableTeam =>
  team({
    team_id: alias,
    team_alias: alias,
    members_with_roles: Array.from({ length: memberCount }, (_, index) => ({
      user_id: `${alias}-user-${index}`,
      role: "user",
    })),
  });

const pagedTeams = Array.from({ length: 30 }, (_, index) => {
  const alias = `Team ${String(index + 1).padStart(2, "0")}`;
  return team({ team_id: alias, team_alias: alias });
});

const renderedAliases = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[0].textContent);

const lastUrlParams = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event.searchParams;
};

describe("AvailableTeamsPanel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the column headers", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([team()]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("Team Name")).toBeInTheDocument();
    });
    expect(screen.getByText("Models")).toBeInTheDocument();
  });

  it("should display teams when available", async () => {
    const mockTeams = [
      team({ team_id: "team-1", team_alias: "Test Team 1" }),
      team({ team_id: "team-2", team_alias: "Test Team 2", models: [] }),
    ];

    vi.mocked(networking.availableTeamListCall).mockResolvedValue(mockTeams);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("Test Team 1")).toBeInTheDocument();
      expect(screen.getByText("Test Team 2")).toBeInTheDocument();
    });
  });

  it("should display the empty state when no teams are available", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText(/No available teams to join/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/See how to set available teams/i)).toBeInTheDocument();
  });

  it("should call teamMemberAddCall when the Join team menu item is clicked", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([team({ team_id: "team-1" })]);
    vi.mocked(networking.teamMemberAddCall).mockResolvedValue({});

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await user.click(await screen.findByTestId("available-team-actions-team-1"));
    await user.click(await screen.findByTestId("available-team-action-join"));

    await waitFor(() => {
      expect(networking.teamMemberAddCall).toHaveBeenCalledWith("token-123", "team-1", {
        user_id: "user-123",
        role: "user",
      });
    });
  });

  it("should show the All Proxy Models badge when a team has no models", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([team({ models: [] })]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    });
  });

  it("should show model badges when a team has models", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([team({ models: ["gpt-4", "gpt-3.5-turbo"] })]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
    });
  });

  it("should resolve to the empty state without fetching when there is no access token", async () => {
    renderWithProviders(<AvailableTeamsPanel accessToken={null} userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText(/No available teams to join/i)).toBeInTheDocument();
    });
    expect(networking.availableTeamListCall).not.toHaveBeenCalled();
  });

  it("should hold the loading skeleton until the fetch settles", async () => {
    let resolveFetch: (teams: AvailableTeam[]) => void = () => {};
    const pending = new Promise<AvailableTeam[]>((resolve) => {
      resolveFetch = resolve;
    });
    vi.mocked(networking.availableTeamListCall).mockReturnValue(pending);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    expect(screen.queryByText(/No available teams to join/i)).not.toBeInTheDocument();

    await act(async () => {
      resolveFetch([]);
    });

    await waitFor(() => {
      expect(screen.getByText(/No available teams to join/i)).toBeInTheDocument();
    });
  });

  describe("URL state", () => {
    const sortableTeams = [teamWithMembers("Alpha", 1), teamWithMembers("Bravo", 3), teamWithMembers("Charlie", 2)];

    it("sorts by team name by default", async () => {
      vi.mocked(networking.availableTeamListCall).mockResolvedValue(sortableTeams);

      renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

      await screen.findByText("Alpha");
      expect(renderedAliases()).toEqual(["Alpha", "Bravo", "Charlie"]);
    });

    it("orders rows by the available_ sort params", async () => {
      vi.mocked(networking.availableTeamListCall).mockResolvedValue(sortableTeams);

      renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />, {
        searchParams: { available_sort_by: "members", available_sort_order: "desc", sort_by: "team_alias" },
      });

      await screen.findByText("Alpha");
      expect(renderedAliases()).toEqual(["Bravo", "Charlie", "Alpha"]);
    });

    it("opens the page named by available_page and ignores the list's page param", async () => {
      vi.mocked(networking.availableTeamListCall).mockResolvedValue(pagedTeams);

      renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />, {
        searchParams: { available_page: "2", page: "1" },
      });

      await screen.findByText("Team 26");
      expect(renderedAliases()).toEqual(["Team 26", "Team 27", "Team 28", "Team 29", "Team 30"]);
    });

    it("writes header sorts and page changes to the available_ params", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      vi.mocked(networking.availableTeamListCall).mockResolvedValue(pagedTeams);

      renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />, {
        searchParams: { page: "4" },
        onUrlUpdate,
      });

      await screen.findByText("Team 01");
      await user.click(screen.getByTestId("pagination-next"));
      await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("available_page")).toBe("2"));
      expect(lastUrlParams(onUrlUpdate).get("page")).toBe("4");
      expect(await screen.findByText("Team 26")).toBeInTheDocument();

      await user.click(screen.getByTestId("sort-header-members"));
      await waitFor(() => expect(lastUrlParams(onUrlUpdate).get("available_sort_by")).toBe("members"));
      expect(lastUrlParams(onUrlUpdate).has("available_page")).toBe(false);
      expect(lastUrlParams(onUrlUpdate).has("sort_by")).toBe(false);
    });
  });
});
