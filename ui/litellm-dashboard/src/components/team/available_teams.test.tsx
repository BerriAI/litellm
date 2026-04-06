import * as networking from "@/components/networking";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import AvailableTeamsPanel from "./available_teams";

vi.mock("@/components/networking", () => ({
  availableTeamListCall: vi.fn(),
  teamMemberAddCall: vi.fn(),
}));

describe("AvailableTeamsPanel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("Team Name")).toBeInTheDocument();
    });
  });

  it("should display teams when available", async () => {
    const mockTeams = [
      {
        team_id: "team-1",
        team_alias: "Test Team 1",
        description: "Test Description 1",
        models: ["gpt-4"],
        members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
      },
      {
        team_id: "team-2",
        team_alias: "Test Team 2",
        description: "Test Description 2",
        models: [],
        members_with_roles: [{ user_id: "user-2", user_email: "user2@test.com", role: "user" }],
      },
    ];

    vi.mocked(networking.availableTeamListCall).mockResolvedValue(mockTeams);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("Test Team 1")).toBeInTheDocument();
      expect(screen.getByText("Test Team 2")).toBeInTheDocument();
    });
  });

  it("should display empty state when no teams are available", async () => {
    vi.mocked(networking.availableTeamListCall).mockResolvedValue([]);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText(/No available teams to join/i)).toBeInTheDocument();
      expect(screen.getByText(/See how to set available teams/i)).toBeInTheDocument();
    });
  });

  it("should call teamMemberAddCall when join team button is clicked", async () => {
    const mockTeams = [
      {
        team_id: "team-1",
        team_alias: "Test Team 1",
        description: "Test Description 1",
        models: ["gpt-4"],
        members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
      },
    ];

    vi.mocked(networking.availableTeamListCall).mockResolvedValue(mockTeams);
    vi.mocked(networking.teamMemberAddCall).mockResolvedValue({});

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("Test Team 1")).toBeInTheDocument();
    });

    const joinButtons = screen.getAllByRole("button", { name: /join team/i });
    await act(async () => {
      fireEvent.click(joinButtons[0]);
    });

    await waitFor(() => {
      expect(networking.teamMemberAddCall).toHaveBeenCalledWith("token-123", "team-1", {
        user_id: "user-123",
        role: "user",
      });
    });
  });

  it("should display All Proxy Models badge when team has no models", async () => {
    const mockTeams = [
      {
        team_id: "team-1",
        team_alias: "Test Team 1",
        description: "Test Description 1",
        models: [],
        members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
      },
    ];

    vi.mocked(networking.availableTeamListCall).mockResolvedValue(mockTeams);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    });
  });

  it("should display model badges when team has models", async () => {
    const mockTeams = [
      {
        team_id: "team-1",
        team_alias: "Test Team 1",
        description: "Test Description 1",
        models: ["gpt-4", "gpt-3.5-turbo"],
        members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
      },
    ];

    vi.mocked(networking.availableTeamListCall).mockResolvedValue(mockTeams);

    renderWithProviders(<AvailableTeamsPanel accessToken="token-123" userID="user-123" />);

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
    });
  });
});
