import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { TeamData } from "./TeamInfo";
import TeamKeysTab from "./TeamKeysTab";
import {
  getTeamInfoDefaultTab,
  getTeamInfoVisibleTabs,
  isTeamInfoTabVisible,
  TEAM_INFO_TAB_KEYS,
  TEAM_INFO_TAB_LABELS,
} from "./tabVisibilityUtils";

vi.mock("../templates/key_info_view", () => ({
  default: ({ onClose, keyData }: { onClose: () => void; keyData: any }) => (
    <div data-testid="key-info-view">
      <span>KeyInfoView: {keyData?.key_alias ?? "unknown"}</span>
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

const createMockTeamData = (overrides: Partial<TeamData> = {}): TeamData => ({
  team_id: "team-123",
  team_info: {
    team_alias: "Test Team",
    team_id: "team-123",
    organization_id: null,
    admins: [],
    members: [],
    members_with_roles: [],
    metadata: {},
    tpm_limit: null,
    rpm_limit: null,
    max_budget: null,
    budget_duration: null,
    models: [],
    blocked: false,
    spend: 0,
    max_parallel_requests: null,
    budget_reset_at: null,
    model_id: null,
    litellm_model_table: null,
    created_at: "2024-01-01T00:00:00Z",
    team_member_budget_table: null,
  },
  keys: [],
  team_memberships: [],
  ...overrides,
});

describe("team_info_tabs", () => {
  describe("TEAM_INFO_TAB_LABELS", () => {
    it("should have label for every tab key", () => {
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.OVERVIEW]).toBe("Overview");
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.MEMBERS]).toBe("Members");
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.KEYS]).toBe("Keys");
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS]).toBe("Member Permissions");
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.SETTINGS]).toBe("Settings");
    });
  });

  describe("getTeamInfoVisibleTabs", () => {
    it("returns only overview when user cannot edit team", () => {
      const tabs = getTeamInfoVisibleTabs(false);
      expect(tabs).toEqual([TEAM_INFO_TAB_KEYS.OVERVIEW]);
    });

    it("returns all tabs when user can edit team", () => {
      const tabs = getTeamInfoVisibleTabs(true);
      expect(tabs).toEqual([
        TEAM_INFO_TAB_KEYS.OVERVIEW,
        TEAM_INFO_TAB_KEYS.MEMBERS,
        TEAM_INFO_TAB_KEYS.KEYS,
        TEAM_INFO_TAB_KEYS.MEMBER_PERMISSIONS,
        TEAM_INFO_TAB_KEYS.SETTINGS,
      ]);
    });
  });

  describe("getTeamInfoDefaultTab", () => {
    it("returns overview when editTeam is false", () => {
      expect(getTeamInfoDefaultTab(false, true)).toBe(TEAM_INFO_TAB_KEYS.OVERVIEW);
      expect(getTeamInfoDefaultTab(false, false)).toBe(TEAM_INFO_TAB_KEYS.OVERVIEW);
    });

    it("returns settings when editTeam is true and user can edit", () => {
      expect(getTeamInfoDefaultTab(true, true)).toBe(TEAM_INFO_TAB_KEYS.SETTINGS);
    });

    it("returns overview when editTeam is true but user cannot edit", () => {
      expect(getTeamInfoDefaultTab(true, false)).toBe(TEAM_INFO_TAB_KEYS.OVERVIEW);
    });
  });

  describe("isTeamInfoTabVisible", () => {
    it("always returns true for overview tab", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.OVERVIEW, false)).toBe(true);
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.OVERVIEW, true)).toBe(true);
    });

    it("returns false for members tab when user cannot edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.MEMBERS, false)).toBe(false);
    });

    it("returns true for members tab when user can edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.MEMBERS, true)).toBe(true);
    });

    it("returns false for keys tab when user cannot edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.KEYS, false)).toBe(false);
    });

    it("returns true for keys tab when user can edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.KEYS, true)).toBe(true);
    });

    it("returns false for settings tab when user cannot edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.SETTINGS, false)).toBe(false);
    });

    it("returns true for settings tab when user can edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.SETTINGS, true)).toBe(true);
    });
  });

  describe("Keys tab (virtual keys table format)", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("should render virtual keys table column headers", () => {
      renderWithProviders(
        <TeamKeysTab teamData={createMockTeamData()} accessToken="test-token" />
      );

      expect(screen.getByText("Key ID")).toBeInTheDocument();
      expect(screen.getByText("Key Alias")).toBeInTheDocument();
      expect(screen.getByText("Secret Key")).toBeInTheDocument();
      expect(screen.getByText("Team Alias")).toBeInTheDocument();
      expect(screen.getByText("Team ID")).toBeInTheDocument();
      expect(screen.getByText("User Email")).toBeInTheDocument();
      expect(screen.getByText("Spend (USD)")).toBeInTheDocument();
      expect(screen.getByText("Budget (USD)")).toBeInTheDocument();
    });

    it("should show empty state when no keys", () => {
      renderWithProviders(
        <TeamKeysTab teamData={createMockTeamData()} accessToken="test-token" />
      );

      expect(screen.getByText("0 Keys")).toBeInTheDocument();
      expect(screen.getByText("No keys in this team")).toBeInTheDocument();
    });

    it("should display keys with virtual keys format (token_id as Key ID, key_alias, user_email)", () => {
      const teamData = createMockTeamData({
        keys: [
          {
            token_id: "key_123",
            key_alias: "my-key",
            user_email: "user@test.com",
            spend: 25.5,
            max_budget: 200,
          },
        ],
      });

      renderWithProviders(
        <TeamKeysTab teamData={teamData} accessToken="test-token" />
      );

      expect(screen.getByText("1 Key")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "key_123" })).toBeInTheDocument();
      expect(screen.getByText("my-key")).toBeInTheDocument();
      expect(screen.getByText("user@test.com")).toBeInTheDocument();
    });

    it("should open KeyInfoView when Key ID button is clicked", async () => {
      const user = userEvent.setup();
      const teamData = createMockTeamData({
        keys: [
          {
            token_id: "key_123",
            key_alias: "clickable-key",
            user_email: "user@test.com",
            spend: 0,
            max_budget: null,
          },
        ],
      });

      renderWithProviders(
        <TeamKeysTab teamData={teamData} accessToken="test-token" />
      );

      await user.click(screen.getByRole("button", { name: "key_123" }));

      expect(screen.getByTestId("key-info-view")).toBeInTheDocument();
      expect(screen.getByText("KeyInfoView: clickable-key")).toBeInTheDocument();
    });

    it("should return null when teamData is null", () => {
      const { container } = renderWithProviders(
        <TeamKeysTab teamData={null as any} accessToken="test-token" />
      );

      expect(container.firstChild).toBeNull();
    });
  });
});
