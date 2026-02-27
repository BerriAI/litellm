import { describe, expect, it } from "vitest";
import {
  getTeamInfoDefaultTab,
  getTeamInfoVisibleTabs,
  isTeamInfoTabVisible,
  TEAM_INFO_TAB_KEYS,
  TEAM_INFO_TAB_LABELS,
} from "./tabVisibilityUtils";

describe("team_info_tabs", () => {
  describe("TEAM_INFO_TAB_LABELS", () => {
    it("should have label for every tab key", () => {
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.OVERVIEW]).toBe("Overview");
      expect(TEAM_INFO_TAB_LABELS[TEAM_INFO_TAB_KEYS.MEMBERS]).toBe("Members");
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

    it("returns false for settings tab when user cannot edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.SETTINGS, false)).toBe(false);
    });

    it("returns true for settings tab when user can edit", () => {
      expect(isTeamInfoTabVisible(TEAM_INFO_TAB_KEYS.SETTINGS, true)).toBe(true);
    });
  });
});
