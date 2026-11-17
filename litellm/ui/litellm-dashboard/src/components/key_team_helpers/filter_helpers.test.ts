import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchAllTeams, fetchTeamFilterOptions } from "./filter_helpers";

const mockKeyListCall = vi.fn();
const mockTeamListCall = vi.fn();

vi.mock("@/components/networking", () => ({
  keyListCall: (...args: unknown[]) => mockKeyListCall(...args),
  teamListCall: (...args: unknown[]) => mockTeamListCall(...args),
  organizationListCall: vi.fn(),
}));

describe("fetchTeamFilterOptions", () => {
  it("should return empty arrays when accessToken is null", async () => {
    const result = await fetchTeamFilterOptions(null, "team-1");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
    expect(mockKeyListCall).not.toHaveBeenCalled();
  });

  it("should return empty arrays when teamId is empty", async () => {
    const result = await fetchTeamFilterOptions("tok-123", "");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
    expect(mockKeyListCall).not.toHaveBeenCalled();
  });

  it("should return sorted key aliases from fetched keys", async () => {
    mockKeyListCall.mockResolvedValue({
      keys: [{ key_alias: "zeta-key" }, { key_alias: "alpha-key" }, { key_alias: "mid-key" }],
      total_pages: 1,
    });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.keyAliases).toEqual(["alpha-key", "mid-key", "zeta-key"]);
  });

  it("should deduplicate organization IDs across pages", async () => {
    mockKeyListCall
      .mockResolvedValueOnce({
        keys: [{ organization_id: "org-b" }, { organization_id: "org-a" }],
        total_pages: 2,
      })
      .mockResolvedValueOnce({
        keys: [{ organization_id: "org-a" }, { organization_id: "org-c" }],
        total_pages: 2,
      });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.organizationIds).toEqual(["org-a", "org-b", "org-c"]);
  });

  it("should map user IDs with email addresses", async () => {
    mockKeyListCall.mockResolvedValue({
      keys: [
        { user_id: "u1", user: { user_email: "alice@example.com" } },
        { user_id: "u2", user: { user_email: "bob@example.com" } },
      ],
      total_pages: 1,
    });

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result.userIds).toEqual(
      expect.arrayContaining([
        { id: "u1", email: "alice@example.com" },
        { id: "u2", email: "bob@example.com" },
      ]),
    );
  });

  it("should handle API errors gracefully and return empty arrays", async () => {
    mockKeyListCall.mockRejectedValue(new Error("Network error"));

    const result = await fetchTeamFilterOptions("tok-123", "team-1");

    expect(result).toEqual({ keyAliases: [], organizationIds: [], userIds: [] });
  });
});

describe("fetchAllTeams", () => {
  beforeEach(() => {
    mockTeamListCall.mockReset();
  });

  it("forwards the scoping user id to /team/list and returns the rows it answers with", async () => {
    mockTeamListCall.mockResolvedValue([{ team_id: "team-a" }, { team_id: "team-b" }]);

    const teams = await fetchAllTeams("tok-123", null, "member-7");

    expect(mockTeamListCall).toHaveBeenCalledWith("tok-123", null, "member-7");
    expect(teams.map((team) => team.team_id)).toEqual(["team-a", "team-b"]);
  });

  it("sends no user id when the caller is entitled to the broad list", async () => {
    mockTeamListCall.mockResolvedValue([]);

    await fetchAllTeams("tok-123");

    expect(mockTeamListCall).toHaveBeenCalledWith("tok-123", null, null);
  });

  it("keeps the organization filter independent of the scoping user id", async () => {
    mockTeamListCall.mockResolvedValue([]);

    await fetchAllTeams("tok-123", "org-1", "member-7");

    expect(mockTeamListCall).toHaveBeenCalledWith("tok-123", "org-1", "member-7");
  });

  it("returns an empty list without calling the endpoint when there is no access token", async () => {
    expect(await fetchAllTeams(null, null, "member-7")).toEqual([]);
    expect(mockTeamListCall).not.toHaveBeenCalled();
  });
});
